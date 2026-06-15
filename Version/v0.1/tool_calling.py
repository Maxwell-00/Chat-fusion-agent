"""工具调用策略（抽象层）。

把"模型如何被告知工具、如何解析工具调用、如何回灌结果"抽象成 ToolCallingStrategy。
- NativeToolCalling：用 OpenAI 原生 tools / function-calling（当前默认，稳定、低延迟）。
- HermesToolCalling：预留接口，未来兼容不支持原生 tools 的本地小模型时，在此实现
  Hermes 风格的 <tool_call> 文本协议解析。

agent 的循环只依赖本抽象，不关心底层用的是哪种机制。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import providers
from config import ModelConfig
from stream_events import ParsedCall, StreamError, TextDelta, TurnComplete
from tools import ToolOutcome, ToolRegistry


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ModelTurn:
    """一次模型回合：要追加的 assistant 消息 + 本回合要求的工具调用（空=最终回答）。"""

    assistant_message: dict
    tool_calls: list[ToolCall]
    content: str


class ToolCallingStrategy(ABC):
    @abstractmethod
    def system_prompt(self, base_system: str, registry: ToolRegistry) -> str:
        """构造系统提示（Hermes 策略会把工具说明注入这里）。"""

    @abstractmethod
    async def model_turn(
        self, cfg: ModelConfig, messages: list[dict], registry: ToolRegistry
    ) -> ModelTurn:
        """跑一次模型，解析出本回合的工具调用或最终回答。"""

    @abstractmethod
    def tool_result_message(self, call: ToolCall, outcome: ToolOutcome) -> dict:
        """把工具执行结果包装成可追加进对话的消息。"""

    @abstractmethod
    async def final_answer(
        self, cfg: ModelConfig, messages: list[dict], registry: ToolRegistry
    ) -> str:
        """达到工具上限后，强制模型基于已有信息直接作答。"""

    # ---- v3 流式（默认不支持；子类覆盖。未来 Hermes 在此实现文本分片解析）----
    async def stream_turn(self, cfg, messages, registry):
        """流式跑一回合：yield TextDelta / StreamError，最后 yield 一个 TurnComplete。"""
        raise NotImplementedError("该策略不支持流式 stream_turn")
        if False:  # pragma: no cover —— 让本方法成为 async generator
            yield

    async def stream_final(self, cfg, messages, registry):
        """达到上限后流式强制作答：逐字 yield 文本。"""
        raise NotImplementedError("该策略不支持流式 stream_final")
        if False:  # pragma: no cover
            yield


def _assistant_to_dict(msg) -> dict:
    """把 SDK 的 assistant message 序列化回可再次发送的 dict。"""
    d: dict = {"role": "assistant", "content": msg.content or ""}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in tcs
        ]
    return d


class NativeToolCalling(ToolCallingStrategy):
    """OpenAI 原生 function-calling。"""

    def system_prompt(self, base_system: str, registry: ToolRegistry) -> str:
        return base_system  # 工具通过 tools 参数下发，无需写进提示词

    async def model_turn(self, cfg, messages, registry) -> ModelTurn:
        msg = await providers.call_with_tools(cfg, messages, registry.schemas())
        calls: list[ToolCall] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            calls.append(ToolCall(tc.id, tc.function.name, args))
        return ModelTurn(_assistant_to_dict(msg), calls, msg.content or "")

    def tool_result_message(self, call: ToolCall, outcome: ToolOutcome) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": outcome.text}

    async def final_answer(self, cfg, messages, registry) -> str:
        # 仍带工具定义(满足部分厂商校验)，但 tool_choice=none 强制直接作答。
        msg = await providers.call_with_tools(
            cfg, messages, registry.schemas(), tool_choice="none"
        )
        return msg.content or "(未能生成最终回答)"

    # ---------- v3 流式 ----------
    async def stream_turn(self, cfg, messages, registry):
        """流式回合 + 分片缓冲。

        要点(均来自真实兼容性坑)：
        - content 与 tool_calls 独立累积(单个 delta 可能两者都有)。
        - tool_call 的 arguments 跨多个 chunk，按 index 拼接后再解析。
        - id 只在首片出现；name 可能晚到 —— 出现时才写入。
        - finish_reason 不可全信 —— 以"缓冲区非空"为触发工具的依据。
        """
        content_parts: list[str] = []
        tool_buf: dict[int, dict] = {}
        finish = None
        try:
            async for chunk in providers.stream_with_tools(cfg, messages, registry.schemas()):
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                delta = choice.delta
                if delta and delta.content:
                    content_parts.append(delta.content)
                    yield TextDelta(delta.content)
                for tc in (getattr(delta, "tool_calls", None) or []):
                    slot = tool_buf.setdefault(tc.index, {"id": None, "name": None, "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if fn.name:
                            slot["name"] = fn.name
                        if fn.arguments:
                            slot["args"] += fn.arguments  # ★ 分片拼接
                if choice.finish_reason:
                    finish = choice.finish_reason
        except Exception as e:
            yield StreamError(str(e))
            return

        content = "".join(content_parts)
        calls: list[ParsedCall] = []
        tool_payload: list[dict] = []
        for idx in sorted(tool_buf):  # ★ 缓冲非空即触发，不只看 finish_reason
            slot = tool_buf[idx]
            cid = slot["id"] or f"call_{idx}"
            raw = slot["args"] or "{}"
            try:
                args = json.loads(raw)            # 拼接完整后才解析
            except json.JSONDecodeError:
                args = None                        # 解析失败 → 交由 agent 降级
            calls.append(ParsedCall(cid, slot["name"], args, raw))
            tool_payload.append(
                {"id": cid, "type": "function",
                 "function": {"name": slot["name"] or "", "arguments": raw}}
            )

        assistant_message: dict = {"role": "assistant", "content": content or ""}
        if tool_payload:
            assistant_message["tool_calls"] = tool_payload
        yield TurnComplete(assistant_message, calls, content, finish)

    async def stream_final(self, cfg, messages, registry):
        async for chunk in providers.stream_with_tools(
            cfg, messages, registry.schemas(), tool_choice="none"
        ):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


class HermesToolCalling(ToolCallingStrategy):
    """预留：Hermes 风格 <tool_call> 文本协议，面向不支持原生 tools 的模型。

    未来实现要点：
      - system_prompt：注入 registry.describe() 的工具清单 + 输出格式约定
        （形如 <tool_call>{"name": ..., "arguments": {...}}</tool_call>）。
      - model_turn：用 providers.call_model 取纯文本，正则解析 <tool_call> 块。
      - tool_result_message：以 <tool_response>...</tool_response> 文本回灌（role=user）。
      - final_answer：提示"不要再调用工具，直接作答"。
    """

    _MSG = (
        "HermesToolCalling 尚未实现：当前版本默认用 NativeToolCalling。"
        "未来兼容不支持原生 tools 的本地小模型时，在此实现 <tool_call> 文本解析。"
    )

    def system_prompt(self, base_system: str, registry: ToolRegistry) -> str:
        raise NotImplementedError(self._MSG)

    async def model_turn(self, cfg, messages, registry) -> ModelTurn:
        raise NotImplementedError(self._MSG)

    def tool_result_message(self, call: ToolCall, outcome: ToolOutcome) -> dict:
        raise NotImplementedError(self._MSG)

    async def final_answer(self, cfg, messages, registry) -> str:
        raise NotImplementedError(self._MSG)
