"""工具调用策略（抽象层，流式）。

把"模型如何被告知工具、如何解析工具调用、如何回灌结果"抽象成 ToolCallingStrategy。
当前仅有 NativeToolCalling：用 OpenAI 原生 tools / function-calling 的流式接口。
agent 的循环只依赖本抽象，不关心底层用的是哪种机制。
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

import providers
from stream_events import ParsedCall, StreamError, TextDelta, TurnComplete
from tools import ToolOutcome, ToolRegistry


class ToolCallingStrategy(ABC):
    @abstractmethod
    def system_prompt(self, base_system: str, registry: ToolRegistry) -> str:
        """构造系统提示。"""

    @abstractmethod
    def tool_result_message(self, call: ParsedCall, outcome: ToolOutcome) -> dict:
        """把工具执行结果包装成可追加进对话的消息。"""

    @abstractmethod
    async def stream_turn(self, cfg, messages, registry):
        """流式跑一回合：yield TextDelta / StreamError，最后 yield 一个 TurnComplete。"""
        ...

    @abstractmethod
    async def stream_final(self, cfg, messages, registry):
        """达到工具上限后流式强制作答：逐字 yield 文本。"""
        ...


class NativeToolCalling(ToolCallingStrategy):
    """OpenAI 原生 function-calling（流式）。"""

    def system_prompt(self, base_system: str, registry: ToolRegistry) -> str:
        return base_system  # 工具通过 tools 参数下发，无需写进提示词

    def tool_result_message(self, call: ParsedCall, outcome: ToolOutcome) -> dict:
        return {"role": "tool", "tool_call_id": call.id, "content": outcome.text}

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
