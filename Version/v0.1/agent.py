"""面板 agent：让单个面板模型自主使用工具检索，再作答。

循环本身与"工具如何调用"解耦——只依赖 ToolRegistry(有哪些工具) 与
ToolCallingStrategy(怎么调用)。除最终内容外，还收集 evidence(url, snippet)，
向上传给裁判做信源提纯与编号。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import providers
from config import ModelConfig
from stream_events import (
    AgentDegraded,
    AgentDone,
    StreamError,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnComplete,
)
from tool_calling import NativeToolCalling, ToolCallingStrategy
from tools import ToolOutcome, ToolRegistry

_DEFAULT_SYSTEM = (Path(__file__).parent / "prompts" / "panel_agent.txt").read_text(
    encoding="utf-8"
)


@dataclass
class AgentResult:
    content: str
    steps: list[str] = field(default_factory=list)    # 人类可读的工具调用轨迹
    sources: list[str] = field(default_factory=list)  # 去重后的来源 URL
    tool_calls: int = 0
    evidence: list = field(default_factory=list)      # [(url, snippet), ...]


def _dedup(xs: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _fmt_step(name: str, args: dict, outcome: ToolOutcome) -> str:
    key = args.get("query") or args.get("url") or ""
    n = len(outcome.urls)
    return f"{name}({key!r}) -> {n} 个来源" if n else f"{name}({key!r})"


async def run_agent(
    cfg: ModelConfig,
    question: str,
    registry: ToolRegistry | None,
    strategy: ToolCallingStrategy | None = None,
    *,
    system_prompt: str | None = None,
    max_steps: int = 4,
) -> AgentResult:
    """非流式版本（v2 路径 / 离线 web fusion 用）。"""
    if registry is None or not cfg.supports_tools:
        content = await providers.call_model(
            cfg, [{"role": "user", "content": question}]
        )
        return AgentResult(content=content)

    strategy = strategy or NativeToolCalling()
    base_system = system_prompt or _DEFAULT_SYSTEM
    messages: list[dict] = [
        {"role": "system", "content": strategy.system_prompt(base_system, registry)},
        {"role": "user", "content": question},
    ]
    steps: list[str] = []
    sources: list[str] = []
    evidence: list = []
    total_calls = 0

    for _ in range(max_steps):
        turn = await strategy.model_turn(cfg, messages, registry)
        messages.append(turn.assistant_message)

        if not turn.tool_calls:
            return AgentResult(turn.content, steps, _dedup(sources), total_calls, evidence)

        for call in turn.tool_calls:
            total_calls += 1
            outcome = await registry.execute(call.name, call.args)
            sources.extend(outcome.urls)
            evidence.extend(outcome.evidence)
            steps.append(_fmt_step(call.name, call.args, outcome))
            messages.append(strategy.tool_result_message(call, outcome))

    messages.append(
        {
            "role": "user",
            "content": "已达到工具调用上限，请基于已获取的信息直接给出最终回答，不要再调用工具。",
        }
    )
    content = await strategy.final_answer(cfg, messages, registry)
    return AgentResult(content, steps, _dedup(sources), total_calls, evidence)


async def stream_agent(
    cfg: ModelConfig,
    question: str,
    registry: ToolRegistry | None,
    strategy: ToolCallingStrategy | None = None,
    *,
    max_steps: int = 4,
):
    """流式版本（状态机见 v3 设计 §6）。

    yield：TextDelta(A/C 文本) / ToolStarted / ToolFinished / AgentDegraded，
    以 AgentDone(带 content/sources/evidence/计数) 收尾。任何异常都降级，不崩流。
    """
    if registry is None or not cfg.supports_tools:
        acc: list[str] = []
        try:
            async for t in providers.stream(cfg, [{"role": "user", "content": question}]):
                acc.append(t)
                yield TextDelta(t)
        except Exception as e:
            yield AgentDegraded(cfg.key, str(e))
        yield AgentDone(cfg.key, "".join(acc), [], [], 0)
        return

    strategy = strategy or NativeToolCalling()
    base_system = _DEFAULT_SYSTEM
    messages: list[dict] = [
        {"role": "system", "content": strategy.system_prompt(base_system, registry)},
        {"role": "user", "content": question},
    ]
    sources: list[str] = []
    evidence: list = []
    steps: list[str] = []
    calls_made = 0
    n_search = 0
    n_fetch = 0

    for _ in range(max_steps):
        turn = None
        errored = False
        async for ev in strategy.stream_turn(cfg, messages, registry):
            if isinstance(ev, TextDelta):
                yield ev
            elif isinstance(ev, StreamError):
                errored = True
                yield AgentDegraded(cfg.key, ev.detail)
                break
            elif isinstance(ev, TurnComplete):
                turn = ev

        if errored:
            acc = []
            try:
                async for t in providers.stream(
                    cfg,
                    [
                        {"role": "system", "content": base_system},
                        {"role": "user", "content": question},
                    ],
                ):
                    acc.append(t)
                    yield TextDelta(t)
            except Exception:
                pass
            yield AgentDone(
                cfg.key, "".join(acc), _dedup(sources), steps, calls_made,
                degraded=True, evidence=evidence, n_search=n_search, n_fetch=n_fetch,
            )
            return

        messages.append(turn.assistant_message)

        if not turn.tool_calls:
            yield AgentDone(
                cfg.key, turn.content, _dedup(sources), steps, calls_made,
                evidence=evidence, n_search=n_search, n_fetch=n_fetch,
            )
            return

        for call in turn.tool_calls:
            calls_made += 1
            if call.name == "web_search":
                n_search += 1
            elif call.name == "web_fetch":
                n_fetch += 1
            if call.args is None:
                yield AgentDegraded(cfg.key, f"{call.name} 参数解析失败")
                outcome = ToolOutcome(
                    "工具参数 JSON 解析失败，已忽略本次调用，请基于已有信息作答。"
                )
            else:
                yield ToolStarted(cfg.key, call.name or "", call.args)
                outcome = await registry.execute(call.name, call.args)
                sources.extend(outcome.urls)
                evidence.extend(outcome.evidence)
                steps.append(_fmt_step(call.name or "", call.args, outcome))
                yield ToolFinished(cfg.key, call.name or "", len(outcome.urls), True)
            messages.append(strategy.tool_result_message(call, outcome))

    messages.append(
        {
            "role": "user",
            "content": "已达到工具调用上限，请基于已获取的信息直接给出最终回答，不要再调用工具。",
        }
    )
    acc = []
    try:
        async for t in strategy.stream_final(cfg, messages, registry):
            acc.append(t)
            yield TextDelta(t)
    except Exception as e:
        yield AgentDegraded(cfg.key, f"最终作答失败：{e}")
    yield AgentDone(
        cfg.key, "".join(acc), _dedup(sources), steps, calls_made,
        evidence=evidence, n_search=n_search, n_fetch=n_fetch,
    )
