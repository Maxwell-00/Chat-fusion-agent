"""面板 agent：让单个面板模型自主使用工具检索，再作答。

循环本身与"工具如何调用"解耦——只依赖 ToolRegistry(有哪些工具) 与
ToolCallingStrategy(怎么调用)。除最终内容外，还收集 evidence(url, snippet)，
向上传给裁判做信源提纯与编号。
"""
from __future__ import annotations

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
