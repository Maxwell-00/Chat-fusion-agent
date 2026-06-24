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
    ReflexionDone,
    ReflexionStarted,
    StreamError,
    TextDelta,
    ToolFinished,
    ToolStarted,
    TurnComplete,
)
from tool_calling import NativeToolCalling, ToolCallingStrategy
from tools import ToolOutcome, ToolRegistry
from tracing import span

_DEFAULT_SYSTEM = (Path(__file__).parent / "prompts" / "panel_agent.txt").read_text(
    encoding="utf-8"
)
_REFLEXION_PROMPT = (Path(__file__).parent / "prompts" / "reflexion.txt").read_text(
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


def _fmt_evidence(evidence: list, max_items: int = 24) -> str:
    """把 (url, snippet) 证据列成喂给反思的文本（封顶，防撑爆上下文）。"""
    if not evidence:
        return "(无检索证据)"
    items = evidence[:max_items] if max_items > 0 else evidence
    lines = [f"- {snip}\n  {url}" for (url, snip) in items]
    if len(evidence) > len(items):
        lines.append(f"…(另有 {len(evidence) - len(items)} 条略)")
    return "\n".join(lines)


async def _reflect(cfg, question: str, draft: str, evidence: list, rounds: int) -> str:
    """单次（或多轮）修订：对照证据改写初稿并返回修订版；任一轮失败即保留当前内容（至少是初稿）。"""
    current = draft
    ev_text = _fmt_evidence(evidence)
    for _ in range(max(1, rounds)):
        prompt = (
            _REFLEXION_PROMPT.replace("{question}", question)
            .replace("{draft}", current)
            .replace("{evidence}", ev_text)
        )
        try:
            with span("reflexion", model=cfg.key):
                revised = await providers.call_model(cfg, [{"role": "user", "content": prompt}])
        except Exception:
            break
        revised = (revised or "").strip()
        if not revised:
            break
        current = revised
    return current


async def _finish(
    cfg, question, content, sources, steps, calls_made, evidence, n_search, n_fetch,
    *, reflexion: bool, rounds: int,
):
    """成功出答的统一收尾：启用反思且有证据时先自检/修订，再产出 AgentDone。"""
    if reflexion and content and evidence:
        yield ReflexionStarted(cfg.key)
        revised = await _reflect(cfg, question, content, evidence, rounds)
        yield ReflexionDone(cfg.key, changed=(revised != content))
        content = revised
    yield AgentDone(
        cfg.key, content, _dedup(sources), steps, calls_made,
        evidence=evidence, n_search=n_search, n_fetch=n_fetch,
    )


async def stream_agent(
    cfg: ModelConfig,
    question: str,
    registry: ToolRegistry | None,
    strategy: ToolCallingStrategy | None = None,
    *,
    max_steps: int = 4,
    reflexion: bool = False,
    reflexion_rounds: int = 1,
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
            async for ev in _finish(
                cfg, question, turn.content, sources, steps, calls_made,
                evidence, n_search, n_fetch,
                reflexion=reflexion, rounds=reflexion_rounds,
            ):
                yield ev
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
    async for ev in _finish(
        cfg, question, "".join(acc), sources, steps, calls_made,
        evidence, n_search, n_fetch,
        reflexion=reflexion, rounds=reflexion_rounds,
    ):
        yield ev
