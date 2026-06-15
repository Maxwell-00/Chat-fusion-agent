"""编排层：把 panel -> judge -> synthesize 串成一条龙。

整条链路是单趟（无自循环），所以不存在无限循环风险。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import providers
from agent import stream_agent
from config import AppConfig
from judge import JudgeAnalysis, judge
from panel import PanelResult, run_panel, run_panel_web
from stream_events import (
    AgentDegraded,
    AgentDone,
    FusionDone,
    JudgeFinished,
    JudgeStarted,
    PanelFinished,
    PanelStarted,
    StageError,
    SynthesisDelta,
    SynthesisStarted,
    ToolFinished,
    ToolStarted,
)
from synthesizer import stream_synthesize, synthesize


@dataclass
class FusionResult:
    """一次 fusion 的完整产物，便于 verbose 展示与后续扩展。"""

    final: str
    panel: list[PanelResult]
    analysis: JudgeAnalysis


async def run_fusion(
    prompt: str,
    panel_keys: list[str],
    cfg: AppConfig,
    *,
    web: bool = False,
    registry=None,
    strategy=None,
) -> FusionResult:
    """对单条问题执行一次完整 fusion。

    v1 约定：fusion 只针对当前这一条问题，不带入多轮历史，保持简单可复现。
    web=True 时面板模型以 agent 方式联网检索后作答(需提供 registry)。
    """
    cfgs = [cfg.get(k) for k in panel_keys]
    if web:
        panel = await run_panel_web(
            cfgs, prompt, registry, strategy, max_steps=cfg.web_max_steps
        )
    else:
        panel = await run_panel(cfgs, [{"role": "user", "content": prompt}])

    ok = [p for p in panel if p.ok]
    if not ok:
        errors = "; ".join(f"{p.model_key}: {p.error}" for p in panel)
        raise RuntimeError(f"所有面板模型都调用失败：{errors}")

    evidence = _aggregate_evidence(ok)
    analysis = await judge(prompt, ok, cfg.get(cfg.judge_model), evidence)
    final = await synthesize(prompt, analysis, cfg.get(cfg.synthesizer_model))
    return FusionResult(final=final, panel=panel, analysis=analysis)


def _dedup_sources(panel: list[PanelResult]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in panel:
        for u in p.sources:
            if u and u not in seen:
                seen.add(u)
                out.append(u)
    return out


def _aggregate_evidence(panel: list[PanelResult]) -> list:
    """汇总各面板的 (url, snippet) 证据，按 url 去重后交给裁判提纯/编号。"""
    seen: set[str] = set()
    out: list = []
    for p in panel:
        for item in getattr(p, "evidence", []) or []:
            url = item[0]
            if url and url not in seen:
                seen.add(url)
                out.append(item)
    return out


async def stream_fusion(
    prompt: str,
    panel_keys: list[str],
    cfg: AppConfig,
    *,
    web: bool = True,
    registry=None,
    strategy=None,
):
    """流式 fusion：并行面板（只上报状态事件）-> 裁判（非流式）-> 合成（流式）。

    web=True：面板以 agent 方式联网检索（stream_agent）。
    web=False：面板各模型并行直接作答（无工具），同样只上报状态、不流式面板文本。
    用 asyncio.Queue 把多个并行面板事件 fan-in 到一条流；合成阶段逐字 yield。
    """
    yield PanelStarted(panel_keys)
    panel_cfgs = [cfg.get(k) for k in panel_keys]
    q: asyncio.Queue = asyncio.Queue()
    results: dict[str, PanelResult] = {}

    async def run_web(ci):
        t0 = time.perf_counter()
        try:
            async for ev in stream_agent(
                ci, prompt, registry, strategy, max_steps=cfg.web_max_steps
            ):
                if isinstance(ev, AgentDone):
                    results[ci.key] = PanelResult(
                        ci.key,
                        ci.name,
                        ev.content,
                        int((time.perf_counter() - t0) * 1000),
                        sources=ev.sources,
                        steps=ev.steps,
                        tool_calls=ev.tool_calls,
                        evidence=ev.evidence,
                    )
                await q.put(ev)
        except Exception as e:  # 兜底：单个面板崩了也不影响整体
            results[ci.key] = PanelResult(
                ci.key, ci.name, None, int((time.perf_counter() - t0) * 1000), error=str(e)
            )
        await q.put(("__done__", ci.key))

    async def run_offline(ci):
        t0 = time.perf_counter()
        try:
            content = await providers.call_model(ci, [{"role": "user", "content": prompt}])
            results[ci.key] = PanelResult(
                ci.key, ci.name, content, int((time.perf_counter() - t0) * 1000)
            )
            await q.put(AgentDone(ci.key, content))
        except Exception as e:
            results[ci.key] = PanelResult(
                ci.key, ci.name, None, int((time.perf_counter() - t0) * 1000), error=str(e)
            )
            await q.put(AgentDegraded(ci.key, str(e)))
            await q.put(AgentDone(ci.key, "", degraded=True))
        await q.put(("__done__", ci.key))

    run_one = run_web if web else run_offline
    tasks = [asyncio.create_task(run_one(c)) for c in panel_cfgs]
    done = 0
    while done < len(tasks):
        ev = await q.get()
        if isinstance(ev, tuple) and ev and ev[0] == "__done__":
            done += 1
            continue
        # 面板阶段只转发状态/完成类事件（丢弃 TextDelta，避免多流交错）
        if isinstance(ev, (ToolStarted, ToolFinished, AgentDegraded, AgentDone)):
            yield ev
    await asyncio.gather(*tasks, return_exceptions=True)

    panel = [results[k] for k in panel_keys if k in results]
    yield PanelFinished(panel)

    ok = [p for p in panel if p.ok]
    if not ok:
        yield StageError("panel", "所有面板模型都失败")
        return

    evidence = _aggregate_evidence(ok)
    yield JudgeStarted()
    try:
        analysis = await judge(prompt, ok, cfg.get(cfg.judge_model), evidence)
    except Exception as e:  # 裁判失败也降级，不中断
        yield StageError("judge", str(e))
        analysis = JudgeAnalysis(raw="(裁判阶段失败，基于面板回答直接合成)")
    yield JudgeFinished(analysis)

    yield SynthesisStarted()
    try:
        async for piece in stream_synthesize(
            prompt, analysis, cfg.get(cfg.synthesizer_model)
        ):
            yield SynthesisDelta(piece)
    except Exception as e:
        yield StageError("synth", str(e))
    # 参考来源用裁判提纯后的 source_mapping（与正文 [n] 一致）；裁判没给则回退原始来源
    sources = analysis.ordered_sources() or _dedup_sources(panel)
    yield FusionDone(sources)
