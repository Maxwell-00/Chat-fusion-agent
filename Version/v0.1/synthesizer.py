"""合成层：消费裁判裁决，产出简明、连贯、带内联角标的最终回答。

合成器只“消费”裁判提供的 verdict / confirmed_facts(带 citations) / source_mapping，
严格按 citations 内联角标，禁止自创编号。参考来源列表由 CLI 依据 source_mapping
确定性渲染（不在此重复罗列）。
"""
from __future__ import annotations

import re
from pathlib import Path

import providers
from config import ModelConfig
from judge import JudgeAnalysis

_PROMPT = (Path(__file__).parent / "prompts" / "synthesizer.txt").read_text(
    encoding="utf-8"
)


def _verdict_text(a: JudgeAnalysis) -> str:
    if a.verdict:
        return a.verdict
    return a.raw or "(裁判未给出明确结论，请基于上下文谨慎作答)"


def _facts_text(a: JudgeAnalysis) -> str:
    if not a.confirmed_facts:
        return "(无已确认事实；可基于裁判结论谨慎作答，但不要编造角标)"
    lines = []
    for f in a.confirmed_facts:
        cites = "".join(f.citations)
        lines.append(f"- {f.fact}  citations: {cites or '(无)'}")
    return "\n".join(lines)


def _rumors_text(a: JudgeAnalysis) -> str:
    return "\n".join(f"- {r}" for r in a.debunked_rumors) if a.debunked_rumors else "(无)"


def _mapping_text(a: JudgeAnalysis) -> str:
    if not a.source_mapping:
        return "(无可用信源编号，正文不要使用任何角标)"

    def num(k: str) -> int:
        m = re.search(r"\d+", k)
        return int(m.group()) if m else 0

    return "\n".join(f"{k} {a.source_mapping[k]}" for k in sorted(a.source_mapping, key=num))


def _render(user_prompt: str, analysis: JudgeAnalysis) -> str:
    return (
        _PROMPT.replace("{user_prompt}", user_prompt)
        .replace("{judge_verdict}", _verdict_text(analysis))
        .replace("{confirmed_facts}", _facts_text(analysis))
        .replace("{debunked_rumors}", _rumors_text(analysis))
        .replace("{source_mapping}", _mapping_text(analysis))
    )


async def synthesize(
    user_prompt: str, analysis: JudgeAnalysis, cfg: ModelConfig
) -> str:
    messages = [{"role": "user", "content": _render(user_prompt, analysis)}]
    return await providers.call_model(cfg, messages)


async def stream_synthesize(
    user_prompt: str, analysis: JudgeAnalysis, cfg: ModelConfig
):
    """流式合成：逐字 yield 最终回答（v3 用，状态 C）。"""
    messages = [{"role": "user", "content": _render(user_prompt, analysis)}]
    async for delta in providers.stream(cfg, messages):
        yield delta
