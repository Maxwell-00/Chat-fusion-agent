"""合成层：基于裁判裁决 + 各面板原始回答，产出完整、连贯、带内联角标的最终研报。

事实判断以裁判为准、内联角标只用裁判的 source_mapping（禁止自创编号）；面板原文仅作
充实细节的素材（保留具体数据/名称/清单等），裁判未采信的内容谨慎或不用。
参考来源列表由 CLI 依据 source_mapping 确定性渲染（不在此重复罗列）。
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


SYNTH_PANEL_CHAR_LIMIT = 8000  # 单个面板原文喂给合成器的上限（保留细节又防爆上下文；0=不限）


def _clip(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n…(中间略 {len(text) - head - tail} 字)…\n{text[-tail:]}"


def _panels_text(panels) -> str:
    if not panels:
        return "(无面板原文)"
    parts = []
    for p in panels:
        if getattr(p, "ok", True) and getattr(p, "content", None):
            parts.append(
                f"【{p.model_key} · {p.model_name}】\n{_clip(p.content, SYNTH_PANEL_CHAR_LIMIT)}"
            )
    return "\n\n".join(parts) if parts else "(无面板原文)"


def _render(user_prompt: str, analysis: JudgeAnalysis, panels) -> str:
    return (
        _PROMPT.replace("{user_prompt}", user_prompt)
        .replace("{judge_verdict}", _verdict_text(analysis))
        .replace("{confirmed_facts}", _facts_text(analysis))
        .replace("{debunked_rumors}", _rumors_text(analysis))
        .replace("{source_mapping}", _mapping_text(analysis))
        .replace("{panel_answers}", _panels_text(panels))
    )


async def stream_synthesize(
    user_prompt: str, analysis: JudgeAnalysis, panels, cfg: ModelConfig
):
    """流式合成：逐字 yield 最终研报。panels 为各面板原始回答(素材)，事实仍以裁判为准。"""
    messages = [{"role": "user", "content": _render(user_prompt, analysis, panels)}]
    async for delta in providers.stream(cfg, messages):
        yield delta
