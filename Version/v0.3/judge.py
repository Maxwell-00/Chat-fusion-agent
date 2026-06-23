"""裁判层：交叉验证 + 信源提纯与编号（证据链在裁判阶段闭环）。

输入：原问题 + 各模型回答 + 面板检索到的信源池(snippet, url)。
输出固定 JSON：
  verdict          权衡后的核心结论
  source_mapping   {"[1]": url, ...}  —— 去重/剔除低质社媒/按权威重排后从 [1] 连续编号
  confirmed_facts  [{fact, citations:["[1]","[2]"]}]
  debunked_rumors  存疑的自媒体黑话/机翻梗/未证实代号
  blind_spots      所有模型都未覆盖但重要的维度
解析失败重试一次，仍失败兜底为原文，保证流水线不中断。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import providers
from config import ModelConfig
from panel import PanelResult

_PROMPT = (Path(__file__).parent / "prompts" / "judge.txt").read_text(encoding="utf-8")


@dataclass
class ConfirmedFact:
    fact: str
    citations: list[str] = field(default_factory=list)


@dataclass
class JudgeAnalysis:
    verdict: str = ""
    source_mapping: dict[str, str] = field(default_factory=dict)
    confirmed_facts: list[ConfirmedFact] = field(default_factory=list)
    debunked_rumors: list[str] = field(default_factory=list)
    blind_spots: list[str] = field(default_factory=list)
    raw: str = ""  # 解析失败时保留原文，兜底用

    @property
    def parsed(self) -> bool:
        return bool(self.verdict or self.confirmed_facts or self.source_mapping)

    def ordered_sources(self) -> list[str]:
        """按 [1][2]… 顺序返回 URL 列表（用于 CLI 渲染参考来源）。"""
        def num(k: str) -> int:
            m = re.search(r"\d+", k)
            return int(m.group()) if m else 0

        return [self.source_mapping[k] for k in sorted(self.source_mapping, key=num)]

    @classmethod
    def from_dict(cls, d: dict) -> "JudgeAnalysis":
        def as_list(v):
            if v is None:
                return []
            return [str(x) for x in v] if isinstance(v, list) else [str(v)]

        facts: list[ConfirmedFact] = []
        for it in d.get("confirmed_facts") or []:
            if isinstance(it, dict):
                cites = it.get("citations") or []
                cites = cites if isinstance(cites, list) else [str(cites)]
                facts.append(ConfirmedFact(str(it.get("fact", "")), [str(c) for c in cites]))
            elif isinstance(it, str):
                facts.append(ConfirmedFact(it, []))

        sm = d.get("source_mapping") or {}
        sm = {str(k): str(v) for k, v in sm.items()} if isinstance(sm, dict) else {}

        return cls(
            verdict=str(d.get("verdict", "")),
            source_mapping=sm,
            confirmed_facts=facts,
            debunked_rumors=as_list(d.get("debunked_rumors")),
            blind_spots=as_list(d.get("blind_spots")),
        )


def _clip(text: str, limit: int) -> str:
    """超长文本头+尾保留、丢中间，保留开头主张与结尾结论；limit<=0 表示不限。"""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n…(此处省略 {omitted} 字)…\n{text[-tail:]}"


def _format_answers(results: list[PanelResult], answer_char_limit: int) -> str:
    return "\n\n".join(
        f"[{r.model_key} · {r.model_name}]\n{_clip(r.content or '', answer_char_limit)}"
        for r in results
    )


def _format_evidence(evidence: list, max_items: int) -> str:
    if not evidence:
        return "(无检索信源；本轮可能未联网或检索为空)"
    items = evidence[:max_items] if max_items > 0 else evidence
    lines = [f"- {snip}\n  {url}" for (url, snip) in items]
    if len(evidence) > len(items):
        lines.append(f"…(另有 {len(evidence) - len(items)} 条信源略)")
    return "\n".join(lines)


def _extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _render(
    user_prompt: str,
    results: list[PanelResult],
    evidence: list,
    answer_char_limit: int,
    evidence_max_items: int,
) -> str:
    return (
        _PROMPT.replace("{user_prompt}", user_prompt)
        .replace("{labeled_answers}", _format_answers(results, answer_char_limit))
        .replace("{evidence}", _format_evidence(evidence, evidence_max_items))
    )


async def judge(
    user_prompt: str,
    results: list[PanelResult],
    cfg: ModelConfig,
    evidence: list | None = None,
    *,
    answer_char_limit: int = 4000,
    evidence_max_items: int = 24,
) -> JudgeAnalysis:
    evidence = evidence or []
    messages = [
        {
            "role": "user",
            "content": _render(
                user_prompt, results, evidence, answer_char_limit, evidence_max_items
            ),
        }
    ]

    try:
        text = await providers.call_model(
            cfg, messages, response_format={"type": "json_object"}
        )
    except Exception:
        text = await providers.call_model(cfg, messages)

    data = _extract_json(text)
    if data is None:
        retry = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "你的上一条输出不是合法 JSON。请只输出符合要求的 JSON 对象，不要任何额外文字。",
            },
        ]
        try:
            text = await providers.call_model(cfg, retry)
            data = _extract_json(text)
        except Exception:
            data = None

    if data is None:
        return JudgeAnalysis(raw=text)
    analysis = JudgeAnalysis.from_dict(data)
    analysis.raw = text
    return analysis
