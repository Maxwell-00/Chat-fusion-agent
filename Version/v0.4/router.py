"""语义路由（v0.4 第 2 步）：轻量判断用户问题该走 直答 / 联网 / Fusion。

两段式（先快后准）：
1) 启发式预判：命中明显时效词 → web；命中对比/综合词 → fusion；两者都中或都不中 → 交给分类器。
2) LLM 分类器：用便宜模型读问题，输出 direct / web / fusion 之一；解析失败 / 调用异常兜底 direct。

opt-in：是否启用路由由调用方（CLI /auto、Web「自动」开关）决定；本模块只负责判定，不做动作。
"""
from __future__ import annotations

from pathlib import Path

import providers
from tracing import span

_ROUTER_PROMPT = (Path(__file__).parent / "prompts" / "router.txt").read_text(encoding="utf-8")

# 标签 → 中文回显（调用方据此显式告知用户本次走了哪条路）
ROUTE_LABELS = {"direct": "直接作答", "web": "联网检索", "fusion": "多模型 Fusion", "local": "本地知识库"}

# 启发式信号词：命中即高置信，省掉一次分类器调用
_WEB_SIGNALS = (
    "最新", "今天", "昨天", "今晨", "现在", "目前", "当前", "实时", "近期", "最近",
    "新闻", "价格", "股价", "汇率", "天气", "发布会", "上市", "多少钱", "几号",
    "2024", "2025", "2026",
)
_FUSION_SIGNALS = (
    "比较", "对比", "vs", "优缺点", "利弊", "权衡", "全面", "各方", "综述",
    "哪个更", "哪个好", "区别", "综合分析",
)
# 本地知识库信号（仅 rag 就绪时启用）：问题指向用户自己的资料
_LOCAL_SIGNALS = ("我的", "笔记", "文档", "资料", "知识库", "本地", "上传")


def _heuristic(question: str, allow_local: bool = False) -> str | None:
    """单一明显信号直接判；0 个或 ≥2 个信号都命中 → None，交给分类器。"""
    ql = question.lower()
    hits: list[str] = []
    if allow_local and any(s in ql for s in _LOCAL_SIGNALS):
        hits.append("local")
    if any(s in ql for s in _WEB_SIGNALS):
        hits.append("web")
    if any(s in ql for s in _FUSION_SIGNALS):
        hits.append("fusion")
    return hits[0] if len(hits) == 1 else None


def _parse_label(text: str) -> str:
    """从模型输出里挑出标签；按 fusion>web>local>direct 优先，解析不出兜底 direct。"""
    t = (text or "").strip().lower()
    for label in ("fusion", "web", "local", "direct"):
        if label in t:
            return label
    return "direct"


async def route(question: str, cfg) -> str:
    """返回 'direct'/'web'/'fusion'/'local'（local 仅 rag 就绪时可能）。启发式能定不调模型；异常兜底 direct。"""
    question = (question or "").strip()
    if not question:
        return "direct"
    allow_local = bool(getattr(cfg, "rag_ready", False))
    pre = _heuristic(question, allow_local)
    if pre:
        return pre
    model_key = cfg.router_model or cfg.summary_model or cfg.default_model
    try:
        model = cfg.get(model_key)
        prompt = _ROUTER_PROMPT.replace("{question}", question)
        if allow_local:
            prompt += ("\n补充：local —— 问题涉及用户自己的本地笔记 / 私有文档 / 已上传的知识库内容时，"
                       "输出 local。")
        with span("router", model=model.key):
            text = await providers.call_model(model, [{"role": "user", "content": prompt}])
    except Exception:
        return "direct"
    label = _parse_label(text)
    if label == "local" and not allow_local:
        return "direct"
    return label
