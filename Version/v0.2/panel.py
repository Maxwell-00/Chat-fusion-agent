"""面板结果数据结构：单个面板模型的作答结果（被流式编排 stream_fusion 与裁判共用）。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PanelResult:
    """单个面板模型的作答结果。"""

    model_key: str
    model_name: str
    content: str | None
    latency_ms: int
    error: str | None = None
    # ---- web 模式下额外携带 ----
    sources: list[str] = field(default_factory=list)  # 该模型用到的来源 URL
    steps: list[str] = field(default_factory=list)    # 工具调用轨迹
    tool_calls: int = 0
    evidence: list = field(default_factory=list)      # [(url, snippet), ...] 传给裁判

    @property
    def ok(self) -> bool:
        return self.content is not None and self.error is None
