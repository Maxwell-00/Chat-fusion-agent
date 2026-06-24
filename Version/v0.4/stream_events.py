"""流式事件模型。

把流水线各阶段产出的事件统一成一组小数据类，供 agent / fusion 向上 yield，
chat 据此渲染。本模块不依赖项目内其它模块，避免循环导入。
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ======================= agent 级（单个面板 agent 向上冒泡）=======================
@dataclass
class TextDelta:
    """状态 A/C：模型输出的文本分片。"""
    text: str


@dataclass
class StreamError:
    """流式过程中出现异常（用于触发降级，不抛出）。"""
    detail: str


@dataclass
class ToolStarted:
    """状态 B：某面板模型开始执行一个工具。"""
    model: str
    name: str
    args: dict


@dataclass
class ToolFinished:
    model: str
    name: str
    n_sources: int
    ok: bool = True


@dataclass
class AgentDegraded:
    model: str
    reason: str


@dataclass
class ReflexionStarted:
    """agent 出初稿后开始对照证据自检 / 修订（仅 reflexion 启用时）。"""
    model: str


@dataclass
class ReflexionDone:
    """自检修订结束；changed 表示修订是否改动了初稿。"""
    model: str
    changed: bool = False


@dataclass
class AgentDone:
    """某面板 agent 结束，带回最终内容与来源/轨迹/证据（供上层组装 PanelResult）。"""
    model: str
    content: str
    sources: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    tool_calls: int = 0
    degraded: bool = False
    evidence: list = field(default_factory=list)   # [(url, snippet), ...] 传给裁判
    n_search: int = 0
    n_fetch: int = 0


# ======================= turn 级（策略层缓冲后产出）=======================
@dataclass
class ParsedCall:
    """分片缓冲拼接 + 解析后的一次工具调用。args=None 表示 JSON 解析失败。"""
    id: str
    name: str | None
    args: dict | None
    raw: str


@dataclass
class TurnComplete:
    """一次模型回合的最终结果。"""
    assistant_message: dict
    tool_calls: list[ParsedCall]
    content: str
    finish_reason: str | None


# ======================= pipeline 级（fusion 向 chat 产出）=======================
@dataclass
class PanelStarted:
    models: list[str]


@dataclass
class PanelFinished:
    results: list  # list[PanelResult]


@dataclass
class JudgeStarted:
    pass


@dataclass
class JudgeFinished:
    analysis: object = None   # 携带 JudgeAnalysis 供 verbose 展示


@dataclass
class SynthesisStarted:
    pass


@dataclass
class SynthesisDelta:
    text: str


@dataclass
class FusionDone:
    sources: list[str]


@dataclass
class StageError:
    stage: str
    detail: str
