"""面板层：把同一个问题并行分发给一组模型。"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import providers
from config import ModelConfig


@dataclass
class PanelResult:
    """单个面板模型的作答结果。"""

    model_key: str
    model_name: str
    content: str | None
    latency_ms: int
    error: str | None = None
    # ---- v2 web 模式下额外携带 ----
    sources: list[str] = field(default_factory=list)  # 该模型用到的来源 URL
    steps: list[str] = field(default_factory=list)    # 工具调用轨迹
    tool_calls: int = 0
    evidence: list = field(default_factory=list)      # [(url, snippet), ...] 传给裁判

    @property
    def ok(self) -> bool:
        return self.content is not None and self.error is None


async def run_panel(cfgs: list[ModelConfig], messages: list[dict]) -> list[PanelResult]:
    """并行调用面板模型；任一模型失败不影响其他，单独记录时延与错误。"""

    async def one(cfg: ModelConfig) -> PanelResult:
        t0 = time.perf_counter()
        try:
            content = await providers.call_model(cfg, messages)
            return PanelResult(
                cfg.key, cfg.name, content, int((time.perf_counter() - t0) * 1000)
            )
        except Exception as e:  # 面板内部不抛出，错误以数据形式向上返回
            return PanelResult(
                cfg.key,
                cfg.name,
                None,
                int((time.perf_counter() - t0) * 1000),
                error=str(e),
            )

    return list(await asyncio.gather(*(one(c) for c in cfgs)))


async def run_panel_web(
    cfgs: list[ModelConfig],
    question: str,
    registry,
    strategy=None,
    *,
    max_steps: int = 4,
) -> list[PanelResult]:
    """以 agent 方式并行运行面板模型：每个模型自主用工具检索后作答。"""
    import agent  # 延迟导入，避免模块级循环依赖

    async def one(cfg: ModelConfig) -> PanelResult:
        t0 = time.perf_counter()
        try:
            res = await agent.run_agent(
                cfg, question, registry, strategy, max_steps=max_steps
            )
            return PanelResult(
                cfg.key,
                cfg.name,
                res.content,
                int((time.perf_counter() - t0) * 1000),
                sources=res.sources,
                steps=res.steps,
                tool_calls=res.tool_calls,
                evidence=res.evidence,
            )
        except Exception as e:
            return PanelResult(
                cfg.key,
                cfg.name,
                None,
                int((time.perf_counter() - t0) * 1000),
                error=str(e),
            )

    return list(await asyncio.gather(*(one(c) for c in cfgs)))

