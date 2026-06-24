"""可观测性（Phoenix / OpenInference）：自动追踪所有模型调用。

零侵入：providers.py 是唯一网络出口、全部走 openai SDK，这里一次性 instrument openai，
所有模型调用即被追踪（token / 延迟 / prompt / 响应）。core 一行不用改。

默认关闭——只有设了环境变量 FUSION_TRACING 才启用；未启用时 init_tracing() 为 no-op，
不引入任何追踪开销，也不依赖追踪库是否安装。

本地查看 trace（一条命令，无需另开终端）：
    pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
                openinference-instrumentation-openai
    FUSION_TRACING=1 python app.py      # 或 python main.py
    # ↑ 开追踪时会自动在进程内拉起本地 Phoenix（默认 http://localhost:6006）；
    #   Web UI 侧栏的「🔍 打开监控面板 (Phoenix)」可直接跳转。
    # 指向外部/自定义 OTLP 端点：PHOENIX_OTLP=http://host:6006/v1/traces（设了就不自动起本地 Phoenix）
    # 不想自动起本地 Phoenix：PHOENIX_NO_LAUNCH=1（仍可自己 `python -m phoenix.server.main serve`）
"""
from __future__ import annotations

import os
from contextlib import contextmanager

_ENABLED = False


def is_enabled() -> bool:
    return _ENABLED


_PHOENIX_URL: str | None = None  # 自动启动的 Phoenix UI 地址（供前端跳转）


def phoenix_url() -> str | None:
    """Phoenix UI 地址，供前端「打开监控面板」跳转。

    自动启动成功后为实际地址；否则在追踪已启用时按 PHOENIX_OTLP / 默认 6006 推断；未启用则 None。
    """
    if _PHOENIX_URL:
        return _PHOENIX_URL
    if not _ENABLED:
        return None
    otlp = os.getenv("PHOENIX_OTLP")
    if otlp:
        return otlp.split("/v1/traces")[0] or None
    return "http://localhost:6006"


def _launch_phoenix() -> str | None:
    """进程内拉起本地 Phoenix（后台线程），返回其 UI 地址；未装/失败/被禁用则返回 None（绝不抛）。"""
    global _PHOENIX_URL
    if os.getenv("PHOENIX_NO_LAUNCH"):
        return None
    try:
        import phoenix as px
    except Exception:
        print("[tracing] 未安装 arize-phoenix，跳过自动启动"
              "（pip install arize-phoenix，或手动 `python -m phoenix.server.main serve`）")
        return None
    try:
        session = px.launch_app()  # 起本地 Phoenix（默认 6006）于后台线程，随主进程存活
        url = (getattr(session, "url", None) or "http://localhost:6006").rstrip("/")
        _PHOENIX_URL = url
        print(f"[tracing] 已在进程内启动 Phoenix：{url}")
        return url
    except Exception as e:  # noqa: BLE001
        print(f"[tracing] Phoenix 自动启动失败（可手动 `phoenix serve`）：{e}")
        return None


@contextmanager
def span(name: str, **attrs):
    """结构化嵌套 span：在 2a 自动 LLM span 之上手动标注 fusion/panel/judge/... 边界。

    - 未启用追踪（FUSION_TRACING 未设、未注入 exporter）→ no-op：yield None 立即返回，零开销，
      也不 import 任何追踪库。core 里 `with span(...)` 在关追踪时等同空操作。
    - 启用后用全局 TracerProvider 起一个 span 并设为 current；其作用域内通过 asyncio.create_task
      新建的任务会捕获该 OTel 上下文，故面板任务里的自动 LLM span 自动挂到本 span 之下。
    - 属性非标量转 str；set_attribute 异常吞掉——追踪绝不拖垮主程序。
    """
    if not _ENABLED:
        yield None
        return
    from opentelemetry import trace

    with trace.get_tracer("fusion_agent").start_as_current_span(name) as sp:
        for k, v in attrs.items():
            try:
                sp.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
            except Exception:  # noqa: BLE001
                pass
        yield sp


def init_tracing(*, span_exporter=None) -> bool:
    """按 env 开关初始化追踪。

    - 未设 FUSION_TRACING 且未注入 exporter → no-op，返回 False。
    - 否则 instrument openai SDK，span 导出到本地 Phoenix(OTLP) 或注入的 exporter。
    - 幂等：重复调用只初始化一次。
    - span_exporter 仅供测试注入内存 exporter（离线、确定性）。
    任何初始化异常都被吞掉并返回 False——追踪绝不拖垮主程序。
    """
    global _ENABLED
    if _ENABLED:
        return True
    if span_exporter is None and not os.getenv("FUSION_TRACING"):
        return False
    try:
        from openinference.instrumentation.openai import OpenAIInstrumentor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = TracerProvider()
        if span_exporter is not None:
            provider.add_span_processor(SimpleSpanProcessor(span_exporter))
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("PHOENIX_OTLP")
            if not endpoint:
                # 未指定外部端点 → 进程内自动起本地 Phoenix（消灭第二个终端）
                base = _launch_phoenix()
                endpoint = (base or "http://localhost:6006").rstrip("/") + "/v1/traces"
            provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))

        trace.set_tracer_provider(provider)
        OpenAIInstrumentor().instrument(tracer_provider=provider)
        _ENABLED = True
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[tracing] 初始化失败，已跳过追踪：{e}")
        return False
