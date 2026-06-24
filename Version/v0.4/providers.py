"""统一调用层（架构的核心简化点）。

因为所有模型都走 OpenAI 兼容接口，这里用官方 openai SDK 统一封装，
对外只暴露 call_model 一个入口。将来若遇到不兼容 OpenAI 协议的厂商，
只需在本文件按 base_url/类型加一个分支，上层逻辑完全不动。
"""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from config import ModelConfig

# 每个 (base_url, api_key) 复用一个客户端，避免每次调用都新建 AsyncOpenAI(底层 httpx
# 连接不复用且不关闭会泄漏)。httpx.AsyncClient 本身支持并发，故面板并行共享同一客户端安全。
# 注意：CLI 每轮用一次 asyncio.run（每轮一个新事件循环），客户端绑定其创建时的循环，
# 不能跨循环复用。因此约定：每个 asyncio.run 顶层结束前调用 aclose_all() 关闭并清空缓存，
# 下一轮再按需重建。
_clients: dict[tuple[str, str], AsyncOpenAI] = {}


def _client(cfg: ModelConfig) -> AsyncOpenAI:
    key = (cfg.base_url, cfg.api_key)
    client = _clients.get(key)
    if client is None:
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        _clients[key] = client
    return client


async def aclose_all() -> None:
    """关闭并清空本轮创建的所有客户端。须在创建它们的同一事件循环内调用
    （即每个 asyncio.run 顶层协程结束前），否则跨循环关闭会出错。"""
    clients = list(_clients.values())
    _clients.clear()
    for client in clients:
        try:
            await client.close()
        except Exception:
            pass


async def call_model(
    cfg: ModelConfig,
    messages: list[dict],
    *,
    response_format: dict | None = None,
) -> str:
    """调用单个模型并返回文本内容。

    客户端按 (base_url, api_key) 复用（见 _client / aclose_all）。
    response_format 用于让裁判输出 JSON（厂商支持时）。
    """
    client = _client(cfg)
    kwargs: dict[str, Any] = dict(
        model=cfg.name,
        messages=messages,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


async def stream(cfg: ModelConfig, messages: list[dict]):
    """无工具的流式调用：异步逐字 yield 文本分片（合成 / 普通对话用）。"""
    client = _client(cfg)
    s = await client.chat.completions.create(
        model=cfg.name,
        messages=messages,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
        stream=True,
    )
    async for chunk in s:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


async def stream_rich(cfg: ModelConfig, messages: list[dict]):
    """流式，区分推理与正文：yield ('reasoning', 文本) / ('content', 文本)。

    思考型模型把推理放在 delta.reasoning_content（部分 SDK 落在 model_extra 里）；
    普通模型只有 content。供 UI 即时展示"思考中"与实时推理流，缓解高首字延迟的体感。
    """
    client = _client(cfg)
    s = await client.chat.completions.create(
        model=cfg.name,
        messages=messages,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
        stream=True,
    )
    async for chunk in s:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        rc = getattr(delta, "reasoning_content", None)
        if rc is None:
            rc = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
        if rc:
            yield ("reasoning", rc)
        if delta.content:
            yield ("content", delta.content)


async def stream_with_tools(
    cfg: ModelConfig,
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str = "auto",
):
    """带工具的流式调用：原样 yield 每个 chunk，分片缓冲交给策略层处理。"""
    client = _client(cfg)
    s = await client.chat.completions.create(
        model=cfg.name,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
        stream=True,
    )
    async for chunk in s:
        yield chunk


async def embed(cfg: ModelConfig, texts: list[str]) -> list[list[float]]:
    """OpenAI 兼容文本嵌入：返回与 texts 等长的向量列表（本地 RAG 用）。客户端复用，同 call_model。"""
    client = _client(cfg)
    resp = await client.embeddings.create(model=cfg.name, input=texts, timeout=cfg.timeout)
    return [list(d.embedding) for d in resp.data]
