"""统一调用层（架构的核心简化点）。

因为所有模型都走 OpenAI 兼容接口，这里用官方 openai SDK 统一封装，
对外只暴露 call_model 一个入口。将来若遇到不兼容 OpenAI 协议的厂商，
只需在本文件按 base_url/类型加一个分支，上层逻辑完全不动。
"""
from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI

from config import ModelConfig


async def call_model(
    cfg: ModelConfig,
    messages: list[dict],
    *,
    response_format: dict | None = None,
) -> str:
    """调用单个模型并返回文本内容。

    每次创建独立的 AsyncOpenAI 客户端，避免跨事件循环复用导致的问题；
    对 CLI 场景这点开销可忽略。response_format 用于让裁判输出 JSON（厂商支持时）。
    """
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
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


async def call_with_tools(
    cfg: ModelConfig,
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str = "auto",
):
    """带工具的调用，返回完整 assistant message(可能含 tool_calls)。

    供面板 agent 的工具循环使用：上层据此判断是否还要继续调用工具。
    tool_choice='none' 用于在达到上限时强制模型直接作答。
    """
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    resp = await client.chat.completions.create(
        model=cfg.name,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
    )
    return resp.choices[0].message


async def stream(cfg: ModelConfig, messages: list[dict]):
    """无工具的流式调用：异步逐字 yield 文本分片（合成 / 普通对话用）。"""
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
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


async def stream_with_tools(
    cfg: ModelConfig,
    messages: list[dict],
    tools: list[dict],
    *,
    tool_choice: str = "auto",
):
    """带工具的流式调用：原样 yield 每个 chunk，分片缓冲交给策略层处理。"""
    client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
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
