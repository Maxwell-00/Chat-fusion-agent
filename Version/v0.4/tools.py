"""工具注册中心。

把"工具"抽象成 name + description + JSON Schema + 执行器，统一登记到 ToolRegistry。
- registry.schemas() 动态产出 OpenAI 原生 tools 的 JSON Schema，交给大模型。
- registry.execute(name, args) 分发到对应执行器。
web_search / web_fetch 两个工具绑定到可插拔的 SearchProvider 上。

执行器除了回灌给模型的文本，还产出 evidence(url, snippet) —— 供裁判做信源提纯/编号。
web_fetch 对视频/社交站做黑名单拦截，避免无意义抓取(成本/延迟杀手)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable
from urllib.parse import urlparse

from search_providers import SearchProvider, SearchResult
from tracing import span

FETCH_CHAR_LIMIT = 6000
EVIDENCE_SNIPPET_LIMIT = 200

# 抓取黑名单：视频/社交站正文无法有效提取，直接拦截不发请求。
FETCH_BLOCKLIST = (
    "youtube.com",
    "youtu.be",
    "instagram.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "tiktok.com",
    "facebook.com",
)


def _is_blocked(url: str) -> bool:
    # 按 host 精确/子域匹配，避免子串误伤（如 "x.com" 命中 "businessx.com"）。
    # 兼容无 scheme 的 URL（如 "www.youtube.com/x"）：补 "//" 让 urlparse 识别 netloc。
    host = (urlparse(url if "://" in url else "//" + url).hostname or "").lower()
    if not host:
        return False
    return any(host == d or host.endswith("." + d) for d in FETCH_BLOCKLIST)


@dataclass
class ToolOutcome:
    text: str                                       # 回灌给模型的内容
    urls: list[str] = field(default_factory=list)   # 涉及的来源 URL（用于汇总 sources）
    evidence: list = field(default_factory=list)    # [(url, snippet), ...] 供裁判提纯


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict                                 # JSON Schema
    executor: Callable[[dict], Awaitable[ToolOutcome]]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]

    async def execute(self, name: str, args: dict) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(f"未知工具：{name}")
        try:
            return await tool.executor(args or {})
        except Exception as e:
            return ToolOutcome(f"工具 {name} 调用失败：{e}")


def _format_search(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f'搜索 "{query}" 没有返回结果。'
    lines = [f'搜索 "{query}" 的结果：']
    for i, r in enumerate(results, 1):
        lines.append(f"[{i}] {r.title}\n{r.url}\n{r.snippet}")
    return "\n\n".join(lines)


_WEB_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "搜索查询词"},
        "max_results": {"type": "integer", "description": "返回结果数(默认5)"},
    },
    "required": ["query"],
}

_WEB_FETCH_PARAMS = {
    "type": "object",
    "properties": {
        "url": {"type": "string", "description": "要抓取的网页 URL"},
    },
    "required": ["url"],
}


def build_web_registry(
    provider: SearchProvider,
    *,
    default_max_results: int = 5,
    fetch_char_limit: int = FETCH_CHAR_LIMIT,
) -> ToolRegistry:
    """根据选定的搜索 provider 构建一套联网工具(web_search/web_fetch)。"""

    async def _search(args: dict) -> ToolOutcome:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutcome("web_search 需要 query 参数。")
        count = int(args.get("max_results") or default_max_results)
        with span("web_search", query=query) as sp:
            results = await provider.search(query, count)
            if sp is not None:
                try:
                    sp.set_attribute("results", len(results))
                except Exception:  # noqa: BLE001
                    pass
            urls = [r.url for r in results if r.url]
            evidence = [
                (r.url, (r.snippet or r.title or "")[:EVIDENCE_SNIPPET_LIMIT])
                for r in results
                if r.url
            ]
            return ToolOutcome(_format_search(query, results), urls, evidence)

    async def _fetch(args: dict) -> ToolOutcome:
        url = str(args.get("url", "")).strip()
        if not url:
            return ToolOutcome("web_fetch 需要 url 参数。")
        if _is_blocked(url):
            # 黑名单：不发真实请求，直接提示模型依赖搜索摘要。
            return ToolOutcome(
                f"该链接（{url}）为视频/社交媒体，无法提取正文，请依赖搜索摘要。"
            )
        with span("web_fetch", url=url):
            content = await provider.fetch(url)
            if len(content) > fetch_char_limit:
                content = content[:fetch_char_limit] + "\n…(内容已截断)"
            return ToolOutcome(
                f"网页内容（{url}）：\n{content}",
                [url],
                [(url, content[:EVIDENCE_SNIPPET_LIMIT])],
            )

    reg = ToolRegistry()
    reg.register(
        Tool(
            "web_search",
            "用搜索引擎检索实时网络信息，返回若干结果(标题/URL/摘要)。"
            "当问题涉及最新信息、具体数据、事实核查，或你不确定/可能过时的内容时使用。",
            _WEB_SEARCH_PARAMS,
            _search,
        )
    )
    reg.register(
        Tool(
            "web_fetch",
            "抓取指定网页的正文内容。仅在搜索摘要不足、确需阅读某权威页面全文时使用；"
            "每次搜索后最多挑 2 个最相关的 URL 抓取，不要逐条抓取所有链接。",
            _WEB_FETCH_PARAMS,
            _fetch,
        )
    )
    return reg
