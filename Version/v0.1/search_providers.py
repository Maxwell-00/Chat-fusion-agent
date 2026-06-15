"""搜索后端（策略模式）。

统一接口 SearchProvider.search()/fetch()，每个厂商一个实现类；通过注册表 + 工厂
按配置动态实例化。新增一个搜索 API = 新增一个实现类 + register_provider，
核心 Agent 调用逻辑完全不用改。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from html.parser import HTMLParser

import httpx


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# --------------------------- 通用网页抓取（provider 无关的兜底） ---------------------------
class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


async def generic_fetch(url: str, *, timeout: float = 30.0, char_limit: int = 20000) -> str:
    """无依赖的通用网页抓取：GET + 提取可见文本。provider 可覆盖为更优实现。"""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FusionAgent/1.0)"}
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as c:
        r = await c.get(url)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "")
        text = r.text
    if "html" in ctype.lower() or "<html" in text[:2000].lower():
        p = _TextExtractor()
        p.feed(text)
        text = "\n".join(p.parts)
    return text[:char_limit]


# --------------------------- 接口 ---------------------------
class SearchProvider(ABC):
    """搜索后端统一接口。新增厂商实现 search()，必要时覆盖 fetch()。"""

    name: str = "base"

    @abstractmethod
    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        ...

    async def fetch(self, url: str) -> str:
        """抓取网页正文。默认用通用抓取；有专用抽取接口的 provider 可覆盖。"""
        return await generic_fetch(url)


# --------------------------- 具体实现 ---------------------------
class TavilyProvider(SearchProvider):
    """Tavily：/search + /extract，Bearer 鉴权。"""

    name = "tavily"

    def __init__(self, api_key: str, *, base_url: str = "https://api.tavily.com", timeout: float = 30.0):
        if not api_key:
            raise ValueError("Tavily 需要 api_key")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"}

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        payload = {
            "query": query,
            "max_results": count,
            "search_depth": "basic",
            "include_answer": False,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/search", headers=self._headers, json=payload)
            r.raise_for_status()
            data = r.json()
        return [
            SearchResult(it.get("title", ""), it.get("url", ""), it.get("content", ""))
            for it in data.get("results", [])
        ]

    async def fetch(self, url: str) -> str:
        payload = {"urls": [url], "extract_depth": "basic"}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/extract", headers=self._headers, json=payload)
            r.raise_for_status()
            data = r.json()
        results = data.get("results", [])
        if not results:
            raise RuntimeError(f"Tavily 抓取失败：{data.get('failed_results') or url}")
        return results[0].get("raw_content", "") or ""


class BochaProvider(SearchProvider):
    """博查 Bocha：/v1/web-search，Bearer 鉴权；抓取走通用兜底。"""

    name = "bocha"

    def __init__(self, api_key: str, *, base_url: str = "https://api.bochaai.com", timeout: float = 30.0):
        if not api_key:
            raise ValueError("Bocha 需要 api_key")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    async def search(self, query: str, count: int = 5) -> list[SearchResult]:
        payload = {"query": query, "count": min(count, 50), "summary": True, "freshness": "noLimit"}
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/v1/web-search", headers=self._headers, json=payload)
            r.raise_for_status()
            data = r.json()
        web = (data.get("data") or {}).get("webPages") or data.get("webPages") or {}
        values = web.get("value", []) if isinstance(web, dict) else []
        return [
            SearchResult(
                it.get("name", ""),
                it.get("url", ""),
                it.get("summary") or it.get("snippet") or "",
            )
            for it in values
        ]


# --------------------------- 注册表 + 工厂 ---------------------------
_PROVIDERS: dict[str, type[SearchProvider]] = {}


def register_provider(name: str, cls: type[SearchProvider]) -> None:
    """登记一个搜索 provider；扩展新后端只需调用本函数。"""
    _PROVIDERS[name.lower()] = cls


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def create_provider(
    name: str,
    api_key: str,
    *,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> SearchProvider:
    """按名字动态实例化对应 provider。"""
    key = name.lower()
    if key not in _PROVIDERS:
        raise ValueError(
            f"未知搜索 provider：{name}，可用：{', '.join(available_providers()) or '(无)'}"
        )
    kwargs: dict = {"timeout": timeout}
    if base_url:
        kwargs["base_url"] = base_url
    return _PROVIDERS[key](api_key, **kwargs)


register_provider("tavily", TavilyProvider)
register_provider("bocha", BochaProvider)
