"""本地 RAG（"向内看"）：把本地笔记 / PDF 切块、嵌入、入向量库，以 agent 工具 local_search 召回。

与联网检索对称：local_search 召回的片段同样产出 evidence(来源文件, 片段)，进裁判证据链、可被合成器编号引用。
嵌入走 providers.embed（OpenAI 兼容，唯一网络出口）；向量库后端可插拔（vector_store.py）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import providers
from config import AppConfig, ModelConfig
from tools import Tool, ToolOutcome
from vector_store import VectorStore, create_vector_store

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}
EVIDENCE_SNIPPET_LIMIT = 200


def _read_text(path: Path) -> str:
    """读出纯文本：.pdf 用 pypdf 抽取正文，其余按 UTF-8 文本读。"""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """按字符滑窗切块（带重叠）。size<=0 表示不切。"""
    text = text.strip()
    if not text:
        return []
    if size <= 0:
        return [text]
    step = max(1, size - max(0, overlap))
    return [text[i:i + size] for i in range(0, len(text), step)]


def iter_doc_files(docs_dir: str) -> list[Path]:
    """递归列出 docs_dir 下受支持的文档（.md/.txt/.pdf）。"""
    base = Path(docs_dir)
    if not base.exists():
        return []
    return sorted(
        p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
    )


class LocalKnowledgeBase:
    """本地知识库：注入 VectorStore + 嵌入模型配置，负责 ingest 与 search。"""

    def __init__(
        self,
        store: VectorStore,
        embed_cfg: ModelConfig,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        self.store = store
        self.embed_cfg = embed_cfg
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        return await providers.embed(self.embed_cfg, texts)

    async def ingest_file(self, path) -> int:
        """单个文件：读文本→切块→嵌入→入库（同源 id 用 upsert 更新）。返回入库的 chunk 数。"""
        path = Path(path)
        chunks = chunk_text(_read_text(path), self.chunk_size, self.chunk_overlap)
        if not chunks:
            return 0
        embeddings = await self._embed(chunks)
        src = str(path)
        sid = hashlib.md5(src.encode("utf-8")).hexdigest()[:8]
        ids = [f"{sid}-{i}" for i in range(len(chunks))]
        # source = 全路径（删除/列举的主键，唯一）；source_file_name = 文件名（展示 / 按名删除）
        metadatas = [
            {"source": src, "source_file_name": path.name, "chunk": i}
            for i in range(len(chunks))
        ]
        self.store.add(ids, embeddings, chunks, metadatas)
        return len(chunks)

    async def ingest_files(self, files: list[Path]) -> dict:
        """批量 ingest（CLI 用）：逐个文件调用 ingest_file。返回 {files, chunks}。"""
        total = 0
        for path in files:
            total += await self.ingest_file(path)
        return {"files": len(files), "chunks": total}

    def delete_document(self, source: str) -> int:
        """从向量库删除某来源（source = ingest 时存的全路径）的全部 chunk，返回删除条数。"""
        return self.store.delete_by_source(source)

    def list_documents(self) -> list[tuple[str, str, int]]:
        """列出已入库文档：[(文件名, 来源路径, chunk 数)]。"""
        return [(Path(src).name, src, n) for src, n in self.store.list_sources()]

    async def search(self, query: str, k: int = 5) -> list[tuple[str, dict, float]]:
        if not query.strip():
            return []
        vec = (await self._embed([query]))[0]
        return self.store.query(vec, k)


def _format_hits(query: str, hits: list[tuple[str, dict, float]]) -> str:
    if not hits:
        return f'本地知识库中没有检索到与 "{query}" 相关的内容。'
    lines = [f'本地知识库检索 "{query}" 的结果：']
    for i, (doc, meta, _dist) in enumerate(hits, 1):
        lines.append(f"[{i}] 来源：{meta.get('source', '?')}\n{doc}")
    return "\n\n".join(lines)


_LOCAL_SEARCH_PARAMS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "要在本地知识库中检索的查询词"},
        "k": {"type": "integer", "description": "返回片段数(默认5)"},
    },
    "required": ["query"],
}


def build_rag_tool(kb: LocalKnowledgeBase, *, default_k: int = 5) -> Tool:
    """把本地知识库包成 agent 工具 local_search（与 web_search 同构，召回片段产出来源+证据）。"""

    async def _local_search(args: dict) -> ToolOutcome:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolOutcome("local_search 需要 query 参数。")
        k = int(args.get("k") or default_k)
        hits = await kb.search(query, k)
        seen: set[str] = set()
        urls: list[str] = []
        evidence: list = []
        for doc, meta, _dist in hits:
            src = meta.get("source", "")
            if src and src not in seen:
                seen.add(src)
                urls.append(src)
            if src:
                evidence.append((src, (doc or "")[:EVIDENCE_SNIPPET_LIMIT]))
        return ToolOutcome(_format_hits(query, hits), urls, evidence)

    return Tool(
        "local_search",
        "检索本地知识库（用户的笔记 / PDF 等本地文档），返回最相关的若干片段及其来源文件。"
        "当问题可能与用户的本地资料 / 私有文档相关，或需要引用本地材料时使用。",
        _LOCAL_SEARCH_PARAMS,
        _local_search,
    )


def build_kb(cfg: AppConfig) -> LocalKnowledgeBase:
    """按 AppConfig 造本地知识库（配置的向量库后端 + 嵌入模型）。需先配置 EMBED_MODEL。"""
    if cfg.embed_cfg is None:
        raise RuntimeError(
            "未配置嵌入模型：请在 .env 设 EMBED_MODEL（及可选 EMBED_BASE_URL / EMBED_API_KEY）。"
        )
    store = create_vector_store(cfg.rag_store, path=cfg.rag_db_path, collection=cfg.rag_collection)
    return LocalKnowledgeBase(
        store, cfg.embed_cfg, chunk_size=cfg.rag_chunk_size, chunk_overlap=cfg.rag_chunk_overlap
    )
