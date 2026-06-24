"""向量库后端（策略模式，仿 search_providers）。

统一接口 VectorStore.add()/query()/count()/clear()，每个后端一个实现类；注册表 + 工厂按名实例化。
新增一个向量库 = 新增实现类 + register_store，RAG 上层（rag.py）完全不用改。

默认后端 chroma（持久化、自带相似度查询）。chromadb 仅在实例化时才 import——
本模块可被无 chromadb 的环境导入（测试用注入式假后端，见 tests/test_rag_mock.py）。
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class VectorStore(ABC):
    """向量库统一接口。embeddings 为等长浮点向量；query 返回按相似度升序的 (document, metadata, distance)。"""

    name: str = "base"

    @abstractmethod
    def add(self, ids: list[str], embeddings: list[list[float]],
            documents: list[str], metadatas: list[dict]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: list[float], k: int = 5) -> list[tuple[str, dict, float]]:
        """返回最相近的 k 条 (document, metadata, distance)；distance 越小越相近。"""
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    def clear(self) -> None:
        """清空库（重建）。可选；默认未实现。"""
        raise NotImplementedError

    def list_sources(self) -> list[tuple[str, int]]:
        """返回 [(source, chunk_count)]，按 source 排序。可选；默认未实现。"""
        raise NotImplementedError

    def delete_by_source(self, source: str) -> int:
        """删除某来源（source 元数据）的所有 chunk，返回删除条数。可选；默认未实现。"""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB 持久化后端（cosine 距离）。chromadb 在 __init__ 时才 import。"""

    name = "chroma"

    def __init__(self, *, path: str = "rag_chroma", collection: str = "fusion_kb"):
        import chromadb

        self._client = chromadb.PersistentClient(path=path)
        self._name = collection
        # cosine 距离更适合文本嵌入（默认是 L2）
        self._col = self._client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadatas) -> None:
        # upsert：同 id 重复 ingest 时更新而非报错
        self._col.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def query(self, embedding, k: int = 5) -> list[tuple[str, dict, float]]:
        res = self._col.query(query_embeddings=[embedding], n_results=k)
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[tuple[str, dict, float]] = []
        for i, d in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 0.0
            out.append((d, meta or {}, float(dist)))
        return out

    def count(self) -> int:
        return self._col.count()

    def clear(self) -> None:
        self._client.delete_collection(self._name)
        self._col = self._client.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )

    def list_sources(self) -> list[tuple[str, int]]:
        got = self._col.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for m in (got.get("metadatas") or []):
            src = (m or {}).get("source")
            if src:
                counts[src] = counts.get(src, 0) + 1
        return sorted(counts.items())

    def delete_by_source(self, source: str) -> int:
        before = self._col.count()
        self._col.delete(where={"source": source})
        return before - self._col.count()


# --------------------------- 注册表 + 工厂 ---------------------------
_STORES: dict[str, type[VectorStore]] = {}


def register_store(name: str, cls: type[VectorStore]) -> None:
    _STORES[name.lower()] = cls


def available_stores() -> list[str]:
    return sorted(_STORES)


def create_vector_store(name: str, **kwargs) -> VectorStore:
    key = name.lower()
    if key not in _STORES:
        raise ValueError(
            f"未知向量库后端：{name}，可用：{', '.join(available_stores()) or '(无)'}"
        )
    return _STORES[key](**kwargs)


register_store("chroma", ChromaVectorStore)
