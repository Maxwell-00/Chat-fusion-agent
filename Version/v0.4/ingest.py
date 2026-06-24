#!/usr/bin/env python3
"""把本地文档（.md / .txt / .pdf）建索引进向量库，供 Web「本地知识库」与 CLI `/kb` 检索。

需要真实嵌入端点（EMBED_MODEL 等，见 .env.example）。用法（在 fusion_agent 目录）：
    python ingest.py                 # 扫描 RAG_DOCS_DIR（默认 docs/），切块嵌入入库（同文件 upsert）
    python ingest.py --rebuild       # 先清空向量库再全量重建
    python ingest.py --docs PATH     # 临时指定文档目录
    python ingest.py --stats         # 只看当前库里有多少块（不嵌入）
"""
from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

load_dotenv()

import providers  # noqa: E402
from config import load_config  # noqa: E402
from rag import build_kb, iter_doc_files  # noqa: E402


async def _run(args) -> int:
    cfg = load_config()
    if cfg.embed_cfg is None:
        print("未配置嵌入模型：请在 .env 设 EMBED_MODEL（及可选 EMBED_BASE_URL / EMBED_API_KEY）。")
        return 2
    try:
        kb = build_kb(cfg)
    except Exception as e:  # noqa: BLE001
        print(f"构建知识库失败：{e}")
        return 2

    label = f"{cfg.rag_store}:{cfg.rag_db_path}/{cfg.rag_collection}"
    if args.stats:
        print(f"向量库 {label} 现有 {kb.store.count()} 块。")
        return 0

    if args.rebuild:
        try:
            kb.store.clear()
            print("已清空旧索引。")
        except NotImplementedError:
            print("该后端不支持清空，跳过（改为 upsert 覆盖同名片段）。")

    docs_dir = args.docs or cfg.rag_docs_dir
    files = iter_doc_files(docs_dir)
    if not files:
        print(f"在 {docs_dir} 下没找到 .md / .txt / .pdf 文档。")
        return 0

    print(f"扫描到 {len(files)} 个文档，开始切块、嵌入、入库（{label}）…")
    try:
        stats = await kb.ingest_files(files)
    finally:
        await providers.aclose_all()
    print(f"完成：{stats['files']} 个文件 → {stats['chunks']} 块；库内现共 {kb.store.count()} 块。")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="本地文档建索引（RAG ingest）。")
    ap.add_argument("--docs", help="文档目录（默认 RAG_DOCS_DIR / docs）")
    ap.add_argument("--rebuild", action="store_true", help="先清空向量库再全量重建")
    ap.add_argument("--stats", action="store_true", help="只显示库内块数（不嵌入）")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
