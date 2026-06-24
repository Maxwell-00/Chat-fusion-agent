#!/usr/bin/env python3
"""一键清空本地 Phoenix 的历史 trace / span / 对话记录（发布前清理环境用）。

只删 Phoenix **自己的**本地 SQLite 数据库文件，不碰本项目任何代码、.env 或 fusion_agent.db。
原理：Phoenix 把 trace/span 持久化在它的工作目录下的 phoenix.db（默认 ~/.phoenix/phoenix.db）。
删掉该文件后，下次 `python -m phoenix.server.main serve` 会自动新建一个空库——干净如初。

用法（在你本机，**先停掉正在运行的 Phoenix 服务**后再运行）：
    python clean_traces.py                # 解析默认路径 → 列出 → 确认 → 删除
    python clean_traces.py -y             # 跳过确认
    python clean_traces.py --dry-run      # 只看会删什么，不动手
    python clean_traces.py --db PATH      # 手动指定 phoenix.db（最高优先级）
    python clean_traces.py --force        # 即使检测到服务在跑也强删（不推荐）

路径解析优先级：
    --db  >  PHOENIX_SQL_DATABASE_URL(sqlite:///…)  >  (--working-dir | PHOENIX_WORKING_DIR | ~/.phoenix)/phoenix.db

注意：
- 若你把 trace 存到了外部 Postgres（PHOENIX_SQL_DATABASE_URL=postgresql://…），本脚本**不**处理；
  请用 Phoenix UI 的 “Remove Data”，或配置 retention policy 自动清理。
- Phoenix 启动时会在终端打印它实际使用的数据库路径；若与本脚本解析的不一致，用 --db 指定。
- 纯 stdlib、无第三方依赖，也不 import 本项目代码，可单独拷到任何地方运行。
"""
from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from urllib.parse import unquote

DEFAULT_PORT = int(os.getenv("PHOENIX_PORT", "6006"))


def resolve_db_path(args) -> Path | None:
    """按优先级解析 phoenix.db 的位置；外部非 sqlite 库返回 None。"""
    if args.db:  # 1) 显式指定
        return Path(args.db).expanduser()

    url = os.getenv("PHOENIX_SQL_DATABASE_URL")  # 2) 数据库 URL
    if url:
        if not url.startswith("sqlite"):
            return None  # postgres 等外部库：交给上层提示
        # 取 sqlite[+driver]:/// 之后的部分即数据库文件路径（SQLAlchemy 约定）：
        #   sqlite:///C:/Users/u/.phoenix/phoenix.db → C:/Users/u/.phoenix/phoenix.db （Win 绝对）
        #   sqlite:////home/u/.phoenix/phoenix.db    → /home/u/.phoenix/phoenix.db    （POSIX 绝对）
        idx = url.find(":///")
        if idx == -1:
            return None  # sqlite:// 内存库 / 异常形式：不处理
        path = unquote(url[idx + 4:]).split("?", 1)[0]
        return Path(path) if path else None

    # 3) 工作目录（默认 ~/.phoenix）下的 phoenix.db
    working = args.working_dir or os.getenv("PHOENIX_WORKING_DIR") or str(Path.home() / ".phoenix")
    return Path(working).expanduser() / "phoenix.db"


def server_running(port: int) -> bool:
    """粗判 Phoenix 是否在跑：本地端口是否有人监听。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="清空本地 Phoenix 历史 trace/span 数据（删 phoenix.db，不影响项目代码）。"
    )
    ap.add_argument("--db", help="直接指定 phoenix.db 路径（最高优先级）")
    ap.add_argument("--working-dir", help="Phoenix 工作目录；默认 PHOENIX_WORKING_DIR 或 ~/.phoenix")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT,
                    help=f"Phoenix 端口，用于检测服务是否在跑（默认 {DEFAULT_PORT}）")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过确认直接删除")
    ap.add_argument("--dry-run", action="store_true", help="只显示会删的文件，不实际删除")
    ap.add_argument("--force", action="store_true", help="即使检测到 Phoenix 在运行也继续（不推荐）")
    args = ap.parse_args()

    url = os.getenv("PHOENIX_SQL_DATABASE_URL", "")
    if url and not url.startswith("sqlite"):
        print(f"检测到 PHOENIX_SQL_DATABASE_URL={url}")
        print("这是外部数据库（非 SQLite），本脚本不会去删外部库。")
        print("请改用 Phoenix UI 的 “Remove Data”，或配置 retention policy 自动清理。")
        return 2

    db = resolve_db_path(args)
    if db is None:
        print("无法解析 phoenix.db 路径，请用 --db 指定。")
        return 2

    # phoenix.db 及其 SQLite WAL/SHM 旁文件（开了 WAL 模式时会有）
    targets = [db, db.with_name(db.name + "-wal"), db.with_name(db.name + "-shm")]
    existing = [p for p in targets if p.exists()]

    print(f"Phoenix 数据库路径：{db}")
    if not existing:
        print("没找到任何数据库文件——可能本就干净，或 Phoenix 用的是别的路径。")
        print("（Phoenix 启动时会打印实际路径；不一致时用 --db 指定。）")
        return 0

    print("将删除以下文件：")
    for p in existing:
        try:
            kb = p.stat().st_size / 1024
        except OSError:
            kb = 0
        print(f"  - {p}  ({kb:.0f} KB)")

    if args.dry_run:
        print("[dry-run] 未删除任何文件。")
        return 0

    if not args.force and server_running(args.port):
        print(f"\n⚠ 检测到 127.0.0.1:{args.port} 有服务在监听——Phoenix 可能正在运行。")
        print("  请先停掉 Phoenix（在其终端按 Ctrl-C）再运行本脚本，避免文件被占用/删不干净。")
        print("  确认无误要强制删除可加 --force。")
        return 1

    if not args.yes:
        ans = input("\n确认删除？此操作不可恢复 [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("已取消。")
            return 0

    failed = []
    for p in existing:
        try:
            p.unlink()
            print(f"已删除 {p}")
        except OSError as e:
            failed.append(p)
            print(f"删除失败 {p}：{e}")

    if failed:
        print("\n部分文件删除失败（多半是 Phoenix 仍在运行占用文件）。请停服后重试。")
        return 1

    print("\n✅ 清理完成。下次 `python -m phoenix.server.main serve` 会自动新建一个空库。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
