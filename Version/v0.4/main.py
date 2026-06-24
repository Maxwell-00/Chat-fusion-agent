"""入口：加载 .env -> 构建配置 -> 启动 CLI。

运行：python main.py
"""
from __future__ import annotations

import sys

from dotenv import load_dotenv


def main():
    load_dotenv()  # 从当前目录的 .env 读取配置（须在 load_config 之前）

    from tracing import init_tracing

    init_tracing()  # 默认 no-op；设 FUSION_TRACING 后自动追踪所有模型调用

    from config import load_config

    try:
        cfg = load_config()
    except Exception as e:
        print(f"启动失败：{e}")
        print("提示：先 `cp .env.example .env` 并填入真实 API key。")
        sys.exit(1)

    from chat import run_repl

    run_repl(cfg)


if __name__ == "__main__":
    main()
