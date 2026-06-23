"""对话持久化层（SQLite，stdlib，无额外依赖）。

存什么 / 怎么存——关键区分（沿用 chat_context 的"全量 vs 有界窗口"思路）：
- messages：**全量**消息日志，供 UI 滚动回看（每轮 user/assistant 都 append）。
- summary + summarized_count（挂在 conversations 行上）：喂给模型那个**有界窗口**的"另一半"。
  summary 概括了前 summarized_count 条较早消息；模型窗口 = messages[summarized_count:]。
  载入会话时据此重建 ChatHistory，既不丢历史、又不超预算。

线程安全：Gradio 可能从不同线程回调，连接用 check_same_thread=False + 一把全局锁串行化。
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

from chat_context import ChatHistory


def _now() -> str:
    # 微秒精度：避免同一秒内多次操作的 updated_at 撞车导致列表排序错乱。
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class ConversationMeta:
    id: int
    title: str
    created_at: str
    updated_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT    NOT NULL,
    summary          TEXT    NOT NULL DEFAULT '',
    summarized_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL,
    content         TEXT    NOT NULL,
    created_at      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


class ConversationStore:
    """多会话持久化：conversations + messages，每会话带 summary/summarized_count。"""

    def __init__(self, path: str = "fusion_agent.db") -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------- conversations ----------------
    def create_conversation(self, title: str = "新对话") -> int:
        ts = _now()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO conversations(title, created_at, updated_at) VALUES (?,?,?)",
                (title, ts, ts),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_conversations(self) -> list[ConversationMeta]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations "
                "ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [ConversationMeta(r["id"], r["title"], r["created_at"], r["updated_at"]) for r in rows]

    def rename_conversation(self, conv_id: int, title: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (title, _now(), conv_id),
            )
            self._conn.commit()

    def delete_conversation(self, conv_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            self._conn.commit()

    # ---------------- messages（全量日志）----------------
    def add_message(self, conv_id: int, role: str, content: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO messages(conversation_id, role, content, created_at) VALUES (?,?,?,?)",
                (conv_id, role, content, _now()),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conv_id)
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_messages(self, conv_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages WHERE conversation_id=? ORDER BY id",
                (conv_id,),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def count_messages(self, conv_id: int) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conv_id,)
                ).fetchone()[0]
            )

    # ---------------- summary / 窗口边界 ----------------
    def get_summary(self, conv_id: int) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
        return row["summary"] if row else ""

    def get_summarized_count(self, conv_id: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT summarized_count FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
        return int(row["summarized_count"]) if row else 0

    def save_window(self, conv_id: int, summary: str, summarized_count: int) -> None:
        """持久化模型窗口的'另一半'：摘要 + 已被摘要覆盖的前缀消息数。"""
        with self._lock:
            self._conn.execute(
                "UPDATE conversations SET summary=?, summarized_count=?, updated_at=? WHERE id=?",
                (summary, summarized_count, _now(), conv_id),
            )
            self._conn.commit()


# ---------------- store <-> ChatHistory 桥接 ----------------
def load_history(
    store: ConversationStore, conv_id: int, *, char_budget: int, keep_recent: int
) -> ChatHistory:
    """据持久化状态重建喂给模型的 ChatHistory：summary + 未被摘要覆盖的尾部消息。"""
    h = ChatHistory(char_budget=char_budget, keep_recent=keep_recent)
    h.summary = store.get_summary(conv_id)
    msgs = store.get_messages(conv_id)
    h.turns = msgs[store.get_summarized_count(conv_id):]
    return h


def persist_turn(
    store: ConversationStore,
    conv_id: int,
    history: ChatHistory,
    user_text: str,
    assistant_text: str,
) -> None:
    """一轮对话后落库：追加全量消息 + 同步窗口边界（summary + summarized_count）。

    边界 = 全量消息数 - 当前模型窗口条数(turns) - 尚未并入摘要的积压(pending)。
    这样重启重建时，pending 会作为最近窗口被重新纳入（下一轮再压缩），不丢信息。
    调用约定：传入的 user_text/assistant_text 应与本轮已 add 进 history 的内容一致。
    """
    store.add_message(conv_id, "user", user_text)
    store.add_message(conv_id, "assistant", assistant_text)
    total = store.count_messages(conv_id)
    boundary = max(0, total - len(history.turns) - history.pending_count())
    store.save_window(conv_id, history.summary, boundary)
