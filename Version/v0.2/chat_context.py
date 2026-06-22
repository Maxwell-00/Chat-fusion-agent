"""普通对话的上下文管理：滚动摘要 + 最近 N 轮（压缩不阻塞作答）。

- turns：最近的逐字消息（user/assistant）。
- summary：更早对话被压缩成的一段"记忆"，作为 system 消息注入。
- _pending：已从 turns 摘出、等待并入摘要的较早消息（不进入发送给模型的 prompt）。

关键设计——压缩与作答解耦，消除"等摘要才回答"的卡顿：
  1) compact()：同步、零模型调用。超预算时把"最近 N 条之外"的较早消息从 turns 摘到
     _pending，turns 立刻落回预算内 —— prompt 当场就是安全的，主模型可立即作答。
  2) summarize_pending()：异步。由上层用 asyncio.create_task 启动，与作答流式并发，
     在作答结束时再 await（通常此刻早已完成，故无感知等待）。把 _pending 并入 summary。

健壮性：
- 摘要失败不丢信息也不崩：_pending 原样保留，下一轮再试；prompt 始终有界。
- 摘要持续失败时 _pending 由 _enforce_pending_cap 兜底限幅（仅防内存膨胀，不影响 prompt）。
- _summarizing 作为可重入保护，避免并发启动两个摘要任务。

为什么按字符而非 token：本项目模型异构（DeepSeek / Mimo / MiniMax…），没有统一 tokenizer，
字符预算 provider 无关、行为可预测、零额外依赖。
"""
from __future__ import annotations

from pathlib import Path

import providers
from config import ModelConfig

_SUMMARY_PROMPT = (Path(__file__).parent / "prompts" / "summary.txt").read_text(
    encoding="utf-8"
)


def _chars(messages: list[dict]) -> int:
    return sum(len(m.get("content") or "") for m in messages)


class ChatHistory:
    """普通多轮对话的历史 + 滚动摘要（同步落预算 + 后台摘要）。"""

    def __init__(self, *, char_budget: int = 8000, keep_recent: int = 6) -> None:
        self.summary: str = ""
        self.turns: list[dict] = []
        self._pending: list[dict] = []
        self._summarizing: bool = False
        self.char_budget = char_budget
        self.keep_recent = keep_recent

    # -------- 写入 --------
    def add_user(self, text: str) -> None:
        self.turns.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self.turns.append({"role": "assistant", "content": text})

    def clear(self) -> None:
        self.summary = ""
        self.turns = []
        self._pending = []
        self._summarizing = False

    # -------- 读取 --------
    def char_count(self) -> int:
        return _chars(self.turns)

    def has_pending(self) -> bool:
        return bool(self._pending)

    def build_messages(self) -> list[dict]:
        """组装实际发给模型的消息：可选的记忆(system) + 最近逐字消息。"""
        messages: list[dict] = []
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": "以下是更早对话的摘要记忆，供你延续上下文：\n" + self.summary,
                }
            )
        messages.extend(self.turns)
        return messages

    # -------- 第 1 步：同步落预算（零等待、零模型调用） --------
    def compact(self) -> bool:
        """超预算时把溢出的较早消息从 turns 摘到 _pending，turns 立即落回预算内。

        不调用任何模型，可在作答前同步调用。返回是否有待摘要的积压（_pending 非空），
        上层据此决定是否启动后台摘要任务（含上一轮失败遗留的积压，会一并重试）。
        """
        if self.char_count() > self.char_budget and len(self.turns) > self.keep_recent:
            keep = self.turns[-self.keep_recent :] if self.keep_recent > 0 else []
            older = self.turns[: len(self.turns) - len(keep)]
            self.turns = keep
            self._pending.extend(older)
            self._enforce_pending_cap()
        return bool(self._pending)

    def _enforce_pending_cap(self) -> None:
        # 仅在摘要持续失败时兜底，防止 _pending 无限增长（不影响 prompt 大小）。
        cap = max(self.char_budget * 4, 20000)
        while self._pending and _chars(self._pending) > cap:
            self._pending.pop(0)

    # -------- 第 2 步：后台摘要（与作答并发，作答结束再 await） --------
    async def summarize_pending(self, cfg: ModelConfig) -> bool:
        """把 _pending 并入 summary。成功返回 True 并移除已并入的这批；失败保留待重试。"""
        if self._summarizing or not self._pending:
            return False
        self._summarizing = True
        try:
            batch = list(self._pending)
            try:
                new_summary = await self._summarize(cfg, batch)
            except Exception:
                return False  # 失败：保留 _pending（turns 已安全），下一轮重试
            self.summary = new_summary
            del self._pending[: len(batch)]  # 仅移除已并入的这批，保留期间可能新增的
            return True
        finally:
            self._summarizing = False

    async def _summarize(self, cfg: ModelConfig, batch: list[dict]) -> str:
        conversation = "\n".join(
            f"{m.get('role', '')}: {m.get('content', '')}" for m in batch
        )
        prompt = _SUMMARY_PROMPT.replace("{prev_summary}", self.summary or "(无)").replace(
            "{conversation}", conversation
        )
        text = await providers.call_model(cfg, [{"role": "user", "content": prompt}])
        return text.strip() or self.summary
