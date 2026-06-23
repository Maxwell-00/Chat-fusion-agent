"""Gradio Web UI（v0.3 第 1 步·动作 B）：单聊天窗 + 多会话持久化 + Fusion 收进对话。

交互参考桌面端聊天：输入框旁「➕」展开两个开关——「Fusion 多模型」「联网」。四种组合：
- 都不勾：普通对话（providers.stream + 多轮上下文）。
- 仅联网：默认模型走 agent 工具循环（stream_agent，单模型联网搜索作答；单次不带多轮）。
- 仅 Fusion：离线多模型 Fusion。
- 都勾：联网多模型 Fusion。
Fusion 的合成研报进对话流；「展开过程」切出右侧详情栏：上为裁判裁决，下为各面板原文分栏。

与 CLI 共用同一套事件驱动核心，本文件只做"事件流接界面 + 多会话持久化"。
本地运行：cd fusion_agent && pip install -r requirements.txt && python app.py
"""
from __future__ import annotations

import asyncio
import os

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from tracing import init_tracing  # noqa: E402

init_tracing()  # 默认 no-op；设 FUSION_TRACING 后自动追踪所有模型调用

import providers  # noqa: E402
from agent import stream_agent  # noqa: E402
from chat_context import ChatHistory  # noqa: E402
from config import load_config  # noqa: E402
from fusion import stream_fusion  # noqa: E402
from search_providers import create_provider  # noqa: E402
from store import ConversationStore, load_history, persist_turn  # noqa: E402
from stream_events import (  # noqa: E402
    AgentDegraded,
    AgentDone,
    FusionDone,
    JudgeFinished,
    PanelFinished,
    StageError,
    SynthesisDelta,
    TextDelta,
    ToolStarted,
)
from tool_calling import NativeToolCalling  # noqa: E402
from tools import build_web_registry  # noqa: E402

CFG = load_config()
STORE = ConversationStore(os.getenv("FUSION_DB", "fusion_agent.db"))


# ----------------------- 基础工具 -----------------------
def _choices() -> list[tuple[str, int]]:
    return [(c.title, c.id) for c in STORE.list_conversations()]


def _new_history() -> ChatHistory:
    return ChatHistory(char_budget=CFG.history_char_budget, keep_recent=CFG.history_keep_recent)


def _load_history(conv_id: int) -> ChatHistory:
    return load_history(
        STORE, conv_id, char_budget=CFG.history_char_budget, keep_recent=CFG.history_keep_recent
    )


def _registry():
    key = CFG.active_search_key()
    if not key:
        raise RuntimeError(f"未配置 {CFG.search_provider} 的搜索 key")
    provider = create_provider(CFG.search_provider, key, base_url=CFG.active_search_base_url())
    return build_web_registry(provider, default_max_results=CFG.web_max_results)


def _num(k: str) -> int:
    digits = "".join(ch for ch in k if ch.isdigit())
    return int(digits) if digits else 0


def _render_judge(a) -> str:
    if a is None:
        return ""
    if not a.parsed and a.raw:
        return f"**（裁判未结构化，原文）**\n\n{a.raw}"
    lines = []
    if a.verdict:
        lines.append(f"**结论**：{a.verdict}")
    if a.confirmed_facts:
        lines.append("\n**已确认事实**")
        lines += [f"- {f.fact} {''.join(f.citations)}" for f in a.confirmed_facts]
    if a.debunked_rumors:
        lines.append("\n**存疑传闻**")
        lines += [f"- {r}" for r in a.debunked_rumors]
    if a.blind_spots:
        lines.append("\n**盲点**")
        lines += [f"- {b}" for b in a.blind_spots]
    if a.source_mapping:
        lines.append("\n**信源**")
        lines += [f"- {k} {a.source_mapping[k]}" for k in sorted(a.source_mapping, key=_num)]
    return "\n".join(lines)


def _panel_updates(results):
    """每个模型一个标签页：按 CFG.models 顺序产出更新；本次未参与的标签显示占位。"""
    by_key = {p.model_key: p for p in results}
    ups = []
    for key in CFG.models:
        p = by_key.get(key)
        if p is None:
            ups.append(gr.update(value="_（本次未参与本面板）_"))
        else:
            body = p.content if p.ok else f"_✗ 失败：{p.error}_"
            ups.append(gr.update(value=f"**{p.model_key} · {p.model_name}**\n\n{body}"))
    return ups


def _bubble(reasoning: str, content: str) -> str:
    """渲染助手气泡：推理折叠成灰色小字 <details>（默认收起），正文正常显示。
    还没有任何输出时显示"思考中…"占位，消除高首字延迟时的空白等待。"""
    parts = []
    if reasoning:
        summary = "💭 思考过程" if content else "💭 思考中…"
        parts.append(
            f"<details><summary>{summary}</summary>\n\n"
            f"<div style=\"color:#888;font-size:0.85em;white-space:pre-wrap\">{reasoning}</div>\n\n"
            f"</details>"
        )
    if content:
        parts.append(content)
    elif not reasoning:
        parts.append("思考中…")
    return "\n\n".join(parts)


# ----------------------- 会话侧栏 -----------------------
def on_load():
    metas = STORE.list_conversations()
    if not metas:
        return gr.update(choices=[], value=None), None, None, []
    cid = metas[0].id
    return gr.update(choices=_choices(), value=cid), cid, _load_history(cid), STORE.get_messages(cid)


def on_new():
    cid = STORE.create_conversation("新对话")
    return gr.update(choices=_choices(), value=cid), cid, _new_history(), []


def on_select(cid):
    if cid is None:
        return None, None, []
    return cid, _load_history(cid), STORE.get_messages(cid)


def on_delete(cid):
    if cid is not None:
        STORE.delete_conversation(cid)
    metas = STORE.list_conversations()
    if metas:
        nid = metas[0].id
        return gr.update(choices=_choices(), value=nid), nid, _load_history(nid), STORE.get_messages(nid)
    nid = STORE.create_conversation("新对话")
    return gr.update(choices=_choices(), value=nid), nid, _new_history(), []


# ----------------------- 折叠开关 -----------------------
def toggle_opts(is_open):
    return gr.update(visible=not is_open), (not is_open)


def toggle_detail(is_open):
    now = not is_open
    return gr.update(visible=now), now, gr.update(value=("收起过程 ◂" if now else "展开过程 ▸"))


# ----------------------- Fusion 事件 → 界面片段 -----------------------
async def _fusion_stream(question, web):
    """yield ('chat', 助手正文) / ('judge', md) / ('panels', results) / ('done', 最终研报)。"""
    registry = strategy = None
    if web:
        registry = _registry()
        strategy = NativeToolCalling()
    synth: list[str] = []
    yield ("chat", "⏳ 面板并行作答中…")
    async for ev in stream_fusion(
        question, list(CFG.default_panel), CFG, web=web, registry=registry, strategy=strategy
    ):
        if isinstance(ev, AgentDone):
            yield ("chat", f"⏳ {ev.model} 已完成…")
        elif isinstance(ev, AgentDegraded):
            yield ("chat", f"⚠ {ev.model} 降级：{ev.reason}")
        elif isinstance(ev, PanelFinished):
            yield ("panels", ev.results)
            yield ("chat", "⚖️ 裁判交叉验证中…")
        elif isinstance(ev, JudgeFinished):
            yield ("judge", _render_judge(ev.analysis))
            yield ("chat", "🤖 合成研报中…")
        elif isinstance(ev, SynthesisDelta):
            synth.append(ev.text)
            yield ("chat", "".join(synth))
        elif isinstance(ev, FusionDone):
            final = "".join(synth)
            if ev.sources:
                final += "\n\n**参考来源**\n" + "\n".join(
                    f"[{i}] {u}" for i, u in enumerate(ev.sources, 1)
                )
            yield ("done", final)
        elif isinstance(ev, StageError):
            yield ("chat", f"⚠ [{ev.stage}] 出错：{ev.detail}")


# ----------------------- 发送（四态路由）-----------------------
async def on_send(user_msg, fusion, web, conv_id, history, chat):
    user_msg = (user_msg or "").strip()
    chat = list(chat or [])
    judge_u = gr.update()
    panel_us = [gr.update() for _ in CFG.models]

    def out(msg_val, radio_val=gr.update()):
        return (msg_val, conv_id, history, chat, radio_val, judge_u, *panel_us)

    if not user_msg:
        yield out(gr.update())
        return
    if conv_id is None:
        conv_id = STORE.create_conversation("新对话")
        history = _new_history()
    elif history is None:
        history = _load_history(conv_id)

    titles = {c.id: c.title for c in STORE.list_conversations()}
    set_title = titles.get(conv_id) in (None, "新对话")
    history.add_user(user_msg)
    chat = chat + [{"role": "user", "content": user_msg}, {"role": "assistant", "content": ""}]

    # 联网但没配 key：直接给出提示，不进流程
    if web and not CFG.active_search_key():
        final = f"联网未启用：未配置 {CFG.search_provider} 的搜索 key（.env 设 {CFG.search_provider.upper()}_API_KEY）。"
        chat[-1]["content"] = final
        history.add_assistant(final)
        persist_turn(STORE, conv_id, history, user_msg, final)
        yield out("", gr.update(choices=_choices(), value=conv_id))
        return

    final = ""
    try:
        if fusion:
            async for kind, payload in _fusion_stream(user_msg, web):
                if kind == "chat":
                    chat[-1]["content"] = payload
                elif kind == "judge":
                    judge_u = gr.update(value=payload)
                elif kind == "panels":
                    panel_us = _panel_updates(payload)
                elif kind == "done":
                    chat[-1]["content"] = payload
                    final = payload
                yield out("")
        elif web:
            acc: list[str] = []
            chat[-1]["content"] = "思考中…"
            yield out("")
            async for ev in stream_agent(
                CFG.get(CFG.default_model), user_msg, _registry(),
                NativeToolCalling(), max_steps=CFG.web_max_steps,
            ):
                if isinstance(ev, TextDelta):
                    acc.append(ev.text)
                    chat[-1]["content"] = "".join(acc)
                    yield out("")
                elif isinstance(ev, ToolStarted):
                    chat[-1]["content"] = ("".join(acc) or "") + "\n\n🔍 联网检索中…"
                    yield out("")
                elif isinstance(ev, AgentDone):
                    final = ev.content or "".join(acc)
                    chat[-1]["content"] = final
                    yield out("")
        else:
            sum_cfg = CFG.get(CFG.summary_model or CFG.default_model)
            sum_task = None
            if history.compact():
                sum_task = asyncio.create_task(history.summarize_pending(sum_cfg))
            think: list[str] = []
            ans: list[str] = []
            chat[-1]["content"] = "思考中…"
            yield out("")
            try:
                async for kind, piece in providers.stream_rich(
                    CFG.get(CFG.default_model), history.build_messages()
                ):
                    (think if kind == "reasoning" else ans).append(piece)
                    chat[-1]["content"] = _bubble("".join(think), "".join(ans))
                    yield out("")
            finally:
                if sum_task is not None:
                    await asyncio.gather(sum_task, return_exceptions=True)
            final = "".join(ans)  # 仅正文入库/历史；推理流是展示用、不持久化
    except Exception as e:  # noqa: BLE001
        final = (chat[-1]["content"] or "") + f"\n\n[调用失败：{e}]"
        chat[-1]["content"] = final
        yield out("")

    history.add_assistant(final)
    persist_turn(STORE, conv_id, history, user_msg, final)
    if set_title:
        STORE.rename_conversation(conv_id, user_msg[:20])
    yield out("", gr.update(choices=_choices(), value=conv_id))


# ----------------------- 界面 -----------------------
_CSS = """
.scrollbox-judge {max-height: 180px; overflow-y: auto;}
.scrollbox-panel {max-height: 360px; overflow-y: auto;}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Fusion Agent", css=_CSS) as demo:  # 主题在 launch() 传（Gradio 6）
        conv_state = gr.State(None)
        hist_state = gr.State(None)
        opts_open = gr.State(False)
        detail_open = gr.State(False)

        gr.Markdown("## 🤝 Fusion Agent")
        with gr.Row():
            # 左：会话列表
            with gr.Column(scale=1, min_width=200):
                new_btn = gr.Button("➕ 新建对话", variant="primary")
                conv_radio = gr.Radio(choices=_choices(), label="对话", value=None, interactive=True)
                del_btn = gr.Button("🗑 删除当前对话", variant="stop")
            # 中：聊天
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=560, label="对话")
                with gr.Row(visible=False) as opts_row:
                    fusion_chk = gr.Checkbox(label="Fusion 多模型", value=False)
                    web_chk = gr.Checkbox(label="联网", value=False)
                with gr.Row():
                    plus_btn = gr.Button("➕", scale=0, min_width=48)
                    msg = gr.Textbox(placeholder="输入消息，回车发送…  （点 ➕ 开启 Fusion / 联网）",
                                     show_label=False, scale=8, autofocus=True)
                    detail_btn = gr.Button("展开过程 ▸", scale=0, min_width=110)
            # 右：Fusion 详情（默认隐藏；裁判在上，面板按模型分标签页在下，均限高内部滚动）
            with gr.Column(scale=2, visible=False, elem_id="detail_col") as detail_col:
                gr.Markdown("### ⚖️ 裁判裁决")
                judge_md = gr.Markdown("", elem_classes=["scrollbox-judge"])
                gr.Markdown("### 🧩 面板原文")
                panel_mds = []
                with gr.Tabs():
                    for _k in CFG.models:
                        with gr.Tab(CFG.models[_k].name or _k):
                            panel_mds.append(gr.Markdown("", elem_classes=["scrollbox-panel"]))

        # 会话接线
        demo.load(on_load, outputs=[conv_radio, conv_state, hist_state, chatbot])
        new_btn.click(on_new, outputs=[conv_radio, conv_state, hist_state, chatbot])
        conv_radio.change(on_select, inputs=conv_radio, outputs=[conv_state, hist_state, chatbot])
        del_btn.click(on_delete, inputs=conv_state, outputs=[conv_radio, conv_state, hist_state, chatbot])
        # 折叠开关
        plus_btn.click(toggle_opts, inputs=opts_open, outputs=[opts_row, opts_open])
        detail_btn.click(toggle_detail, inputs=detail_open, outputs=[detail_col, detail_open, detail_btn])
        # 发送
        send_outputs = [msg, conv_state, hist_state, chatbot, conv_radio, judge_md, *panel_mds]
        msg.submit(on_send, inputs=[msg, fusion_chk, web_chk, conv_state, hist_state, chatbot], outputs=send_outputs)
    return demo


if __name__ == "__main__":
    build().launch(theme=gr.themes.Soft(primary_hue="orange"))
