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
import shutil

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from tracing import init_tracing, phoenix_url  # noqa: E402

init_tracing()  # 默认 no-op；设 FUSION_TRACING 后自动追踪所有模型调用（并进程内自动起 Phoenix）

import providers  # noqa: E402
from agent import stream_agent  # noqa: E402
from chat_context import ChatHistory  # noqa: E402
from config import apply_overrides, load_config, save_user_settings  # noqa: E402
from fusion import stream_fusion  # noqa: E402
from router import ROUTE_LABELS, route  # noqa: E402
from search_providers import available_providers, create_provider  # noqa: E402
from store import ConversationStore, load_history, persist_turn  # noqa: E402
from stream_events import (  # noqa: E402
    AgentDegraded,
    AgentDone,
    FusionDone,
    JudgeFinished,
    PanelFinished,
    ReflexionStarted,
    StageError,
    SynthesisDelta,
    TextDelta,
    ToolStarted,
)
from rag import build_kb, build_rag_tool  # noqa: E402
from tool_calling import NativeToolCalling  # noqa: E402
from tools import ToolRegistry, build_web_registry  # noqa: E402

CFG = load_config()
STORE = ConversationStore(os.getenv("FUSION_DB", "fusion_agent.db"))

KB = None  # 本地知识库（进程内建一次；需 RAG_ENABLED + EMBED_MODEL，且已 `python ingest.py` 建索引）
if CFG.rag_ready:
    try:
        KB = build_kb(CFG)
    except Exception as e:  # noqa: BLE001
        print(f"[rag] 本地知识库初始化失败，已禁用本地检索：{e}")


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


def _agent_registry(*, web: bool, rag: bool):
    """单模型 agent 的工具集：web → web_search/web_fetch；rag → local_search（可叠加）。"""
    reg = _registry() if web else ToolRegistry()
    if rag and KB is not None:
        reg.register(build_rag_tool(KB, default_k=CFG.rag_top_k))
    return reg


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


def _send_begin():
    """进入生成态：隐藏「发送」、显示「停止」、禁用输入框（防回车重复提交）。"""
    return gr.update(visible=False), gr.update(visible=True), gr.update(interactive=False)


def _send_end():
    """生成结束 / 被停止：复原「发送」、隐藏「停止」、重新启用输入框。"""
    return gr.update(visible=True), gr.update(visible=False), gr.update(interactive=True)


def _excl_auto(on):
    """勾「自动选择」→ 清掉其余三个手动开关（仅在勾上时动作，避免与 _excl_manual 回环）。"""
    if on:
        return gr.update(value=False), gr.update(value=False), gr.update(value=False)
    return gr.update(), gr.update(), gr.update()


def _excl_manual(on):
    """勾任一手动开关（Fusion/联网/本地知识库）→ 清掉「自动选择」。"""
    return gr.update(value=False) if on else gr.update()


# ----------------------- 设置（user_settings.json 覆盖层）-----------------------
_ROLE_KEYS = ["default_model", "judge_model", "synthesizer_model", "summary_model", "router_model"]
_ROLE_LABELS = ["默认对话", "裁判", "合成", "摘要", "路由"]
# 设置页表单顺序里的 9 个数字项（与 build() 里 _nums 的创建顺序、_collect 的解析顺序严格一致）
_SET_NUM_KEYS = ["history_keep_recent", "judge_answer_char_limit", "judge_evidence_max_items",
                 "web_max_steps", "web_max_results", "rag_top_k", "reflexion_max_rounds",
                 "rag_chunk_size", "rag_chunk_overlap"]
# 上下文预算离散挡位（避免拖出奇怪数字）
_BUDGET_PRESETS = [8000, 16000, 32000, 64000, 128000, 256000]
_BUDGET_LABELS = ["8k", "16k", "32k", "64k", "128k", "256k"]


def _budget_idx(v) -> int:
    v = int(v or 8000)
    return min(range(len(_BUDGET_PRESETS)), key=lambda i: abs(_BUDGET_PRESETS[i] - v))


def _collect_settings_overrides(roles, budget_idx, nums, reflexion_en, rag_en, provider,
                                tavily, bocha, emb_model, emb_base, emb_key,
                                rag_store, rag_coll, rag_docs, panel, models) -> dict:
    """把设置表单的值组装成覆盖 dict（纯函数，便于测试）。空 Key/URL 不覆盖；数字转 int。"""
    ov: dict = {}
    for k, val in zip(_ROLE_KEYS, roles):
        if val:
            ov[k] = val
    try:
        ov["history_char_budget"] = _BUDGET_PRESETS[int(budget_idx)]
    except (TypeError, ValueError, IndexError):
        pass
    for k, val in zip(_SET_NUM_KEYS, nums):
        if val is not None and str(val) != "":
            try:
                ov[k] = int(val)
            except (TypeError, ValueError):
                pass
    ov["reflexion_enabled"] = bool(reflexion_en)
    ov["rag_enabled"] = bool(rag_en)
    if provider:
        ov["search_provider"] = provider
    sk = {}
    if str(tavily or "").strip():
        sk["tavily"] = str(tavily).strip()
    if str(bocha or "").strip():
        sk["bocha"] = str(bocha).strip()
    if sk:
        ov["search_keys"] = sk
    emb = {}
    if str(emb_model or "").strip():
        emb["model"] = str(emb_model).strip()
    if str(emb_base or "").strip():
        emb["base_url"] = str(emb_base).strip()
    if str(emb_key or "").strip():
        emb["api_key"] = str(emb_key).strip()
    if emb:
        ov["embed"] = emb
    if str(rag_store or "").strip():
        ov["rag_store"] = str(rag_store).strip()
    if str(rag_coll or "").strip():
        ov["rag_collection"] = str(rag_coll).strip()
    if str(rag_docs or "").strip():
        ov["rag_docs_dir"] = str(rag_docs).strip()
    if panel:
        ov["default_panel"] = list(panel)
    # 模型全集 diff：UI 里有的 → 改/增；现有但 UI 没了 → 删（null）
    if models is not None:
        m_ov: dict = {}
        ui_keys = set()
        for m in models:
            mk = (m.get("key") or "").strip()
            if not mk:
                continue
            ui_keys.add(mk)
            entry: dict = {"supports_tools": bool(m.get("supports_tools", True))}
            for fk in ("name", "base_url", "api_key"):
                if str(m.get(fk) or "").strip():
                    entry[fk] = str(m[fk]).strip()
            m_ov[mk] = entry
        for mk in list(CFG.models):
            if mk not in ui_keys:
                m_ov[mk] = None
        if m_ov:
            ov["models"] = m_ov
    return ov


def _settings_save(*vals):
    """保存设置：写 user_settings.json + 就地热更内存 CFG；模型/接口改动提示刷新连接池。"""
    v = list(vals)
    n = len(_SET_NUM_KEYS)
    roles = v[0:5]
    budget_idx = v[5]
    nums = v[6:6 + n]
    j = 6 + n
    reflexion_en, rag_en = v[j], v[j + 1]
    provider, tavily, bocha = v[j + 2], v[j + 3], v[j + 4]
    emb_model, emb_base, emb_key = v[j + 5], v[j + 6], v[j + 7]
    rag_store, rag_coll, rag_docs = v[j + 8], v[j + 9], v[j + 10]
    panel = v[j + 11]
    models = v[j + 12] if len(v) > j + 12 else None
    ov = _collect_settings_overrides(roles, budget_idx, nums, reflexion_en, rag_en, provider,
                                     tavily, bocha, emb_model, emb_base, emb_key,
                                     rag_store, rag_coll, rag_docs, panel, models)
    save_user_settings(ov)
    apply_overrides(CFG, ov)
    note = ("；模型/嵌入/搜索接口有改动，建议点「🔄 刷新连接池」"
            if any(k in ov for k in ("models", "embed", "search_keys")) else "")
    gr.Info("设置已保存并热生效" + note)
    return f"✅ 已保存并热生效{note}"


async def _refresh_pool():
    """清空 providers 的客户端缓存：下次调用按新接口配置重建连接。"""
    await providers.aclose_all()
    gr.Info("已刷新模型连接池（下次调用按新接口配置重建客户端）")
    return "🔄 连接池已刷新。"


# ----------------------- 知识库管理（📚）-----------------------
def _kb_rows():
    """当前向量库文档列表 → Dataframe 行 [文件, 片段数, 来源路径]。"""
    if KB is None:
        return []
    return [[name, n, src] for (name, src, n) in KB.list_documents()]


async def _kb_upload(files, progress=gr.Progress()):
    """上传 → 拷进 RAG_DOCS_DIR → 逐个 ingest（生成器 yield 进度 + gr.Progress 进度条，避免 UI 假死）。

    据实区分 成功 / 失败 / 空(0 块，如扫描版无文本 PDF)；失败不再显示成功。
    """
    if KB is None:
        yield "本地知识库未启用。", gr.update()
        return
    if not files:
        yield "（未选择文件）", gr.update()
        return
    os.makedirs(CFG.rag_docs_dir, exist_ok=True)
    total = len(files)
    ok: list[str] = []
    failed: list[str] = []
    for i, f in enumerate(files):
        src_path = getattr(f, "name", None) or f       # gr.File 给临时文件路径
        base = os.path.basename(src_path)
        dest = os.path.join(CFG.rag_docs_dir, base)
        try:
            progress((i, total), desc=f"处理 {base}")
        except Exception:  # noqa: BLE001  非 Gradio 运行（测试）时 progress 不可用
            pass
        yield f"⏳ 处理中：{base} …（{i + 1}/{total}）", gr.update()
        try:
            if os.path.abspath(src_path) != os.path.abspath(dest):
                shutil.copy(src_path, dest)
            n = await KB.ingest_file(dest)
            if n > 0:
                ok.append(f"{base}（{n} 块）")
            else:
                failed.append(f"{base}（未提取到文本）")     # 0 块＝没真入库
        except Exception as e:  # noqa: BLE001
            failed.append(f"{base}（{e}）")
    try:
        progress((total, total), desc="完成")
    except Exception:  # noqa: BLE001
        pass
    if failed and ok:
        msg = f"⚠ 入库结束：成功 {len(ok)}，失败 {len(failed)}。失败：{'；'.join(failed)}"
    elif failed:
        msg = f"❌ 入库失败（{len(failed)}/{total}）：{'；'.join(failed)}"
    else:
        msg = f"✅ 全部入库成功：{len(ok)}/{total}（{'，'.join(ok)}）"
    yield msg, gr.update(value=_kb_rows())
    await providers.aclose_all()


def _kb_select(df, evt: gr.SelectData):
    """记录选中行的来源路径（第 3 列），供删除用。"""
    try:
        row = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        return _kb_rows()[row][2]
    except (IndexError, TypeError):
        return None


def _kb_delete(src):
    """删除选中文档：清向量库该来源全部 chunk + 删源文件（二次确认在前端 JS）。"""
    if not src:
        gr.Warning("请先在列表里点选要删除的文档。")
        return gr.update(), "（未选择文档）"
    n = KB.delete_document(src) if KB is not None else 0
    try:
        if os.path.isfile(src):
            os.remove(src)
    except OSError:
        pass
    gr.Info(f"已删除文档及其 {n} 个片段。")
    return gr.update(value=_kb_rows()), f"🗑 已删除：{os.path.basename(src)}（{n} 块）"


def _build_kb_tab():
    """📚 知识库管理：上传 / 列表 / 删除。KB 未就绪时只显示提示。"""
    if KB is None:
        gr.Markdown(
            "本地知识库未启用：请在 `.env` 设 `RAG_ENABLED=1` 与 `EMBED_MODEL`（嵌入端点），"
            "重启应用后即可在此上传 / 管理文档。"
        )
        return
    gr.Markdown("#### 📚 本地知识库（上传即自动切块嵌入入库；点选某行后可删除）")
    kb_status = gr.Markdown("")
    kb_upload = gr.File(label="上传文档（.md / .txt / .pdf，可拖拽 / 多选）",
                        file_count="multiple", file_types=[".md", ".txt", ".pdf"])
    with gr.Row():
        kb_refresh_btn = gr.Button("🔄 刷新列表")
        kb_del_btn = gr.Button("🗑 删除选中（连源文件）", variant="stop")
    kb_df = gr.Dataframe(headers=["文件", "片段数", "来源路径"], datatype=["str", "number", "str"],
                         value=_kb_rows(), interactive=False, label="已入库文档")
    sel_src = gr.State(None)

    kb_upload.upload(_kb_upload, inputs=kb_upload, outputs=[kb_status, kb_df])
    kb_refresh_btn.click(lambda: gr.update(value=_kb_rows()), outputs=kb_df)
    kb_df.select(_kb_select, inputs=kb_df, outputs=sel_src)
    kb_del_btn.click(
        _kb_delete, inputs=sel_src, outputs=[kb_df, kb_status],
        js="(s) => { if(!s){ alert('请先点选要删除的文档'); throw 0; } "
           "if(!confirm('将同时删除源文件和向量索引，不可恢复。确定删除？')) throw 0; return s; }",
    )


# ----------------------- Fusion 事件 → 界面片段 -----------------------
async def _fusion_stream(question, web, rag):
    """yield ('chat', 助手正文) / ('judge', md) / ('panels', results) / ('done', 最终研报)。

    web 或 rag 任一开启 → 面板以 agent 方式运行（带 web_search/web_fetch 和/或 local_search）。
    """
    rag_on = rag and KB is not None
    agentic = web or rag_on
    registry = strategy = None
    if agentic:
        registry = _registry() if web else ToolRegistry()
        if rag_on:
            registry.register(build_rag_tool(KB, default_k=CFG.rag_top_k))
        strategy = NativeToolCalling()
    synth: list[str] = []
    yield ("chat", "⏳ 面板并行作答中…")
    async for ev in stream_fusion(
        question, list(CFG.default_panel), CFG, web=agentic, registry=registry, strategy=strategy
    ):
        if isinstance(ev, AgentDone):
            yield ("chat", f"⏳ {ev.model} 已完成…")
        elif isinstance(ev, AgentDegraded):
            yield ("chat", f"⚠ {ev.model} 降级：{ev.reason}")
        elif isinstance(ev, ReflexionStarted):
            yield ("chat", f"🔍 {ev.model} 自检证据中…")
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
async def on_send(user_msg, fusion, web, auto, rag, conv_id, history, chat):
    user_msg = (user_msg or "").strip()
    chat = list(chat or [])
    rag_on = bool(rag) and KB is not None
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

    # 「自动」开关：由语义路由决定走 直答/联网单模型/离线 Fusion（覆盖两个手动勾），并回显所选路由
    route_note = ""
    if auto:
        try:
            label = await route(user_msg, CFG)
        except Exception:  # noqa: BLE001
            label = "direct"
        rag_ok = CFG.rag_ready
        fusion, web, rag = {
            "web": (False, True, rag_ok),      # 联网 agent，rag 就绪则也挂 local_search
            "fusion": (True, False, rag_ok),   # Fusion 面板，rag 就绪则面板也带 local_search
            "local": (False, False, True),     # 纯本地知识库 agent
            "direct": (False, False, False),
        }.get(label, (False, False, False))
        rag_on = bool(rag) and KB is not None
        route_note = f"🧭 自动路由 → {ROUTE_LABELS.get(label, label)}\n\n"
        chat[-1]["content"] = route_note + "（处理中…）"
        yield out("")

    # 联网但没配 key：直接给出提示，不进流程
    if web and not CFG.active_search_key():
        final = route_note + f"联网未启用：未配置 {CFG.search_provider} 的搜索 key（.env 设 {CFG.search_provider.upper()}_API_KEY）。"
        chat[-1]["content"] = final
        history.add_assistant(final)
        persist_turn(STORE, conv_id, history, user_msg, final)
        yield out("", gr.update(choices=_choices(), value=conv_id))
        return

    final = ""
    persisted = False

    def _persist(text: str):
        """落库一次（幂等）：正常完成 / 出错 / 被「停止」中断都走这里，避免 history、store 出现"有问无答"的悬挂。"""
        nonlocal persisted
        if persisted:
            return
        t = text
        if route_note and not t.startswith(route_note):
            t = route_note + t
            chat[-1]["content"] = t   # 仅自动路由回显时改写气泡；普通对话保留含推理折叠的气泡
        history.add_assistant(t)
        persist_turn(STORE, conv_id, history, user_msg, t)
        if set_title:
            STORE.rename_conversation(conv_id, user_msg[:20])
        persisted = True

    try:
        if fusion:
            async for kind, payload in _fusion_stream(user_msg, web, rag):
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
        elif web or rag_on:
            acc: list[str] = []
            chat[-1]["content"] = "思考中…"
            yield out("")
            async for ev in stream_agent(
                CFG.get(CFG.default_model), user_msg, _agent_registry(web=web, rag=rag_on),
                NativeToolCalling(), max_steps=CFG.web_max_steps,
                reflexion=CFG.reflexion_enabled, reflexion_rounds=CFG.reflexion_max_rounds,
            ):
                if isinstance(ev, TextDelta):
                    acc.append(ev.text)
                    chat[-1]["content"] = "".join(acc)
                    yield out("")
                elif isinstance(ev, ToolStarted):
                    tip = "📚 查本地知识库中…" if ev.name == "local_search" else "🔍 联网检索中…"
                    chat[-1]["content"] = ("".join(acc) or "") + "\n\n" + tip
                    yield out("")
                elif isinstance(ev, ReflexionStarted):
                    chat[-1]["content"] = ("".join(acc) or "") + "\n\n🔍 自检证据、修订中…"
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
                    final = "".join(ans)  # 持续更新纯正文，便于中断时落库（不含推理）
                    chat[-1]["content"] = _bubble("".join(think), "".join(ans))
                    yield out("")
            finally:
                if sum_task is not None:
                    await asyncio.gather(sum_task, return_exceptions=True)
            final = "".join(ans)  # 仅正文入库/历史；推理流是展示用、不持久化
        _persist(final)
        yield out("", gr.update(choices=_choices(), value=conv_id))
    except Exception as e:  # noqa: BLE001  真正的调用错误（CancelledError/GeneratorExit 是 BaseException，不在此捕获）
        final = (chat[-1]["content"] or "") + f"\n\n[调用失败：{e}]"
        _persist(final)
        yield out("", gr.update(choices=_choices(), value=conv_id))
    finally:
        # 被「停止」中断时上面未跑完（GeneratorExit/CancelledError 直达 finally）→ 落库已生成的部分
        if not persisted:
            _persist(final or (chat[-1]["content"] if chat else ""))


# ----------------------- 界面 -----------------------
_CSS = """
.scrollbox-judge {max-height: 180px; overflow-y: auto;}
.scrollbox-panel {max-height: 360px; overflow-y: auto;}
/* 对话列表：一行一条，整行可点，长标题省略号截断 */
.conv-radio .wrap {flex-direction: column !important; gap: 2px;}
.conv-radio label {width: 100%; display: flex; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
"""


def build() -> gr.Blocks:
    with gr.Blocks(title="Fusion Agent", css=_CSS) as demo:  # 主题在 launch() 传（Gradio 6）
        conv_state = gr.State(None)
        hist_state = gr.State(None)
        opts_open = gr.State(False)
        detail_open = gr.State(False)

        gr.Markdown("## 🤝 Fusion Agent")
        with gr.Tabs():
            with gr.Tab("💬 对话"):
                with gr.Row():
                    # 左：会话列表
                    with gr.Column(scale=1, min_width=200):
                        new_btn = gr.Button("➕ 新建对话", variant="primary")
                        conv_radio = gr.Radio(choices=_choices(), label="对话", value=None,
                                              interactive=True, elem_classes=["conv-radio"])
                        del_btn = gr.Button("🗑 删除当前对话", variant="stop")
                        _purl = phoenix_url()  # 仅在开了追踪并起了 Phoenix 时显示监控入口
                        if _purl:
                            phoenix_btn = gr.Button("🔍 打开监控面板 (Phoenix)", variant="secondary")
                            phoenix_btn.click(None, js=f"() => window.open('{_purl}', '_blank')")
                    # 中：聊天
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(height=560, label="对话")
                        with gr.Row(visible=False) as opts_row:
                            auto_chk = gr.Checkbox(label="自动选择", value=False)
                            fusion_chk = gr.Checkbox(label="Fusion 多模型", value=False)
                            web_chk = gr.Checkbox(label="联网", value=False)
                            rag_chk = gr.Checkbox(label="本地知识库", value=False, visible=CFG.rag_ready)
                        with gr.Row():
                            plus_btn = gr.Button("➕", scale=0, min_width=48)
                            msg = gr.Textbox(placeholder="输入消息，回车或点「发送」…  （点 ➕ 开启 Fusion / 联网 / 自动）",
                                             show_label=False, scale=8, autofocus=True)
                            send_btn = gr.Button("发送", scale=0, min_width=80, variant="primary")
                            stop_btn = gr.Button("⏹ 停止", scale=0, min_width=80, variant="stop", visible=False)
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

            with gr.Tab("⚙️ 设置"):
                set_status = gr.Markdown("调整后点「💾 保存」：写入 user_settings.json 并热生效（不动 .env）。")
                gr.Markdown("#### 角色分工")
                with gr.Row():
                    _role_dd = [gr.Dropdown(choices=list(CFG.models), value=getattr(CFG, rk) or None,
                                            label=lbl, interactive=True)
                                for rk, lbl in zip(_ROLE_KEYS, _ROLE_LABELS)]
                gr.Markdown("#### 上下文预算（挡位）")
                _budget = gr.Slider(0, len(_BUDGET_PRESETS) - 1, value=_budget_idx(CFG.history_char_budget),
                                    step=1, label="拖动选挡位（8k–256k）")
                _budget_lbl = gr.Markdown(f"当前预算：**{_BUDGET_LABELS[_budget_idx(CFG.history_char_budget)]}**")
                gr.Markdown("#### 系统参数")
                with gr.Row():
                    _n_keep = gr.Number(value=CFG.history_keep_recent, label="保留最近(条)", precision=0)
                    _n_jans = gr.Number(value=CFG.judge_answer_char_limit, label="裁判答案上限", precision=0)
                    _n_jev = gr.Number(value=CFG.judge_evidence_max_items, label="裁判信源上限", precision=0)
                    _n_wsteps = gr.Number(value=CFG.web_max_steps, label="联网最大轮数", precision=0)
                with gr.Row():
                    _n_wres = gr.Number(value=CFG.web_max_results, label="每次搜索结果数", precision=0)
                    _n_topk = gr.Number(value=CFG.rag_top_k, label="RAG 召回数", precision=0)
                    _n_refl = gr.Number(value=CFG.reflexion_max_rounds, label="反思轮数", precision=0)
                    _n_chunk = gr.Number(value=CFG.rag_chunk_size, label="RAG 分块字符", precision=0)
                with gr.Row():
                    _n_overlap = gr.Number(value=CFG.rag_chunk_overlap, label="RAG 分块重叠", precision=0)
                _nums = [_n_keep, _n_jans, _n_jev, _n_wsteps, _n_wres, _n_topk, _n_refl, _n_chunk, _n_overlap]
                gr.Markdown("#### 开关")
                with gr.Row():
                    _reflexion_chk = gr.Checkbox(value=CFG.reflexion_enabled, label="反思 Reflexion")
                    _rag_en_chk = gr.Checkbox(value=CFG.rag_enabled, label="本地 RAG 启用")
                gr.Markdown("#### 联网搜索")
                with gr.Row():
                    _provider_dd = gr.Dropdown(choices=(available_providers() or ["tavily", "bocha"]),
                                               value=CFG.search_provider, label="搜索后端")
                    _tavily_tb = gr.Textbox(value="", label="Tavily Key", type="password", placeholder="留空＝不改")
                    _bocha_tb = gr.Textbox(value="", label="Bocha Key", type="password", placeholder="留空＝不改")
                gr.Markdown("#### 本地 RAG / 嵌入")
                with gr.Row():
                    _emb_model_tb = gr.Textbox(value=(CFG.embed_cfg.name if CFG.embed_cfg else ""), label="嵌入模型")
                    _emb_base_tb = gr.Textbox(value=(CFG.embed_cfg.base_url if CFG.embed_cfg else ""), label="嵌入 Base URL")
                    _emb_key_tb = gr.Textbox(value="", label="嵌入 API Key", type="password", placeholder="留空＝不改")
                with gr.Row():
                    _ragstore_tb = gr.Textbox(value=CFG.rag_store, label="向量库后端")
                    _ragcoll_tb = gr.Textbox(value=CFG.rag_collection, label="Collection")
                    _ragdocs_tb = gr.Textbox(value=CFG.rag_docs_dir, label="文档目录")
                gr.Markdown("#### 面板模型（Fusion 默认面板）")
                _panel_cbg = gr.CheckboxGroup(choices=list(CFG.models), value=list(CFG.default_panel),
                                              label="勾选作为面板模型")
                gr.Markdown("#### 模型与 API（API Key 留空＝不改；可增删）")
                _models_state = gr.State([
                    {"key": k, "name": m.name, "base_url": m.base_url, "api_key": "",
                     "supports_tools": m.supports_tools}
                    for k, m in CFG.models.items()
                ])

                @gr.render(inputs=_models_state)
                def _render_models(models):
                    for _idx, _m in enumerate(models):
                        with gr.Row():
                            _nm = gr.Textbox(value=_m["name"], label=f"{_m['key']} · Name", scale=2)
                            _bu = gr.Textbox(value=_m["base_url"], label="Base URL", scale=3)
                            _ak = gr.Textbox(value=_m.get("api_key", ""), label="API Key",
                                             type="password", placeholder="留空＝不改", scale=2)
                            _st = gr.Checkbox(value=_m.get("supports_tools", True), label="tools",
                                              scale=0, min_width=70)
                            _dl = gr.Button("🗑", scale=0, min_width=44)

                        def _commit(name, base, key, sup, i=_idx, cur=models):
                            nxt = [dict(x) for x in cur]
                            nxt[i] = {**nxt[i], "name": name, "base_url": base,
                                      "api_key": key, "supports_tools": sup}
                            return nxt

                        for _c in (_nm, _bu, _ak):
                            _c.blur(_commit, inputs=[_nm, _bu, _ak, _st], outputs=_models_state)
                        _st.change(_commit, inputs=[_nm, _bu, _ak, _st], outputs=_models_state)
                        _dl.click(lambda i=_idx, cur=models: [x for j, x in enumerate(cur) if j != i],
                                  outputs=_models_state)

                _add_model_btn = gr.Button("➕ 添加模型")

                def _add_model(models):
                    used = {m["key"] for m in models} | set(CFG.models)
                    n = 1
                    while f"model{n}" in used:
                        n += 1
                    return models + [{"key": f"model{n}", "name": "", "base_url": "",
                                      "api_key": "", "supports_tools": True}]

                _add_model_btn.click(_add_model, inputs=_models_state, outputs=_models_state)
                with gr.Row():
                    save_btn = gr.Button("💾 保存设置", variant="primary")
                    pool_btn = gr.Button("🔄 刷新连接池")
                _budget.change(lambda i: f"当前预算：**{_BUDGET_LABELS[int(i)]}**",
                               inputs=_budget, outputs=_budget_lbl)

            with gr.Tab("📚 知识库"):
                kb_tab_body = _build_kb_tab()

        # 会话接线
        demo.load(on_load, outputs=[conv_radio, conv_state, hist_state, chatbot])
        new_btn.click(on_new, outputs=[conv_radio, conv_state, hist_state, chatbot])
        conv_radio.change(on_select, inputs=conv_radio, outputs=[conv_state, hist_state, chatbot])
        del_btn.click(on_delete, inputs=conv_state, outputs=[conv_radio, conv_state, hist_state, chatbot])
        # 折叠开关
        plus_btn.click(toggle_opts, inputs=opts_open, outputs=[opts_row, opts_open])
        detail_btn.click(toggle_detail, inputs=detail_open, outputs=[detail_col, detail_open, detail_btn])
        # 「自动选择」与三个手动开关互斥（仅在勾上时清对方，避免回环）
        auto_chk.change(_excl_auto, inputs=auto_chk, outputs=[fusion_chk, web_chk, rag_chk])
        for _chk in (fusion_chk, web_chk, rag_chk):
            _chk.change(_excl_manual, inputs=_chk, outputs=auto_chk)
        # 发送 / 停止：发送或回车先切「停止」态并禁用输入，跑 on_send，结束复原；
        # 「停止」用 cancels 取消正在跑的生成（中断模型输出），并复原按钮。
        send_inputs = [msg, fusion_chk, web_chk, auto_chk, rag_chk, conv_state, hist_state, chatbot]
        send_outputs = [msg, conv_state, hist_state, chatbot, conv_radio, judge_md, *panel_mds]
        toggle_io = [send_btn, stop_btn, msg]
        gen_click = send_btn.click(_send_begin, None, toggle_io).then(on_send, send_inputs, send_outputs)
        gen_click.then(_send_end, None, toggle_io)
        gen_submit = msg.submit(_send_begin, None, toggle_io).then(on_send, send_inputs, send_outputs)
        gen_submit.then(_send_end, None, toggle_io)
        stop_btn.click(_send_end, None, toggle_io, cancels=[gen_click, gen_submit])
        # 设置：保存（写 user_settings.json + 热更内存 CFG）/ 刷新连接池
        save_inputs = (_role_dd + [_budget] + _nums
                       + [_reflexion_chk, _rag_en_chk, _provider_dd, _tavily_tb, _bocha_tb,
                          _emb_model_tb, _emb_base_tb, _emb_key_tb, _ragstore_tb, _ragcoll_tb,
                          _ragdocs_tb, _panel_cbg, _models_state])
        save_btn.click(_settings_save, inputs=save_inputs, outputs=set_status)
        pool_btn.click(_refresh_pool, outputs=set_status)
    return demo


if __name__ == "__main__":
    build().launch(theme=gr.themes.Soft(primary_hue="orange"))
