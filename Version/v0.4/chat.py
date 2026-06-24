"""CLI 主循环：普通对话(流式) + 指令解析(/fusionmodel、/fusionweb 均流式)。

渲染（按用户确认）：静默统计 + 节点汇总 + 视觉分隔，不用 \\r 覆盖刷新。
- 工具事件不逐条打印；每个模型完成时打一行汇总。
- 合成开始前打醒目分隔线，把"后台过程"和"前台研报"切断。
- 末尾灰字打印统计。参考来源按裁判 source_mapping 渲染。
- /verbose 时在面板/裁判节点额外打印面板原文与裁决。
"""
from __future__ import annotations

import asyncio
import time

import providers
from agent import stream_agent
from chat_context import ChatHistory
from config import AppConfig
from fusion import stream_fusion
from rag import build_kb, build_rag_tool
from router import ROUTE_LABELS, route
from search_providers import available_providers, create_provider
from stream_events import (
    AgentDegraded,
    AgentDone,
    FusionDone,
    JudgeFinished,
    PanelFinished,
    PanelStarted,
    ReflexionStarted,
    StageError,
    SynthesisDelta,
    SynthesisStarted,
    TextDelta,
    ToolStarted,
)
from tool_calling import NativeToolCalling
from tools import ToolRegistry, build_web_registry

HELP = """\
可用指令：
  /fusionmodel <问题>                  离线 Fusion（并行作答 -> 裁判 -> 合成，流式）
  /fusionmodel model1,model2 <问题>    指定面板的离线 Fusion
  /fusionweb <问题>                    联网 Fusion（面板 agent 自主检索 + 流式合成）
  /fusionweb model1,model2 <问题>      指定面板的联网 Fusion
  /auto <问题>                         语义路由：自动判定走 直答 / 联网单模型 / 离线 Fusion
  /kb <问题>                           查本地知识库（local_search agent；需 RAG_ENABLED + 先 ingest.py 建索引）
  /search [provider]                   查看 / 切换搜索后端（tavily、bocha…）
  /models                              列出已配置的模型与角色分工
  /verbose                             开关详细模式（展示面板原文 + 裁判裁决）
  /context                             查看普通对话的记忆摘要与上下文占用
  /reset                               清空普通对话历史（含记忆摘要）
  /help                                显示本帮助
  /exit                                退出
直接输入文字（不以 / 开头）即为普通多轮对话（流式），使用默认模型。
"""


class ChatSession:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.ctx = ChatHistory(
            char_budget=cfg.history_char_budget, keep_recent=cfg.history_keep_recent
        )
        self.verbose = False
        self._kb = None  # 本地知识库（惰性建一次）

    # -------------------- 普通对话（流式） --------------------
    def _chat_stream(self, text: str):
        self.ctx.add_user(text)
        cfg = self.cfg.get(self.cfg.default_model)
        sum_cfg = self.cfg.get(self.cfg.summary_model or self.cfg.default_model)
        acc: list[str] = []

        async def go():
            sum_task = None
            try:
                # 1) 同步落预算：超额时把最老消息摘到 _pending，turns 当场安全，立即作答。
                if self.ctx.compact():
                    # 2) 后台摘要：与下面的流式作答并发；作答结束时再 await，无感知等待。
                    sum_task = asyncio.create_task(self.ctx.summarize_pending(sum_cfg))
                async for t in providers.stream(cfg, self.ctx.build_messages()):
                    acc.append(t)
                    print(t, end="", flush=True)
                if sum_task is not None and await sum_task:
                    print("\n\033[90m(已在后台把较早对话压缩进记忆，/context 查看)\033[0m", end="")
            finally:
                # 收尾：确保后台摘要在事件循环关闭/客户端关闭前结算，避免任务被杀或用到已关客户端。
                if sum_task is not None and not sum_task.done():
                    await asyncio.gather(sum_task, return_exceptions=True)
                await providers.aclose_all()

        print()
        try:
            asyncio.run(go())
        except KeyboardInterrupt:
            print("\n\033[90m(已中断本次生成)\033[0m")
        except Exception as e:
            print(f"调用失败：{e}")
        print("\n")
        self.ctx.add_assistant("".join(acc))

    # -------------------- 指令分发 --------------------
    def handle(self, line: str) -> bool:
        line = line.strip()
        if not line:
            return True

        if not line.startswith("/"):
            self._chat_stream(line)
            return True

        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()

        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/models":
            self._print_models()
        elif cmd == "/search":
            self._do_search(rest)
        elif cmd == "/verbose":
            self.verbose = not self.verbose
            print(f"verbose 模式：{'开' if self.verbose else '关'}")
        elif cmd == "/reset":
            self.ctx.clear()
            print("已清空对话历史（含记忆摘要）。")
        elif cmd == "/context":
            self._print_context()
        elif cmd == "/fusionmodel":
            self._do_fusion(rest, web=False)
        elif cmd == "/fusionweb":
            self._do_fusion(rest, web=True)
        elif cmd == "/auto":
            self._do_auto(rest)
        elif cmd == "/kb":
            self._do_kb(rest)
        else:
            print(f"未知指令 {cmd}，输入 /help 查看帮助。")
        return True

    # -------------------- 搜索后端 --------------------
    def _build_registry(self):
        key = self.cfg.active_search_key()
        if not key:
            raise RuntimeError(
                f"未配置 {self.cfg.search_provider} 的搜索 key：请在 .env 设置 "
                f"{self.cfg.search_provider.upper()}_API_KEY，或用 /search 切到已配置的后端。"
            )
        provider = create_provider(
            self.cfg.search_provider, key, base_url=self.cfg.active_search_base_url()
        )
        return build_web_registry(provider, default_max_results=self.cfg.web_max_results)

    def _do_search(self, rest: str):
        rest = rest.strip().lower()
        configured = set(self.cfg.search_keys)
        if not rest:
            print("\n搜索后端：")
            for name in available_providers():
                flag = "✓ 已配置 key" if name in configured else "✗ 未配置 key"
                active = "  (当前)" if name == self.cfg.search_provider else ""
                print(f"  {name}{active}  —— {flag}")
            print("\n切换：/search <provider>；key 在 .env 配 <PROVIDER>_API_KEY\n")
            return
        if rest not in available_providers():
            print(f"未知后端 {rest}，可用：{', '.join(available_providers())}")
            return
        self.cfg.search_provider = rest
        if rest in configured:
            print(f"已切到搜索后端：{rest}")
        else:
            print(f"已切到 {rest}，但尚未配置 {rest.upper()}_API_KEY；联网前请先在 .env 填好。")

    # -------------------- Fusion（统一流式） --------------------
    def _do_fusion(self, rest: str, *, web: bool):
        rest = rest.strip()
        verb = "/fusionweb" if web else "/fusionmodel"
        if not rest:
            print(f"用法：{verb} [model1,model2,...] <问题>")
            return
        panel_keys, question = self._parse_fusion_args(rest)

        registry = strategy = None
        if web:
            try:
                registry = self._build_registry()
            except Exception as e:
                print(f"联网未启用：{e}")
                return
            strategy = NativeToolCalling()
            label = f"联网[{self.cfg.search_provider}]"
        else:
            label = "离线"

        print(
            f"\n[Fusion·{label}] 面板={', '.join(panel_keys)}  "
            f"裁判={self.cfg.judge_model}  合成={self.cfg.synthesizer_model}\n"
        )
        self._consume_fusion(question, panel_keys, web=web, registry=registry, strategy=strategy)

    # -------------------- 语义路由（/auto） --------------------
    def _do_auto(self, rest: str):
        q = rest.strip()
        if not q:
            print("用法：/auto <问题>")
            return

        async def _decide():
            try:
                return await route(q, self.cfg)
            finally:
                await providers.aclose_all()  # 路由占一次事件循环，收尾关客户端（CLI 约定）

        try:
            label = asyncio.run(_decide())
        except KeyboardInterrupt:
            print("\n\033[90m(已中断)\033[0m")
            return
        print(f"\n🧭 自动路由 → {ROUTE_LABELS.get(label, label)}")
        if label == "web":
            self._run_agent(q, web=True, rag=self.cfg.rag_ready)
        elif label == "fusion":
            self._do_fusion(q, web=False)
        elif label == "local":
            self._run_agent(q, rag=True)
        else:
            self._chat_stream(q)

    # -------------------- 本地知识库（/kb） --------------------
    def _do_kb(self, rest: str):
        q = rest.strip()
        if not q:
            print("用法：/kb <问题>")
            return
        if not self.cfg.rag_ready:
            print("本地知识库未启用：请在 .env 设 RAG_ENABLED=1 与 EMBED_MODEL，"
                  "并先用 `python ingest.py` 建索引。")
            return
        self._run_agent(q, rag=True)

    def _kb_or_none(self):
        """惰性建本地知识库（进程内一次）；未配置/失败返回 None。"""
        if self._kb is None and self.cfg.rag_ready:
            try:
                self._kb = build_kb(self.cfg)
            except Exception as e:
                print(f"本地知识库不可用：{e}")
        return self._kb

    def _run_agent(self, question: str, *, web: bool = False, rag: bool = False):
        """单模型 agent：web→web_search/web_fetch，rag→local_search（可叠加）。单次作答、不带多轮。"""
        reg = None
        bits: list[str] = []
        if web:
            try:
                reg = self._build_registry()
                bits.append("联网")
            except Exception as e:
                print(f"联网未启用：{e}")
        if rag:
            kb = self._kb_or_none()
            if kb is not None:
                reg = reg or ToolRegistry()
                reg.register(build_rag_tool(kb, default_k=self.cfg.rag_top_k))
                bits.append("本地知识库")
        if reg is None:
            print("（无可用工具，转直接作答）\n")
            self._chat_stream(question)
            return
        print(f"\n[Agent·{'+'.join(bits)}] 模型={self.cfg.default_model}\n")
        acc: list[str] = []

        async def go():
            try:
                async for ev in stream_agent(
                    self.cfg.get(self.cfg.default_model), question, reg,
                    NativeToolCalling(), max_steps=self.cfg.web_max_steps,
                    reflexion=self.cfg.reflexion_enabled,
                    reflexion_rounds=self.cfg.reflexion_max_rounds,
                ):
                    if isinstance(ev, TextDelta):
                        acc.append(ev.text)
                        print(ev.text, end="", flush=True)
                    elif isinstance(ev, ToolStarted):
                        verb = "查本地知识库" if ev.name == "local_search" else "检索"
                        print(f"\n\033[90m🔧 [{ev.model}] {ev.name} {verb}…\033[0m")
                    elif isinstance(ev, ReflexionStarted):
                        print(f"\n\033[90m🔍 [{ev.model}] 自检证据、修订中…\033[0m")
                    elif isinstance(ev, AgentDone):
                        if ev.content and ev.content != "".join(acc):
                            print("\n\n\033[90m[已按证据修订]\033[0m\n" + ev.content)
                        if ev.sources:
                            print("\n\n参考来源：")
                            for i, u in enumerate(ev.sources, 1):
                                print(f"  [{i}] {u}")
            finally:
                await providers.aclose_all()

        try:
            asyncio.run(go())
        except KeyboardInterrupt:
            print("\n\033[90m(已中断本次生成)\033[0m")
        print("\n")

    def _consume_fusion(self, question, panel_keys, *, web, registry, strategy):
        degraded: dict[str, str] = {}
        totals = {"search": 0, "fetch": 0}
        t0 = time.perf_counter()

        def render(ev):
            if isinstance(ev, PanelStarted):
                print("⏳ [面板] 正在并行检索与深度抓取..." if web else "⏳ [面板] 正在并行作答...")
            elif isinstance(ev, AgentDegraded):
                degraded[ev.model] = ev.reason
            elif isinstance(ev, ReflexionStarted):
                print(f"  🔍 [{ev.model}] 自检证据、修订中…")
            elif isinstance(ev, AgentDone):
                totals["search"] += ev.n_search
                totals["fetch"] += ev.n_fetch
                self._render_agent_done(ev, web, degraded)
            elif isinstance(ev, PanelFinished):
                if self.verbose:
                    self._print_panels_verbose(ev.results)
            elif isinstance(ev, JudgeFinished):
                print("⚖️ [裁判] 交叉验证与裁决完成")
                if self.verbose:
                    self._print_judge_verbose(ev.analysis)
            elif isinstance(ev, SynthesisStarted):
                print("\n" + "=" * 50)
                print("🤖 Fusion 深度研报 (开始流式输出)")
                print("=" * 50 + "\n")
            elif isinstance(ev, SynthesisDelta):
                print(ev.text, end="", flush=True)
            elif isinstance(ev, FusionDone):
                if ev.sources:
                    print("\n\n参考来源：")
                    for i, u in enumerate(ev.sources, 1):
                        print(f"  [{i}] {u}")
                elapsed = time.perf_counter() - t0
                if web:
                    print(
                        f"\n\033[90m⏱️ 耗时: {elapsed:.1f}s | "
                        f"🔍 搜索: {totals['search']}次 | 📄 抓取: {totals['fetch']}次\033[0m"
                    )
                else:
                    print(f"\n\033[90m⏱️ 耗时: {elapsed:.1f}s\033[0m")
            elif isinstance(ev, StageError):
                print(f"\n⚠ [{ev.stage}] 出错：{ev.detail}")

        async def consume():
            try:
                async for ev in stream_fusion(
                    question, panel_keys, self.cfg, web=web, registry=registry, strategy=strategy
                ):
                    render(ev)
            finally:
                await providers.aclose_all()

        try:
            asyncio.run(consume())
        except KeyboardInterrupt:
            print("\n\033[90m(已中断本次 Fusion)\033[0m")
        except Exception as e:
            print(f"\nFusion 失败：{e}")
        print()

    def _render_agent_done(self, ev, web: bool, degraded: dict):
        if ev.degraded or ev.model in degraded:
            reason = degraded.get(ev.model, "未知")
            if web:
                print(
                    f"  ⚠ [{ev.model}] 降级完成 "
                    f"({ev.n_search}次搜索, {ev.n_fetch}次抓取 | 原因: {reason})"
                )
            else:
                print(f"  ⚠ [{ev.model}] 作答失败 (原因: {reason})")
        elif web:
            print(
                f"  ✓ [{ev.model}] 检索完成 "
                f"({ev.n_search}次搜索, {ev.n_fetch}次抓取, 获取 {len(ev.sources)} 个有效信源)"
            )
        else:
            print(f"  ✓ [{ev.model}] 作答完成")

    def _parse_fusion_args(self, rest: str) -> tuple[list[str], str]:
        first, _, maybe_q = rest.partition(" ")
        tokens = [t for t in first.split(",") if t]
        if tokens and all(t in self.cfg.models for t in tokens) and maybe_q.strip():
            return tokens, maybe_q.strip()
        return self.cfg.default_panel, rest.strip()

    # -------------------- 展示 --------------------
    def _print_models(self):
        print("\n已配置模型：")
        for key, m in self.cfg.models.items():
            tools = "" if m.supports_tools else "  [无原生 tools]"
            print(f"  {key:8s} -> {m.name}  ({m.base_url}){tools}")
        print("\n角色分工：")
        print(f"  默认对话  DEFAULT_MODEL     = {self.cfg.default_model}")
        print(f"  裁判      JUDGE_MODEL       = {self.cfg.judge_model}")
        print(f"  合成      SYNTHESIZER_MODEL = {self.cfg.synthesizer_model}")
        print(f"  摘要      SUMMARY_MODEL     = {self.cfg.summary_model or self.cfg.default_model}")
        print(f"  默认面板  DEFAULT_PANEL     = {', '.join(self.cfg.default_panel)}")
        active_key = "已配置" if self.cfg.active_search_key() else "未配置 key"
        print(f"  搜索后端  SEARCH_PROVIDER   = {self.cfg.search_provider} ({active_key})\n")

    def _print_context(self):
        c = self.ctx
        print(
            f"\n对话上下文：最近 {len(c.turns)} 条消息（约 {c.char_count()} 字），"
            f"预算 {c.char_budget} 字 / 始终保留最近 {c.keep_recent} 条"
        )
        if c.summary:
            print(f"\n记忆摘要：\n{c.summary}\n")
        else:
            print("（暂无记忆摘要：对话尚未超出预算）\n")

    def _print_panels_verbose(self, results):
        print("\n—— 面板原始回答 ——")
        for p in results:
            tag = f"{p.model_key} · {p.model_name}  ({p.latency_ms}ms, {p.tool_calls} 次工具)"
            if p.ok:
                print(f"\n[{tag}]")
                if p.steps:
                    print("  工具轨迹：" + "; ".join(p.steps))
                print(p.content)
            else:
                print(f"\n[{tag}]  ✗ 失败：{p.error}")

    def _print_judge_verbose(self, a):
        print("\n—— 裁判裁决 ——")
        if a is None:
            return
        if not a.parsed and a.raw:
            print(a.raw)
            return
        print(f"  结论：{a.verdict}")
        if a.confirmed_facts:
            print("  已确认事实：")
            for f in a.confirmed_facts:
                print(f"      - {f.fact} {''.join(f.citations)}")
        if a.debunked_rumors:
            print("  存疑传闻：")
            for r in a.debunked_rumors:
                print(f"      - {r}")
        if a.blind_spots:
            print("  盲点：")
            for b in a.blind_spots:
                print(f"      - {b}")
        if a.source_mapping:
            print("  信源：")
            for k in sorted(a.source_mapping):
                print(f"      {k} {a.source_mapping[k]}")


def run_repl(cfg: AppConfig):
    print("Fusion Agent 已启动。输入 /help 查看指令，/exit 退出。")
    print("（生成过程中按 Ctrl-C 可中断本次；退出请用 /exit 或 Ctrl-D）\n")
    session = ChatSession(cfg)
    while True:
        try:
            line = input("你 > ")
        except EOFError:            # Ctrl-D：退出
            print("\n再见。")
            break
        except KeyboardInterrupt:   # 空闲时 Ctrl-C：只给新提示符，不退出整个程序
            print()
            continue
        try:
            if not session.handle(line):
                print("再见。")
                break
        except KeyboardInterrupt:   # 兜底：处理途中的 Ctrl-C 同样不退出，回到提示符
            print("\n\033[90m(已中断)\033[0m")
