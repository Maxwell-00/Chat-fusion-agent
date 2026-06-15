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
from config import AppConfig
from fusion import stream_fusion
from search_providers import available_providers, create_provider
from stream_events import (
    AgentDegraded,
    AgentDone,
    FusionDone,
    JudgeFinished,
    PanelFinished,
    PanelStarted,
    StageError,
    SynthesisDelta,
    SynthesisStarted,
)
from tool_calling import NativeToolCalling
from tools import build_web_registry

HELP = """\
可用指令：
  /fusionmodel <问题>                  离线 Fusion（并行作答 -> 裁判 -> 合成，流式）
  /fusionmodel model1,model2 <问题>    指定面板的离线 Fusion
  /fusionweb <问题>                    联网 Fusion（面板 agent 自主检索 + 流式合成）
  /fusionweb model1,model2 <问题>      指定面板的联网 Fusion
  /search [provider]                   查看 / 切换搜索后端（tavily、bocha…）
  /models                              列出已配置的模型与角色分工
  /verbose                             开关详细模式（展示面板原文 + 裁判裁决）
  /reset                               清空普通对话历史
  /help                                显示本帮助
  /exit                                退出
直接输入文字（不以 / 开头）即为普通多轮对话（流式），使用默认模型。
"""


class ChatSession:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.history: list[dict] = []
        self.verbose = False

    # -------------------- 普通对话（流式） --------------------
    def _chat_stream(self, text: str):
        self.history.append({"role": "user", "content": text})
        cfg = self.cfg.get(self.cfg.default_model)
        acc: list[str] = []

        async def go():
            async for t in providers.stream(cfg, self.history):
                acc.append(t)
                print(t, end="", flush=True)

        print()
        try:
            asyncio.run(go())
        except Exception as e:
            print(f"调用失败：{e}")
        print("\n")
        self.history.append({"role": "assistant", "content": "".join(acc)})

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
            self.history.clear()
            print("已清空对话历史。")
        elif cmd == "/fusionmodel":
            self._do_fusion(rest, web=False)
        elif cmd == "/fusionweb":
            self._do_fusion(rest, web=True)
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

    def _consume_fusion(self, question, panel_keys, *, web, registry, strategy):
        degraded: dict[str, str] = {}
        totals = {"search": 0, "fetch": 0}
        t0 = time.perf_counter()

        def render(ev):
            if isinstance(ev, PanelStarted):
                print("⏳ [面板] 正在并行检索与深度抓取..." if web else "⏳ [面板] 正在并行作答...")
            elif isinstance(ev, AgentDegraded):
                degraded[ev.model] = ev.reason
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
            async for ev in stream_fusion(
                question, panel_keys, self.cfg, web=web, registry=registry, strategy=strategy
            ):
                render(ev)

        try:
            asyncio.run(consume())
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
        print(f"  默认面板  DEFAULT_PANEL     = {', '.join(self.cfg.default_panel)}")
        active_key = "已配置" if self.cfg.active_search_key() else "未配置 key"
        print(f"  搜索后端  SEARCH_PROVIDER   = {self.cfg.search_provider} ({active_key})\n")

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
    print("Fusion Agent 已启动。输入 /help 查看指令，/exit 退出。\n")
    session = ChatSession(cfg)
    while True:
        try:
            line = input("你 > ")
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not session.handle(line):
            print("再见。")
            break
