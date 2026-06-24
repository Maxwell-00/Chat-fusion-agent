"""真实 API 冒烟测试（需要 .env 里填好真实 key，会产生真实调用与少量费用）。

与 tests/ 下的 mock 自测不同：这个脚本真的去打各家模型和搜索后端，用来在"封装成应用"
之前验证集成层——单元测试全是 mock，覆盖不到厂商差异（裁判 JSON 模式是否支持、流式
tool-call 的分片形状、各搜索后端返回结构等）。

逐项独立检查，单项失败只记录不中断，末尾给汇总：
  [1] 每个模型基础调用            [2] 裁判 JSON 模式(response_format)
  [3] 每个已配置搜索后端(search+fetch)  [4] 普通流式
  [5] 流式工具调用(stream_agent，验证分片缓冲对真实分片的兼容)
  [6] 离线 Fusion 全链路(面板→裁判→合成)
  [7] 联网 Fusion 全链路(仅 --web 时跑，更花钱/慢)

用法：
    cd fusion_agent
    cp .env.example .env       # 填好真实 key
    python smoke_test.py
    python smoke_test.py --web # 额外跑一次联网 Fusion

提示：默认大约产生 (2×模型数 + 8) 次模型调用 + 数次搜索；嫌多可注释掉对应小节。
"""
from __future__ import annotations

import asyncio
import json
import sys
import time

from dotenv import load_dotenv

OK = "\033[32m✓ PASS\033[0m"
BAD = "\033[31m✗ FAIL\033[0m"
WARN = "\033[33m‼ WARN\033[0m"


async def run(run_web: bool) -> int:
    load_dotenv()  # 须在 load_config 之前
    from config import load_config

    try:
        cfg = load_config()
    except Exception as e:
        print(f"{BAD} 配置加载失败：{e}")
        print("提示：先 `cp .env.example .env` 并填入真实 key。")
        return 1

    import providers
    from agent import stream_agent
    from fusion import stream_fusion
    from search_providers import create_provider
    from stream_events import AgentDegraded, AgentDone, FusionDone, StageError, SynthesisDelta
    from tool_calling import NativeToolCalling
    from tools import build_web_registry

    results: list[tuple[str, bool]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok))
        print(f"  {OK if ok else BAD} {name}" + (f" — {detail}" if detail else ""))

    print(
        f"\n配置：模型 {list(cfg.models)}；默认={cfg.default_model} 裁判={cfg.judge_model} "
        f"合成={cfg.synthesizer_model} 摘要={cfg.summary_model or cfg.default_model}"
    )
    print(
        f"      面板={cfg.default_panel}；已配置搜索后端={list(cfg.search_keys) or '(无)'}"
        f"（当前 {cfg.search_provider}）"
    )

    try:
        # [1] 每个模型基础调用
        print("\n[1] 各模型基础调用")
        for key, mc in cfg.models.items():
            t0 = time.perf_counter()
            try:
                txt = await providers.call_model(mc, [{"role": "user", "content": "只回复两个字：你好"}])
                dt = int((time.perf_counter() - t0) * 1000)
                record(f"{key} ({mc.name})", bool(txt.strip()), f"{dt}ms，返回 {len(txt)} 字")
            except Exception as e:
                record(f"{key} ({mc.name})", False, f"{type(e).__name__}: {e}")

        # [2] 裁判 JSON 模式
        print("\n[2] 裁判 JSON 模式（response_format=json_object）")
        jc = cfg.get(cfg.judge_model)
        try:
            txt = await providers.call_model(
                jc,
                [{"role": "user", "content": '只输出这个 JSON、不要任何其他文字：{"ok": true}'}],
                response_format={"type": "json_object"},
            )
            try:
                json.loads(txt)
                record(f"{cfg.judge_model} 原生 JSON 模式", True, "返回合法 JSON")
            except Exception:
                record(f"{cfg.judge_model} 原生 JSON 模式", False, f"返回非 JSON：{txt[:60]!r}（裁判会走文本抠取兜底）")
        except Exception as e:
            print(
                f"  {WARN} {cfg.judge_model} 不接受 response_format（{type(e).__name__}）"
                f" → 裁判会自动回退到无 JSON 模式 + 文本抠取（已有兜底，不致命）"
            )

        # [3] 搜索后端
        print("\n[3] 搜索后端（search + fetch）")
        if not cfg.search_keys:
            print(f"  {WARN} 未配置任何搜索 key，跳过（联网 Fusion 将不可用）")
        for name, k in cfg.search_keys.items():
            try:
                prov = create_provider(name, k, base_url=cfg.search_base_urls.get(name))
                res = await prov.search("OpenAI 最新模型", 3)
                record(f"{name} search", bool(res), f"返回 {len(res)} 条")
                if res and res[0].url:
                    try:
                        body = await prov.fetch(res[0].url)
                        record(f"{name} fetch", bool(body.strip()), f"{len(body)} 字")
                    except Exception as e:
                        print(f"  {WARN} {name} fetch 失败（不致命，可只靠摘要）：{type(e).__name__}: {e}")
            except Exception as e:
                record(f"{name} search", False, f"{type(e).__name__}: {e}")

        # [4] 普通流式
        print("\n[4] 普通流式")
        dc = cfg.get(cfg.default_model)
        try:
            acc: list[str] = []
            async for t in providers.stream(dc, [{"role": "user", "content": "用一句话介绍你自己"}]):
                acc.append(t)
            record("默认模型流式", bool(acc), f"{len(acc)} 个分片，{sum(len(x) for x in acc)} 字")
        except Exception as e:
            record("默认模型流式", False, f"{type(e).__name__}: {e}")

        # [5] 流式工具调用（验证分片缓冲对真实分片形状的兼容）
        print("\n[5] 流式工具调用（stream_agent）")
        if cfg.active_search_key():
            try:
                prov = create_provider(
                    cfg.search_provider, cfg.active_search_key(), base_url=cfg.active_search_base_url()
                )
                reg = build_web_registry(prov, default_max_results=cfg.web_max_results)
                done = None
                degraded = False
                async for ev in stream_agent(
                    dc, "用一句话回答：北京今天天气如何？需要的话联网查。", reg,
                    NativeToolCalling(), max_steps=2,
                ):
                    if isinstance(ev, AgentDegraded):
                        degraded = True
                    elif isinstance(ev, AgentDone):
                        done = ev
                if done is not None:
                    record(
                        "stream_agent 跑通", not degraded,
                        f"工具调用 {done.tool_calls} 次，{'降级' if degraded else '正常'}，最终 {len(done.content)} 字",
                    )
                else:
                    record("stream_agent 跑通", False, "未收到 AgentDone")
            except Exception as e:
                record("stream_agent 跑通", False, f"{type(e).__name__}: {e}")
        else:
            print(f"  {WARN} 当前搜索后端 {cfg.search_provider} 未配置 key，跳过")

        # [6] 离线 Fusion 全链路
        print("\n[6] 离线 Fusion 全链路（面板→裁判→合成）")
        try:
            synth: list[str] = []
            fusion_done = None
            stage_errs: list[tuple[str, str]] = []
            async for ev in stream_fusion("用一句话说：Python 列表和元组的区别？", cfg.default_panel, cfg, web=False):
                if isinstance(ev, SynthesisDelta):
                    synth.append(ev.text)
                elif isinstance(ev, FusionDone):
                    fusion_done = ev
                elif isinstance(ev, StageError):
                    stage_errs.append((ev.stage, ev.detail))
            detail = f"合成 {sum(len(x) for x in synth)} 字"
            if stage_errs:
                detail += f"；阶段降级 {stage_errs}"
            record("离线 Fusion", bool("".join(synth).strip()) and fusion_done is not None, detail)
        except Exception as e:
            record("离线 Fusion", False, f"{type(e).__name__}: {e}")

        # [7] 联网 Fusion 全链路（可选）
        if run_web:
            print("\n[7] 联网 Fusion 全链路（--web）")
            if cfg.active_search_key():
                try:
                    prov = create_provider(
                        cfg.search_provider, cfg.active_search_key(), base_url=cfg.active_search_base_url()
                    )
                    reg = build_web_registry(prov, default_max_results=cfg.web_max_results)
                    synth = []
                    fusion_done = None
                    async for ev in stream_fusion(
                        "用一两句话说：最近 AI 领域有什么大新闻？", cfg.default_panel, cfg,
                        web=True, registry=reg, strategy=NativeToolCalling(),
                    ):
                        if isinstance(ev, SynthesisDelta):
                            synth.append(ev.text)
                        elif isinstance(ev, FusionDone):
                            fusion_done = ev
                    record(
                        "联网 Fusion",
                        bool("".join(synth).strip()) and fusion_done is not None,
                        f"合成 {sum(len(x) for x in synth)} 字，来源 {len(fusion_done.sources) if fusion_done else 0} 条",
                    )
                except Exception as e:
                    record("联网 Fusion", False, f"{type(e).__name__}: {e}")
            else:
                print(f"  {WARN} 当前搜索后端未配置 key，跳过联网 Fusion")
    finally:
        await providers.aclose_all()  # 全程一个事件循环，统一收尾关闭客户端

    passed = [r for r in results if r[1]]
    fails = [r for r in results if not r[1]]
    print("\n" + "=" * 50)
    print(f"汇总：{len(passed)} 通过，{len(fails)} 失败")
    if fails:
        print("失败项：")
        for name, _ in fails:
            print(f"  - {name}")
    print("（WARN 项不计入失败：多为厂商不支持某可选能力，代码已有兜底。）")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run("--web" in sys.argv)))
