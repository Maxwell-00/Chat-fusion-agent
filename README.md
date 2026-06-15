# Fusion Agent

> 多模型并行作答 → 裁判交叉验证 → 合成输出的命令行 Agent。
> 复刻 OpenRouter Fusion 的核心效果：多个不同厂商的模型同时回答同一个问题，由裁判模型做结构化对比与信源提纯，再由合成模型产出带引用的最终答案。

所有模型统一走 **OpenAI 兼容接口**，新增厂商只需改 `.env`，无需改代码。支持**离线**与**联网检索**两种模式，全链路**流式输出**。

> 当前版本：**v0.1**，代码位于 [`Version/v0.1/`](Version/v0.1)。

---

## 特性

- **三角色 Fusion**：面板（1～N 个模型并行作答）、裁判（交叉验证 + 信源提纯，不直接答题）、合成（产出最终回答）。角色可由任意已配置模型担任。
- **两种模式**
  - `/fusionmodel` 离线：面板模型并行直接作答。
  - `/fusionweb` 联网：每个面板模型作为 **agent**，用原生 function-calling 自主调用 `web_search` / `web_fetch` 检索后再作答。
- **流式输出**：联网阶段实时上报进度（谁在搜、命中几条、是否降级），合成阶段逐字输出，缓解高首字延迟。
- **证据链闭环 + 内联引用**：面板上交 `(url, snippet)` 证据，裁判去重、剔除低质社媒源、按权威重排并从 `[1]` 连续编号，合成器只能引用裁判给出的编号，正文带 `[1][2]` 角标，杜绝引用幻觉。
- **可插拔搜索后端（策略模式）**：内置 Tavily、博查 Bocha，新增后端只需写一个实现类并注册。
- **健壮容错**：面板单点失败不影响整体；裁判非法 JSON 重试 + 兜底；agent 工具循环有步数上限防止无限循环；流式分片缓冲 + 错误降级，不崩流。

---

## 架构

```
普通对话：  用户 ──► 默认模型（流式）──► 回答（维护多轮历史）

Fusion：
  用户问题
     │
     ├─► 并行面板 ──► [model1] ┐
     │              [model2] ├─► 各自作答（离线直接答 / 联网 agent 检索后答）
     │              [model3] ┘
     │
     ├─► 裁判 ──► 交叉验证 + 信源提纯/编号 → 结构化 JSON
     │            （verdict / source_mapping / confirmed_facts / debunked_rumors / blind_spots）
     │
     └─► 合成 ──► 基于裁决产出带内联角标 [n] 的最终回答（流式）
```

时延 ≈ `max(面板各模型) + 裁判 + 合成`（面板并行）；成本 ≈ 面板数 + 2。

### 仓库结构

```
Chat fusion agent/
├── LICENSE                  MIT
├── README.md                本文件
└── Version/
    └── v0.1/                ← 当前版本（代码在这里）
        ├── main.py          入口：加载 .env -> 构建配置 -> 启动 CLI
        ├── config.py        扫描 MODEL{n}_*、角色分工、搜索配置（含校验）
        ├── providers.py     统一调用层：call_model / call_with_tools / stream / stream_with_tools
        ├── chat.py          CLI 主循环、指令解析、流式渲染
        ├── fusion.py        编排：panel -> judge -> synthesize（run_fusion / stream_fusion）
        ├── panel.py         面板：并行分发（离线 / 联网）
        ├── judge.py         裁判：交叉验证 + 信源提纯，输出结构化 JSON
        ├── synthesizer.py   合成：消费裁决，产出带内联角标的回答
        ├── agent.py         面板 agent：工具循环（run_agent / stream_agent）
        ├── tool_calling.py  工具调用策略：Native（原生）+ Hermes（预留）
        ├── tools.py         工具注册中心：tools schema + 分发 + web_fetch 黑名单
        ├── search_providers.py 搜索后端（策略模式）：Tavily / Bocha + 工厂
        ├── stream_events.py 流式事件模型
        ├── prompts/         裁判 / 合成 / 面板 agent 的 prompt 模板
        ├── requirements.txt
        └── .env.example
```

---

## 快速开始

需要 Python 3.10+。

```bash
git clone <your-repo-url>
cd "Chat fusion agent/Version/v0.1"
pip install -r requirements.txt
cp .env.example .env        # 然后填入真实 API key
python main.py
```

启动后进入交互式命令行：

```
你 > 你好                                       # 普通多轮对话（流式）
你 > /fusionmodel 比较 Rust 和 Go 的并发模型        # 离线 Fusion
你 > /fusionweb 上个月 AI 领域有什么大新闻           # 联网 Fusion（面板 agent 检索）
你 > /search bocha                              # 切换搜索后端
你 > /verbose                                   # 显示面板原文 + 裁判裁决
你 > /models     /help     /exit
```

---

## 指令

| 指令 | 说明 |
|------|------|
| `/fusionmodel [面板] <问题>` | 离线 Fusion（并行作答 → 裁判 → 合成，流式） |
| `/fusionweb [面板] <问题>` | 联网 Fusion（面板 agent 自主检索 + 流式合成） |
| `/search [provider]` | 查看 / 切换搜索后端（tavily、bocha…） |
| `/models` | 列出已配置的模型与角色分工 |
| `/verbose` | 开关详细模式（展示面板原文 + 裁判裁决） |
| `/reset` | 清空普通对话历史 |
| `/help` `/exit` | 帮助 / 退出 |

> 面板可选：`/fusionweb model1,model2 <问题>` 指定本次参与的模型；不指定则用 `DEFAULT_PANEL`。

---

## 配置（.env）

程序从 `MODEL1` 起递增扫描 `MODEL{n}_NAME`，直到某个序号缺失为止。完整示例见 [`Version/v0.1/.env.example`](Version/v0.1/.env.example)。

| 变量 | 说明 |
|------|------|
| `MODEL{n}_NAME / _BASE_URL / _API_KEY` | 模型 id、OpenAI 兼容端点、key |
| `MODEL{n}_TEMPERATURE / _TIMEOUT` | 可选 |
| `MODEL{n}_SUPPORTS_TOOLS` | 该模型是否支持原生 tools（默认 true；false 时联网会直接作答） |
| `DEFAULT_MODEL / JUDGE_MODEL / SYNTHESIZER_MODEL` | 三种角色各用哪个模型 |
| `DEFAULT_PANEL` | `/fusionmodel`、`/fusionweb` 默认面板 |
| `SEARCH_PROVIDER` | 默认搜索后端：`tavily` / `bocha` |
| `TAVILY_API_KEY` / `BOCHA_API_KEY` | 各搜索后端的 key（填了就能用，运行时 `/search` 可切） |
| `WEB_MAX_STEPS` / `WEB_MAX_RESULTS` | 单个 agent 工具调用轮数上限 / 每次搜索结果数 |

---

## 工作原理要点

- **离线 vs 联网**：离线面板各模型并行直接作答；联网面板每个模型以 agent 方式在工具循环里 `web_search` / `web_fetch`，受步数上限约束。两种模式都只把**状态**上报到 CLI，只有合成阶段逐字流式（避免多模型文本交错）。
- **裁判主导引用**：裁判负责信源提纯——去重、剔除视频/社交等低质来源、按「官方/权威媒体 > 自媒体 > 论坛」取舍，从 `[1]` 连续重编号，并为每条 `confirmed_facts` 标注 `citations`；同时识别 `debunked_rumors`（仅单一自媒体出现的奇特名词/机翻梗）。合成器**只能**使用裁判给的编号，参考来源列表由程序按 `source_mapping` 确定性渲染，与正文角标一致。
- **抓取治理**：`web_fetch` 对 youtube / instagram / x / reddit 等做黑名单拦截（不发请求），面板被要求每次搜索最多抓 2 个最相关 URL，控制成本与延迟。
- **流式 + Function Calling**：分片缓冲按 `index` 拼接跨 chunk 的工具参数，处理「id 仅首片、name 晚到、finish_reason 误报 stop、单 delta 同时含文本与工具调用」等真实兼容性问题；不变量是每个 `tool_call_id` 必有一条 tool 响应。

---

## 扩展

- **加模型 / 换厂商**：`.env` 加一组 `MODEL{n}_*`。不兼容 OpenAI 协议的厂商只改 `providers.py` 一处。
- **加搜索后端**：在 `search_providers.py` 写 `class XxxProvider(SearchProvider)` 并 `register_provider("xxx", XxxProvider)`，再在 `.env` 配 `XXX_API_KEY`。
- **兼容不支持原生 tools 的模型**：实现 `tool_calling.py` 里预留的 `HermesToolCalling`（`<tool_call>` 文本协议解析），agent 循环无需改动。
- **调效果**：直接改 `prompts/` 下的裁判 / 合成 / 面板 agent 模板。

---

## 安全

- 所有密钥只放 `.env`；仓库通过 `.gitignore` 排除 `.env`，只提交 `.env.example`（占位符）。
- 不在日志中打印完整 api_key。
- 联网 Fusion 一次会产生「面板数 + 2」次模型调用并可能多轮检索，注意成本；可用 `WEB_MAX_STEPS` / `WEB_MAX_RESULTS` 约束。

---

## 路线图

流式文本工具调用、YAML 预设面板、加权 / 投票、结果缓存、对外暴露 OpenAI 兼容服务。

---

## 许可证

[MIT](LICENSE) © 2026 Maxwell-00
