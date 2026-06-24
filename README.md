# Fusion Agent

> 多模型并行作答 → 裁判交叉验证 → 合成输出的 Agent，**命令行（CLI）与网页（Web UI）双前端**。
> 复刻 OpenRouter Fusion 的核心效果：多个不同厂商的模型同时回答同一个问题，由裁判模型做结构化对比与信源提纯，再由合成模型产出带引用的最终答案。

所有模型统一走 **OpenAI 兼容接口**，新增厂商只需改 `.env` 或在网页「设置」里点几下，无需改代码。支持**离线**与**联网检索**两种模式、全链路**流式输出**、长对话自动**滚动压缩上下文**、多会话**持久化**（重启不丢）。v0.4 起新增：**语义路由**（自动判定该直答 / 联网 / 多模型 / 查本地）、**反思 Reflexion**（联网作答后对照证据自检修订）、**本地 RAG**（把笔记 / PDF 建成知识库，agent 可"向内查"）、**可视化设置中心**（在网页里改模型与参数、增删模型，不碰 `.env`），以及可选的**全链路可观测性**（OpenTelemetry，开追踪即在进程内自动起本地 Phoenix）。

> 当前版本：**v0.4**，代码位于 [`Version/v0.4/`](Version/v0.4)。更新内容见下方[版本历史](#版本历史)。

---

## 特性

- **三角色 Fusion**：面板（1～N 个模型并行作答）、裁判（交叉验证 + 信源提纯，不直接答题）、合成（产出最终回答）。角色可由任意已配置模型担任。
- **多种作答模式**
  - 普通直答：默认模型多轮对话。
  - 联网单模型：默认模型作为 **agent** 自主 `web_search` / `web_fetch` 检索后作答。
  - Fusion：多模型并行 → 裁判 → 合成；离线或联网。
  - 本地知识库：agent 用 `local_search` 检索你的本地文档作答（需开 RAG）。
- **语义路由（自动选模式）**：开启「自动选择」/ `/auto` 后，先启发式预判、再用便宜模型分类，自动决定走 直答 / 联网 / Fusion / 本地，并**显式回显**所选路由。
- **反思 Reflexion**：联网 agent 出初稿后，对照检索到的证据自检、剔除/修正没有支撑的主张，再产出修订版（opt-in，默认关）。
- **本地 RAG（"向内看"）**：把 `.md / .txt / .pdf` 切块、嵌入、入向量库（ChromaDB）；agent 通过 `local_search` 工具召回片段，召回内容同样进裁判证据链、可被合成器编号引用。Web 有「📚 知识库」可上传 / 列表 / 删除，CLI 有 `/kb`，命令行 `ingest.py` 可批量建索引。
- **双前端，同一内核**：核心是事件驱动的流式编排（`stream_fusion` / `stream_agent`），CLI（`main.py`）与 Web UI（`app.py`）只是两个渲染器——加前端不用动内核。
  - **CLI**：交互式命令行，流式渲染，Ctrl-C 只中断当前任务。
  - **Web UI（Gradio）**：顶部 Tabs「💬 对话 / ⚙️ 设置 / 📚 知识库」。对话页 = 多会话侧栏（一行一条）+ 单聊天窗 + ➕ 展开 自动 / Fusion / 联网 / 本地知识库 四个勾（自动与手动互斥）+ **发送 / 停止（可中断生成）** + Fusion「展开过程」右侧详情栏。
- **可视化设置中心**：在网页里调角色、系统参数、各模型 Name/Base URL/API Key（密码框掩码）、**增删模型**、勾选面板模型、上下文预算挡位、开关与搜索/嵌入 key。保存写入 `user_settings.json` **覆盖层**（启动时叠加在 `.env` 上、**不改 `.env`**）并**热生效**。
- **多会话持久化（SQLite）**：对话与历史落本地库，重启不丢；普通对话窗口可从摘要 + 尾部消息**无损重建**。
- **全链路流式输出**：联网阶段实时上报进度，合成阶段逐字输出；普通对话把模型**推理流**单独折叠展示，缓解高首字延迟的体感。
- **长对话上下文管理（滚动摘要 + 最近 N 轮）**：历史超过字符预算时同步把最老消息移出对话（即时作答），后台用便宜模型把它们压成"记忆"，压缩与作答并发、零感知等待。
- **裁判输入有上限 + 证据链闭环**：面板回答头+尾截断、信源条数封顶；裁判去重/剔除低质源/按权威重排并从 `[1]` 连续编号，合成器只能引用裁判给的编号，正文带 `[1][2]` 角标，杜绝引用幻觉。
- **可插拔后端（策略模式）**：搜索（Tavily / 博查 Bocha）、向量库（ChromaDB）都是"接口 + 注册表 + 工厂"，新增一个只写一个实现类并注册。
- **可选可观测性（OpenTelemetry）**：一个开关即追踪所有模型调用（token / 延迟 / prompt），并把 `fusion → panel → judge / synthesize`、`web_search / web_fetch / local_search / reflexion` 标成结构化嵌套 span，得到完整 trace 树；**开追踪即进程内自动起本地 Phoenix**，Web 侧栏一键打开监控面板。默认关闭、零侵入、零开销。
- **健壮容错**：面板单点失败不影响整体；裁判非法 JSON 重试 + 兜底；agent 工具循环有步数上限；流式分片缓冲 + 错误降级，不崩流；生成中 **Ctrl-C / 「停止」只中断本次，不退出程序**。

---

## 架构

```
普通对话：  用户 ──► 默认模型（流式，推理流折叠 + 正文）──► 回答（多轮历史 + 超预算滚动摘要压缩）

语义路由（自动）：用户问题 ──► router ──► direct / web / fusion / local 之一（启发式 + 便宜模型分类）

Fusion：
  用户问题
     │
     ├─► 并行面板 ──► [model1] ┐
     │              [model2] ├─► 各自作答（离线直接答 / 联网 agent：web_search·web_fetch·local_search，可反思修订）
     │              [model3] ┘
     │
     ├─► 裁判 ──► 交叉验证 + 信源提纯/编号 → 结构化 JSON
     │            （verdict / source_mapping / confirmed_facts / debunked_rumors / blind_spots）
     │
     └─► 合成 ──► 基于裁决产出带内联角标 [n] 的最终回答（流式）

横切：  持久化（SQLite，多会话）  ·  设置覆盖层（user_settings.json）  ·  本地 RAG（向量库 + 嵌入）  ·  可观测性（OpenTelemetry → 本地 Phoenix，可选）
前端：  CLI（main.py）           ·  Web UI（app.py，Gradio：对话/设置/知识库）   ← 同一事件流的两个渲染器
```

时延 ≈ `max(面板各模型) + 裁判 + 合成`（面板并行）；成本 ≈ 面板数 + 2（联网 / 反思各自再加调用）。

### 仓库结构

```
Chat fusion agent/
├── LICENSE                  MIT
├── README.md                本文件
└── Version/
    ├── v0.1/  v0.2/  v0.3/  历史版本（见版本历史）
    └── v0.4/                ← 当前版本（代码在这里）
        ├── main.py          CLI 入口：加载 .env -> init_tracing -> 构建配置 -> 启动 REPL
        ├── app.py           Web UI（Gradio）：Tabs 对话/设置/知识库；发送停止、自动路由、设置中心
        ├── chat.py          CLI 主循环、指令解析（含 /auto、/kb）、流式渲染、Ctrl-C 中断
        ├── config.py        扫描 MODEL{n}_*、角色/上下文/裁判/搜索/RAG 配置；user_settings.json 覆盖层（增删改模型）
        ├── providers.py     唯一网络出口：call_model / stream / stream_rich / stream_with_tools / embed；客户端复用
        ├── chat_context.py  普通对话上下文：滚动摘要 + 最近 N 轮（压缩不阻塞作答）
        ├── store.py         SQLite 持久化：conversations / messages + 每会话摘要，窗口可无损重建
        ├── fusion.py        编排（流式）：panel -> judge -> synthesize（stream_fusion）
        ├── agent.py         面板 / 单模型 agent：流式工具循环 stream_agent（可选反思修订）
        ├── router.py        语义路由：启发式 + LLM 分类器 → direct / web / fusion / local
        ├── judge.py         裁判：交叉验证 + 信源提纯，输出结构化 JSON（含输入上限）
        ├── synthesizer.py   合成：消费裁决 + 各面板原文，流式产出带内联角标的研报
        ├── tool_calling.py  工具调用策略：NativeToolCalling（原生 function-calling，流式分片缓冲）
        ├── tools.py         工具注册中心：web_search / web_fetch（域名黑名单）
        ├── search_providers.py 搜索后端（策略模式）：Tavily / Bocha + 工厂
        ├── rag.py           本地 RAG：分块 / ingest / search / local_search 工具 / 知识库增删查
        ├── vector_store.py  向量库后端（策略模式）：ChromaVectorStore（cosine 持久化）+ 工厂
        ├── ingest.py        建索引 CLI：扫描文档目录 → 切块嵌入入库（--rebuild / --stats）
        ├── tracing.py       可观测性：FUSION_TRACING 开关；结构化 span；进程内自动起 Phoenix
        ├── clean_traces.py  运维：一键清空本地 Phoenix 历史 trace（不碰应用代码）
        ├── stream_events.py 流式事件模型
        ├── prompts/         judge / synthesizer / panel_agent / summary / reflexion / router 模板
        ├── requirements.txt
        └── .env.example
```

---

## 快速开始

需要 Python 3.10+。

```bash
git clone <your-repo-url>
cd "Chat fusion agent/Version/v0.4"
pip install -r requirements.txt        # 含 gradio + chromadb/pypdf（RAG）；追踪依赖在注释里
cp .env.example .env                    # 然后填入真实 API key

python main.py                          # 命令行（CLI）
python app.py                           # 网页（Web UI，默认 http://127.0.0.1:7860）
```

CLI 启动后进入交互式命令行：

```
你 > 你好                                       # 普通多轮对话（流式，长对话自动压缩上下文）
你 > /auto 上个月 AI 领域有什么大新闻               # 语义路由：自动判定走 直答/联网/Fusion/本地
你 > /fusionmodel 比较 Rust 和 Go 的并发模型        # 离线 Fusion
你 > /fusionweb 英伟达最新财报关键数字              # 联网 Fusion（面板 agent 检索）
你 > /kb 我的笔记里关于向量库的结论                  # 查本地知识库（需开 RAG 并先 ingest）
你 > /search bocha   /context   /verbose   /models   /help   /exit
```

> 生成过程中按 **Ctrl-C** 只中断当前这次作答 / Fusion，回到提示符；退出请用 `/exit` 或 Ctrl-D。

**Web UI** 把上述能力图形化为三个 Tab：

- **💬 对话**：左侧多会话侧栏（一行一条，新建 / 切换 / 删除）；中间聊天窗，输入框旁 ➕ 展开「自动选择 / Fusion 多模型 / 联网 / 本地知识库」（自动与其余三者互斥），右侧「发送」按钮，生成时变「停止」可中断；Fusion 合成流进对话气泡，「展开过程 ▸」切出裁判裁决 + 各面板标签页。
- **⚙️ 设置**：可视化改角色、系统参数、上下文预算挡位、开关、搜索 / 嵌入 key、各模型（含**增删模型**）、勾选面板模型。保存即写 `user_settings.json` 并热生效（不动 `.env`）。
- **📚 知识库**：拖拽上传 `.md / .txt / .pdf` → 自动切块嵌入入库；表格列出已入库文档与片段数；选中可删除（连同源文件，删除前二次确认）。

---

## 指令（CLI）

| 指令 | 说明 |
|------|------|
| `/auto <问题>` | **语义路由**：自动判定走 直答 / 联网单模型 / 离线 Fusion / 本地知识库 |
| `/fusionmodel [面板] <问题>` | 离线 Fusion（并行作答 → 裁判 → 合成，流式） |
| `/fusionweb [面板] <问题>` | 联网 Fusion（面板 agent 自主检索 + 流式合成） |
| `/kb <问题>` | 查本地知识库（`local_search` agent；需 `RAG_ENABLED` 并先 `ingest.py` 建索引） |
| `/search [provider]` | 查看 / 切换搜索后端（tavily、bocha…） |
| `/models` `/context` `/verbose` `/reset` | 列模型 / 看记忆占用 / 详细模式 / 清空历史 |
| `/help` `/exit` | 帮助 / 退出 |

> 面板可选：`/fusionweb model1,model2 <问题>` 指定本次参与的模型；不指定则用 `DEFAULT_PANEL`。

---

## 配置（.env）

程序从 `MODEL1` 起递增扫描 `MODEL{n}_NAME`，直到某个序号缺失为止（缺 `_BASE_URL` / `_API_KEY` 会直接报错指名缺哪项）。完整示例见 [`Version/v0.4/.env.example`](Version/v0.4/.env.example)。也可在网页「设置」里改，写入 `user_settings.json`（覆盖层，启动叠加在 `.env` 上，不改本文件）。

| 变量 | 说明 |
|------|------|
| `MODEL{n}_NAME / _BASE_URL / _API_KEY` | 模型 id、OpenAI 兼容端点、key |
| `MODEL{n}_TEMPERATURE / _TIMEOUT / _SUPPORTS_TOOLS` | 可选；`SUPPORTS_TOOLS=false` 时该模型联网会直接作答 |
| `DEFAULT_MODEL / JUDGE_MODEL / SYNTHESIZER_MODEL / SUMMARY_MODEL / ROUTER_MODEL` | 各角色用哪个模型（摘要、路由建议选便宜的；缺省回退默认模型） |
| `DEFAULT_PANEL` | Fusion 默认面板 |
| `CHAT_HISTORY_CHAR_BUDGET / CHAT_HISTORY_KEEP_RECENT` | 触发压缩的字符预算 / 始终保留的最近消息条数 |
| `JUDGE_ANSWER_CHAR_LIMIT / JUDGE_EVIDENCE_MAX_ITEMS` | 喂裁判的每个回答字符上限 / 信源条数上限（0=不限） |
| `SEARCH_PROVIDER` + `TAVILY_API_KEY` / `BOCHA_API_KEY` | 默认搜索后端与各后端 key（运行时 `/search` 可切） |
| `WEB_MAX_STEPS / WEB_MAX_RESULTS` | 单个 agent 工具调用轮数上限 / 每次搜索结果数 |
| `REFLEXION_ENABLED / REFLEXION_MAX_ROUNDS` | 反思开关（默认关）/ 修订轮数 |
| `RAG_ENABLED` + `EMBED_MODEL / EMBED_BASE_URL / EMBED_API_KEY` | 本地 RAG 开关 + 嵌入模型（OpenAI 兼容端点；BASE_URL/KEY 不填则复用默认模型的） |
| `RAG_DOCS_DIR / RAG_DB_PATH / RAG_COLLECTION / RAG_TOP_K / RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP / RAG_STORE` | 文档目录 / 向量库路径 / 集合 / 召回数 / 分块参数 / 后端 |

**运行时可选环境变量**（不在 `.env.example` 里，按需在启动时设置）：

| 变量 | 说明 |
|------|------|
| `FUSION_DB` | Web UI 的 SQLite 库路径（默认 `fusion_agent.db`） |
| `FUSION_SETTINGS` | 设置覆盖层文件路径（默认 `user_settings.json`） |
| `FUSION_TRACING` | 设为任意非空值即开启追踪（默认关、零开销） |
| `PHOENIX_OTLP` / `PHOENIX_NO_LAUNCH` | 自定义 OTLP 端点 / 禁用进程内自动起 Phoenix |

---

## 本地 RAG（知识库）

把本地资料接进工具链，让 agent 既能"向外搜"也能"向内查"。

1. `.env` 配 `RAG_ENABLED=true` 与 `EMBED_MODEL`（及可选 `EMBED_BASE_URL` / `EMBED_API_KEY`，需一个 OpenAI 兼容的嵌入端点）。
2. 把笔记 / PDF 放进 `RAG_DOCS_DIR`（默认 `docs/`），跑 `python ingest.py` 建索引（`--rebuild` 清库重建、`--stats` 看块数）；或在网页「📚 知识库」里直接拖拽上传。
3. 之后：Web 勾「本地知识库」或 CLI `/kb <问题>`，agent 即用 `local_search` 检索本地资料；开「自动选择」时，路由判为本地意图也会自动走它。

实现要点：嵌入走 `providers.embed`（OpenAI 兼容，唯一网络出口）；向量库后端可插拔（`vector_store.py`，默认 ChromaDB，cosine 持久化）；`local_search` 与 `web_search` 同构，召回片段产出 `(来源文件, 片段)` 证据，进裁判证据链、可被合成器编号引用。注意：扫描版无文本层的 PDF 取不到正文。

---

## 设置中心（覆盖层）

网页「⚙️ 设置」里改的东西不写回 `.env`，而是写进 `user_settings.json`：启动时先读 `.env`，再叠加该覆盖层，所以**模板与注释不被破坏**。保存即**热生效**（上下文长度等即时生效；改了模型接口 / 嵌入 / 搜索 key 后点「🔄 刷新连接池」让客户端按新配置重建）。

- 留空的 API Key / Base URL **不会**覆盖已有值（防把 key 误抹成空）。
- 支持**增删模型**：每个模型一行（Name / Base URL / API Key 掩码 / 是否支持 tools / 删除），还有「➕ 添加模型」；删掉被某角色或面板引用的模型时会自动回退到现存模型，不会悬空。
- 上下文预算用**离散挡位**（8k / 16k / 32k / 64k / 128k / 256k），避免拖出奇怪数字。
- `user_settings.json` 可能含 API Key，已被 `.gitignore` 忽略。

---

## 可观测性（可选）

零侵入：`providers.py` 是唯一网络出口、全部走 openai SDK，`tracing.py` 一次性 instrument openai，所有模型调用即被追踪（token / 延迟 / prompt / 响应），core 一行不用改。**默认关闭**——只有设了 `FUSION_TRACING` 才启用。

启用后除了每次模型调用的自动 LLM span，还会把链路标成**结构化嵌套 span**（`fusion → panel → judge / synthesize`，以及 `web_search / web_fetch / local_search / reflexion`），得到完整 trace 树。

一条命令查看（**无需另开终端**）：

```bash
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp-proto-http \
            openinference-instrumentation-openai
FUSION_TRACING=1 python app.py        # 或 FUSION_TRACING=1 python main.py
# ↑ 开追踪时会自动在进程内拉起本地 Phoenix（默认 http://localhost:6006）；
#   Web 侧栏的「🔍 打开监控面板 (Phoenix)」可直接跳转。
# 指向外部 OTLP：PHOENIX_OTLP=...   不想自动起本地 Phoenix：PHOENIX_NO_LAUNCH=1
```

**清空历史 trace**：Phoenix 把数据存在它自己的工作目录（默认 `~/.phoenix/phoenix.db`）。想要干净环境，**先停掉 Phoenix**，再跑随附的 `python clean_traces.py`（`--dry-run` 预览 / `--db PATH` 指定）；只删 Phoenix 自己的库文件，不碰本项目代码或 `.env`。

---

## 扩展

- **加模型 / 换厂商**：网页「设置」点「➕ 添加模型」，或 `.env` 加一组 `MODEL{n}_*`。不兼容 OpenAI 协议的厂商只改 `providers.py` 一处。
- **加搜索后端**：在 `search_providers.py` 写 `class XxxProvider(SearchProvider)` 并 `register_provider`，再配 `XXX_API_KEY`。
- **换向量库**：在 `vector_store.py` 写 `class XxxVectorStore(VectorStore)` 并 `register_store`，`RAG_STORE` 选它即可，`rag.py` 不用改。
- **加前端**：核心是事件驱动（`stream_fusion` / `stream_agent` 产出事件），CLI 与 Web 都只是渲染器——新前端消费同一套事件流即可。
- **调效果**：直接改 `prompts/` 下的裁判 / 合成 / 面板 agent / 摘要 / 反思 / 路由 模板。

---

## 安全

- 所有密钥只放 `.env` 或网页设置写的 `user_settings.json`；两者都被 `.gitignore` 排除，仓库只提交 `.env.example`（占位符）。本地数据（向量库 `rag_chroma/`、会话库 `*.db`、`docs/`）也已忽略。
- 不在日志中打印完整 api_key；网页里 API Key 框一律密码掩码。
- 可观测性默认关闭、纯本地（trace 只发往你自己起的本地 Phoenix）；开了也不会把 key 写进 span。
- 联网 Fusion 一次会产生「面板数 + 2」次模型调用并可能多轮检索，开反思再各加一次；注意成本，可用 `WEB_MAX_STEPS` / `WEB_MAX_RESULTS` / `REFLEXION_*` 约束。

---

## 版本历史

- **v0.4（当前）** —— 从"能跑"走向"好用、可控、能向内看"的一轮：
  - **语义路由**（`router.py`）：启发式 + 便宜模型分类，自动判定 直答 / 联网 / Fusion / 本地，并显式回显；CLI `/auto`、Web「自动选择」。
  - **反思 Reflexion**（`agent.py`）：联网出初稿后对照证据自检、剔除/修正无支撑主张再产出修订版（opt-in）。
  - **本地 RAG**（`rag.py` / `vector_store.py` / `ingest.py`）：`.md/.txt/.pdf` → 切块 → 嵌入（OpenAI 兼容）→ ChromaDB；agent `local_search` 召回进证据链；Web「📚 知识库」上传/列表/删除，CLI `/kb`。
  - **可视化设置中心**（Web「⚙️ 设置」+ `config.py` 覆盖层）：改角色/参数/各模型/搜索/嵌入 key、**增删模型**、勾面板、预算挡位；写 `user_settings.json` 不碰 `.env`，热生效。
  - **Web 体验**：Tabs（对话/设置/知识库）、**发送/停止按钮（可中断生成）**、开关互斥、对话列表一行一条；开追踪时**进程内自动起 Phoenix** + 侧栏「打开监控面板」。
- **v0.3** —— 持久化 + Web UI + 可观测性：`store.py`（SQLite 多会话，窗口可无损重建）、`app.py`（Gradio）、`tracing.py`（自动追踪 + 结构化 span，本地 Phoenix）、推理流折叠的延迟体感优化、合成吃面板原文写"深度研报"。
- **v0.2** —— 长对话上下文管理、裁判输入上限、Ctrl-C 只中断当前任务、架构精简为单一流式路径、若干稳健性修复。
- **v0.1** —— 初版：离线 / 联网两种 Fusion 模式，全链路流式，证据链闭环 + 内联引用，可插拔搜索后端。

---

## 路线图

- Fusion 实时**三栏流式辩论** + 裁判**雷达评分**（面板带模型标签透传 + 裁判产出每模型维度评分）。
- 联网单模型 / Fusion 路径也展示推理流；语义路由更细（local 与 web/fusion 叠加的更优策略）。
- 文本协议工具调用（兼容无原生 tools 的模型）、YAML 预设面板、加权 / 投票、结果缓存、对外暴露 OpenAI 兼容服务。

---

## 许可证

[MIT](LICENSE) © 2026 Maxwell-00
