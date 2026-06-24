"""配置加载层。

从环境变量扫描 MODEL{n}_* 构建模型注册表，并读取角色分工与 v2 联网搜索配置。

设计要点：
- 所有模型统一走 OpenAI 兼容接口，新增厂商只需在 .env 加一组 MODEL{n}_*。
- 搜索后端可插拔：填了哪个 provider 的 *_API_KEY 就能用哪个，SEARCH_PROVIDER 选默认。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from search_providers import available_providers


@dataclass
class ModelConfig:
    """单个模型的配置。"""

    key: str            # 逻辑标识，如 "model1"
    name: str           # 厂商侧模型 id，如 "deepseek-v4-pro"
    base_url: str
    api_key: str
    temperature: float = 0.7
    timeout: float = 60.0
    supports_tools: bool = True  # 是否支持 OpenAI 原生 function-calling


@dataclass
class AppConfig:
    """整个应用的配置：模型注册表 + 角色分工 + 搜索后端。"""

    models: dict[str, ModelConfig]
    default_model: str
    judge_model: str
    synthesizer_model: str
    default_panel: list[str]
    # ---- 普通对话上下文管理（滚动摘要 + 最近 N 轮）----
    summary_model: str = ""          # 压缩对话用哪个模型；空则回退 default_model
    history_char_budget: int = 8000  # turns 超过这么多字符就触发压缩
    history_keep_recent: int = 6     # 始终逐字保留的最近消息条数
    # ---- 裁判输入上限（防长答案 / 多信源撑爆裁判上下文）----
    judge_answer_char_limit: int = 4000  # 每个面板回答喂给裁判的字符上限（头+尾截断，0=不限）
    judge_evidence_max_items: int = 24   # 喂给裁判的信源条数上限（0=不限）
    # ---- v2：面板联网（搜索后端可插拔）----
    search_provider: str = "tavily"
    search_keys: dict[str, str] = field(default_factory=dict)      # provider -> api_key
    search_base_urls: dict[str, str] = field(default_factory=dict)  # provider -> base_url(可选)
    web_max_steps: int = 4       # 单个面板 agent 最多工具调用轮数
    web_max_results: int = 5     # 每次 web_search 返回结果数
    # ---- v0.4：反思 Reflexion（agent 出初稿后对照证据自检/修订）----
    reflexion_enabled: bool = False   # 默认关；opt-in，每启用一次额外一次模型调用
    reflexion_max_rounds: int = 1     # 修订轮数（draft→revise 迭代次数）
    # ---- v0.4：语义路由（自动判定 直答/联网/Fusion）----
    router_model: str = ""            # 路由分类器用哪个模型；空则回退 summary_model/default_model
    # ---- 5.3：本地 RAG（向量库 + 嵌入；agent 的 local_search 工具，默认关）----
    rag_enabled: bool = False
    embed_cfg: "ModelConfig | None" = None   # 嵌入模型（OpenAI 兼容端点）；None=未配置
    rag_store: str = "chroma"                # 向量库后端
    rag_db_path: str = "rag_chroma"          # 向量库持久化目录
    rag_collection: str = "fusion_kb"
    rag_docs_dir: str = "docs"               # ingest 扫描的文档目录
    rag_top_k: int = 5
    rag_chunk_size: int = 1000               # 分块字符数
    rag_chunk_overlap: int = 150

    def get(self, key: str) -> ModelConfig:
        if key not in self.models:
            raise KeyError(
                f"未找到模型 '{key}'，已配置：{', '.join(self.models) or '(无)'}"
            )
        return self.models[key]

    # ---- 搜索后端便捷访问 ----
    @property
    def web_enabled(self) -> bool:
        return self.search_provider in self.search_keys

    @property
    def rag_ready(self) -> bool:
        """本地 RAG 可用：开了开关且配了嵌入模型（还需先 `python ingest.py` 建索引）。"""
        return self.rag_enabled and self.embed_cfg is not None

    def active_search_key(self) -> str | None:
        return self.search_keys.get(self.search_provider)

    def active_search_base_url(self) -> str | None:
        return self.search_base_urls.get(self.search_provider)


def _as_bool(v: str, default: bool = True) -> bool:
    return v.strip().lower() not in ("false", "0", "no", "off") if v else default


def load_models() -> dict[str, ModelConfig]:
    """从 MODEL1 开始递增扫描，直到某个序号的 *_NAME 缺失为止。"""
    models: dict[str, ModelConfig] = {}
    i = 1
    while os.getenv(f"MODEL{i}_NAME"):
        key = f"model{i}"
        # 只判 _NAME 存在就建模型，但 _BASE_URL/_API_KEY 缺失会在下面直接 KeyError，
        # 报错很难懂；这里显式校验，缺哪个就指名道姓。
        missing = [
            f"MODEL{i}_{suffix}"
            for suffix in ("BASE_URL", "API_KEY")
            if not os.getenv(f"MODEL{i}_{suffix}")
        ]
        if missing:
            raise RuntimeError(
                f"{key} 配置不完整，缺少：{', '.join(missing)}。"
                f"请在 .env 中补全（每个模型需要 _NAME / _BASE_URL / _API_KEY）。"
            )
        models[key] = ModelConfig(
            key=key,
            name=os.environ[f"MODEL{i}_NAME"],
            base_url=os.environ[f"MODEL{i}_BASE_URL"],
            api_key=os.environ[f"MODEL{i}_API_KEY"],
            temperature=float(os.getenv(f"MODEL{i}_TEMPERATURE", "0.7")),
            timeout=float(os.getenv(f"MODEL{i}_TIMEOUT", "60")),
            supports_tools=_as_bool(os.getenv(f"MODEL{i}_SUPPORTS_TOOLS", "true")),
        )
        i += 1
    return models


def _load_search() -> tuple[str, dict[str, str], dict[str, str]]:
    """按已注册的搜索 provider 扫描 {NAME}_API_KEY / {NAME}_BASE_URL。"""
    keys: dict[str, str] = {}
    base_urls: dict[str, str] = {}
    for name in available_providers():
        k = os.getenv(f"{name.upper()}_API_KEY")
        if k:
            keys[name] = k
        b = os.getenv(f"{name.upper()}_BASE_URL")
        if b:
            base_urls[name] = b
    provider = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower()
    return provider, keys, base_urls


# ===========================================================================
# user_settings.json 覆盖层：启动时叠加在 .env 之上；Web「设置」页保存即写这里、热更内存 AppConfig。
# 不碰 .env（避免破坏模板/格式）；空的敏感字段（API Key/Base URL/Name）不覆盖已有值。
# ===========================================================================
SETTINGS_FILE = os.getenv("FUSION_SETTINGS", "user_settings.json")

_OVERRIDABLE_SCALARS = (
    "history_char_budget", "history_keep_recent",
    "judge_answer_char_limit", "judge_evidence_max_items",
    "web_max_steps", "web_max_results", "rag_top_k", "reflexion_max_rounds",
    "reflexion_enabled", "rag_enabled",
    "default_model", "judge_model", "synthesizer_model", "summary_model",
    "router_model", "search_provider", "default_panel",
    "rag_store", "rag_db_path", "rag_collection", "rag_docs_dir",
    "rag_chunk_size", "rag_chunk_overlap",
)
_SENSITIVE_FIELDS = ("name", "base_url", "api_key")  # 模型字段里这些为空时不覆盖


def load_user_settings(path: str | None = None) -> dict:
    """读 user_settings.json（覆盖层）；不存在 / 损坏 → {}。"""
    try:
        with open(path or SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _merge_settings(cur: dict, new: dict) -> dict:
    """把 new 合并进 cur（深合并 models）；空字符串的敏感字段不覆盖已有值。"""
    out = dict(cur)
    for k, v in new.items():
        if k == "models":
            models = dict(out.get("models") or {})
            for mk, mv in (v or {}).items():
                if mv is None:
                    models[mk] = None          # 删除标记，保留以便启动时删除
                    continue
                m = dict(models.get(mk) or {})
                for fk, fv in (mv or {}).items():
                    if fk in _SENSITIVE_FIELDS and not str(fv or "").strip():
                        continue
                    m[fk] = fv
                models[mk] = m
            out["models"] = models
        elif k == "embed":
            e = dict(out.get("embed") or {})
            for fk, fv in (v or {}).items():
                if fk in ("model", "base_url", "api_key") and not str(fv or "").strip():
                    continue
                e[fk] = fv
            out["embed"] = e
        elif k == "search_keys":
            sk = dict(out.get("search_keys") or {})
            for prov, key in (v or {}).items():
                if str(key or "").strip():
                    sk[prov] = key
            out["search_keys"] = sk
        elif v is None or (isinstance(v, str) and not v.strip()):
            continue
        else:
            out[k] = v
    return out


def save_user_settings(overrides: dict, path: str | None = None) -> dict:
    """合并写入 user_settings.json（保留旧键、空敏感字段不覆盖）；返回合并后的全量 dict。"""
    p = path or SETTINGS_FILE
    merged = _merge_settings(load_user_settings(p), overrides)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return merged


def apply_overrides(cfg: AppConfig, ov: dict) -> None:
    """把覆盖层就地应用到 AppConfig（标量 + 各模型 name/base_url/api_key），供启动加载与运行时热更。"""
    if not ov:
        return
    for k in _OVERRIDABLE_SCALARS:
        if ov.get(k) is not None:
            setattr(cfg, k, ov[k])
    for mkey, mov in (ov.get("models") or {}).items():
        if mov is None or (isinstance(mov, dict) and mov.get("_deleted")):
            cfg.models.pop(mkey, None)                      # 删除模型
            continue
        if not isinstance(mov, dict):
            continue
        mc = cfg.models.get(mkey)
        if mc is None:                                      # 新增：齐全才建
            name = str(mov.get("name") or "").strip()
            base = str(mov.get("base_url") or "").strip()
            akey = str(mov.get("api_key") or "").strip()
            if name and base and akey:
                cfg.models[mkey] = ModelConfig(
                    mkey, name, base, akey,
                    supports_tools=bool(mov.get("supports_tools", True)),
                )
            continue
        for fk in _SENSITIVE_FIELDS:                        # patch 已有：空敏感字段不覆盖
            val = mov.get(fk)
            if val and str(val).strip():
                setattr(mc, fk, val)
        if "supports_tools" in mov:
            mc.supports_tools = bool(mov["supports_tools"])

    # 嵌入模型覆盖（EMBED_*）：model 非空才建；base_url/api_key 空则沿用现有 / 默认模型
    emb = ov.get("embed")
    if isinstance(emb, dict) and str(emb.get("model") or "").strip():
        cur = cfg.embed_cfg
        fb = cfg.models.get(cfg.default_model)
        base = str(emb.get("base_url") or "").strip() or (cur.base_url if cur else (fb.base_url if fb else ""))
        akey = str(emb.get("api_key") or "").strip() or (cur.api_key if cur else (fb.api_key if fb else ""))
        cfg.embed_cfg = ModelConfig("embed", str(emb["model"]).strip(), base, akey)

    # 搜索后端 key 覆盖（空不覆盖）
    sk = ov.get("search_keys")
    if isinstance(sk, dict):
        for prov, key in sk.items():
            if str(key or "").strip():
                cfg.search_keys[prov] = str(key).strip()

    # 删模型 / 改角色后修复悬空引用，避免后续 cfg.get() 崩
    if cfg.models:
        first = next(iter(cfg.models))
        for role in ("default_model", "judge_model", "synthesizer_model", "summary_model", "router_model"):
            if getattr(cfg, role) not in cfg.models:
                setattr(cfg, role, first)
        cfg.default_panel = [k for k in cfg.default_panel if k in cfg.models] or [first]


def load_config() -> AppConfig:
    """加载完整配置，并对角色引用做基础校验（尽早失败）。"""
    models = load_models()
    if not models:
        raise RuntimeError(
            "没有扫描到任何模型配置。请先 `cp .env.example .env` 并至少配置 MODEL1_*。"
        )

    first = next(iter(models))
    default_model = os.getenv("DEFAULT_MODEL", first)
    judge_model = os.getenv("JUDGE_MODEL", default_model)
    synthesizer_model = os.getenv("SYNTHESIZER_MODEL", default_model)
    summary_model = os.getenv("SUMMARY_MODEL", default_model)
    router_model = os.getenv("ROUTER_MODEL", summary_model)

    raw_panel = os.getenv("DEFAULT_PANEL", ",".join(models))
    default_panel = [p.strip() for p in raw_panel.split(",") if p.strip()]

    for role, key in (
        ("DEFAULT_MODEL", default_model),
        ("JUDGE_MODEL", judge_model),
        ("SYNTHESIZER_MODEL", synthesizer_model),
        ("SUMMARY_MODEL", summary_model),
        ("ROUTER_MODEL", router_model),
    ):
        if key not in models:
            raise RuntimeError(f"{role}={key} 不在已配置模型中：{', '.join(models)}")
    for key in default_panel:
        if key not in models:
            raise RuntimeError(
                f"DEFAULT_PANEL 含未知模型 {key}，可用：{', '.join(models)}"
            )

    search_provider, search_keys, search_base_urls = _load_search()

    embed_cfg = None
    if os.getenv("EMBED_MODEL"):
        dm = models[default_model]  # 嵌入端点默认复用默认模型的 base_url/key（同厂商常通用）
        embed_cfg = ModelConfig(
            "embed",
            os.environ["EMBED_MODEL"],
            os.getenv("EMBED_BASE_URL", dm.base_url),
            os.getenv("EMBED_API_KEY", dm.api_key),
            timeout=float(os.getenv("EMBED_TIMEOUT", "60")),
        )

    cfg = AppConfig(
        models=models,
        default_model=default_model,
        judge_model=judge_model,
        synthesizer_model=synthesizer_model,
        summary_model=summary_model,
        router_model=router_model,
        history_char_budget=int(os.getenv("CHAT_HISTORY_CHAR_BUDGET", "8000")),
        history_keep_recent=int(os.getenv("CHAT_HISTORY_KEEP_RECENT", "6")),
        judge_answer_char_limit=int(os.getenv("JUDGE_ANSWER_CHAR_LIMIT", "4000")),
        judge_evidence_max_items=int(os.getenv("JUDGE_EVIDENCE_MAX_ITEMS", "24")),
        default_panel=default_panel,
        search_provider=search_provider,
        search_keys=search_keys,
        search_base_urls=search_base_urls,
        web_max_steps=int(os.getenv("WEB_MAX_STEPS", "4")),
        web_max_results=int(os.getenv("WEB_MAX_RESULTS", "5")),
        reflexion_enabled=_as_bool(os.getenv("REFLEXION_ENABLED", ""), default=False),
        reflexion_max_rounds=int(os.getenv("REFLEXION_MAX_ROUNDS", "1")),
        rag_enabled=_as_bool(os.getenv("RAG_ENABLED", ""), default=False),
        embed_cfg=embed_cfg,
        rag_store=os.getenv("RAG_STORE", "chroma"),
        rag_db_path=os.getenv("RAG_DB_PATH", "rag_chroma"),
        rag_collection=os.getenv("RAG_COLLECTION", "fusion_kb"),
        rag_docs_dir=os.getenv("RAG_DOCS_DIR", "docs"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "5")),
        rag_chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "1000")),
        rag_chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "150")),
    )
    apply_overrides(cfg, load_user_settings())  # user_settings.json 覆盖 .env（启动即生效）
    return cfg
