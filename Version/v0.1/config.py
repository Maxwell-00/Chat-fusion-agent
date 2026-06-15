"""配置加载层。

从环境变量扫描 MODEL{n}_* 构建模型注册表，并读取角色分工与 v2 联网搜索配置。

设计要点：
- 所有模型统一走 OpenAI 兼容接口，新增厂商只需在 .env 加一组 MODEL{n}_*。
- 搜索后端可插拔：填了哪个 provider 的 *_API_KEY 就能用哪个，SEARCH_PROVIDER 选默认。
"""
from __future__ import annotations

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
    # ---- v2：面板联网（搜索后端可插拔）----
    search_provider: str = "tavily"
    search_keys: dict[str, str] = field(default_factory=dict)      # provider -> api_key
    search_base_urls: dict[str, str] = field(default_factory=dict)  # provider -> base_url(可选)
    web_max_steps: int = 4       # 单个面板 agent 最多工具调用轮数
    web_max_results: int = 5     # 每次 web_search 返回结果数

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

    raw_panel = os.getenv("DEFAULT_PANEL", ",".join(models))
    default_panel = [p.strip() for p in raw_panel.split(",") if p.strip()]

    for role, key in (
        ("DEFAULT_MODEL", default_model),
        ("JUDGE_MODEL", judge_model),
        ("SYNTHESIZER_MODEL", synthesizer_model),
    ):
        if key not in models:
            raise RuntimeError(f"{role}={key} 不在已配置模型中：{', '.join(models)}")
    for key in default_panel:
        if key not in models:
            raise RuntimeError(
                f"DEFAULT_PANEL 含未知模型 {key}，可用：{', '.join(models)}"
            )

    search_provider, search_keys, search_base_urls = _load_search()

    return AppConfig(
        models=models,
        default_model=default_model,
        judge_model=judge_model,
        synthesizer_model=synthesizer_model,
        default_panel=default_panel,
        search_provider=search_provider,
        search_keys=search_keys,
        search_base_urls=search_base_urls,
        web_max_steps=int(os.getenv("WEB_MAX_STEPS", "4")),
        web_max_results=int(os.getenv("WEB_MAX_RESULTS", "5")),
    )
