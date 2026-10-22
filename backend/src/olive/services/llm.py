"""LLM provider & model catalog service powered by any-llm and models.dev (port of llm/service.ts)."""

from __future__ import annotations

import inspect
import json
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import any_llm
import httpx
from sqlalchemy import text
from sqlalchemy.engine import Engine

from ..db import DbHandle, get_db
from ..logging_db import logger
from ..schemas import (
    ChatUsage,
    LlmGenerateOptions,
    LlmGenerateResult,
    LlmModelCatalogItem,
    LlmModelCost,
    LlmProviderSummary,
    LlmSettings,
    LlmThinkingLevel,
    UpdateLlmSettingsInput,
)
from ..settings import is_masked_secret, resolve_paths

PROVIDER_ENV_KEY_MAP: dict[str, list[str]] = {
    "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "groq": ["GROQ_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY"],
    "deepseek": ["DEEPSEEK_API_KEY"],
    "mistral": ["MISTRAL_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "fireworks": ["FIREWORKS_API_KEY"],
    "cerebras": ["CEREBRAS_API_KEY"],
    "xai": ["XAI_API_KEY"],
    "baseten": ["BASETEN_API_KEY"],
}

PROVIDER_PRIORITY: dict[str, int] = {
    "google": 1,
    "anthropic": 2,
    "openai": 3,
    "groq": 4,
    "openrouter": 5,
    "deepseek": 6,
    "mistral": 7,
}

DEFAULT_PROVIDER_BASE_URL: dict[str, str] = {
    "google": "https://generativelanguage.googleapis.com",
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "mistral": "https://api.mistral.ai/v1",
    "together": "https://api.together.xyz/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "cerebras": "https://api.cerebras.ai/v1",
    "xai": "https://api.x.ai/v1",
}

WEB_CATALOG_URL = "https://models.dev/api.json"
WEB_CATALOG_CACHE_FILENAME = "models-web-cache.json"
# Fresh installs have no cache, so the first boot fetches the live catalog.
# Subsequent boots skip the network call while the cache is younger than this.
WEB_CATALOG_STALE_AFTER_SECONDS = 24 * 3600

_ENDPOINT_SUFFIX_RE = re.compile(
    r"(?:/v1/chat/completions|/chat/completions|/v1/messages|/messages|/v1/responses|/responses|/v1beta/models|/v1beta1|/v1beta|/v1alpha|/v1)+$",
    re.IGNORECASE,
)


def normalize_provider_base_url(provider: str, base_url: str | None) -> str | None:
    """Normalize a user-supplied base URL for the target provider's SDK.

    - ``google`` / ``gemini`` (``google-genai`` SDK) appends ``/v1beta/models/...`` itself,
      so ``/v1``, ``/v1beta``, ``/v1/messages``, or ``/v1/chat/completions`` are stripped.
    - ``anthropic`` (``anthropic`` SDK) appends ``/v1/messages`` itself, so ``/v1`` and
      ``/v1/messages`` are stripped.
    - OpenAI-compatible SDKs (``openai``, ``groq``, ``openrouter``, etc.) append
      ``/chat/completions`` directly and require a trailing ``/v1`` path segment.
    """
    if not base_url or not base_url.strip():
        return None

    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        return None

    prov = provider.strip().lower()
    root = _ENDPOINT_SUFFIX_RE.sub("", cleaned).rstrip("/")
    if not root:
        root = cleaned

    if prov in ("google", "gemini", "anthropic"):
        return root

    if re.search(r"/v\d+(?:alpha|beta\d*)?$", root, re.IGNORECASE):
        return root
    return f"{root}/v1"


# Built-in snapshot of models across major providers (offline fallback for models.dev)
BUILTIN_CATALOG: list[dict[str, Any]] = [
    # Google Gemini
    {
        "id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "reasoning": True,
        "cost": {"input": 0.15, "output": 0.60},
    },
    {
        "id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "reasoning": True,
        "cost": {"input": 1.25, "output": 10.0},
    },
    {
        "id": "gemini-2.5-flash-lite",
        "name": "Gemini 2.5 Flash Lite",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "reasoning": True,
        "cost": {"input": 0.10, "output": 0.40},
    },
    {
        "id": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.10, "output": 0.40},
    },
    {
        "id": "gemini-2.0-flash-lite",
        "name": "Gemini 2.0 Flash Lite",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.075, "output": 0.30},
    },
    {
        "id": "gemini-3.7-flash",
        "name": "Gemini 3.7 Flash",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "reasoning": True,
        "cost": {"input": 0.15, "output": 0.60},
    },
    {
        "id": "gemini-1.5-pro",
        "name": "Gemini 1.5 Pro",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 2097152,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 1.25, "output": 5.0},
    },
    {
        "id": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "provider": "google",
        "api": "google-generative-ai",
        "contextWindow": 1048576,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.075, "output": 0.30},
    },
    # Anthropic
    {
        "id": "claude-sonnet-4-20250514",
        "name": "Claude Sonnet 4",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 64000,
        "reasoning": True,
        "cost": {"input": 3.0, "output": 15.0},
    },
    {
        "id": "claude-opus-4-20250514",
        "name": "Claude Opus 4",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 32000,
        "reasoning": True,
        "cost": {"input": 15.0, "output": 75.0},
    },
    {
        "id": "claude-3-7-sonnet-20250219",
        "name": "Claude 3.7 Sonnet",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 64000,
        "reasoning": True,
        "cost": {"input": 3.0, "output": 15.0},
    },
    {
        "id": "claude-3-5-sonnet-20241022",
        "name": "Claude 3.5 Sonnet",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 3.0, "output": 15.0},
    },
    {
        "id": "claude-3-5-haiku-20241022",
        "name": "Claude 3.5 Haiku",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.80, "output": 4.0},
    },
    {
        "id": "claude-3-opus-20240229",
        "name": "Claude 3 Opus",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 4096,
        "reasoning": False,
        "cost": {"input": 15.0, "output": 75.0},
    },
    {
        "id": "claude-3-haiku-20240307",
        "name": "Claude 3 Haiku",
        "provider": "anthropic",
        "api": "anthropic-messages",
        "contextWindow": 200000,
        "maxTokens": 4096,
        "reasoning": False,
        "cost": {"input": 0.25, "output": 1.25},
    },
    # OpenAI
    {
        "id": "gpt-4o",
        "name": "GPT-4o",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": False,
        "cost": {"input": 2.5, "output": 10.0},
    },
    {
        "id": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": False,
        "cost": {"input": 0.15, "output": 0.60},
    },
    {
        "id": "gpt-4.1",
        "name": "GPT-4.1",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 1047576,
        "maxTokens": 32768,
        "reasoning": False,
        "cost": {"input": 2.0, "output": 8.0},
    },
    {
        "id": "gpt-4.1-mini",
        "name": "GPT-4.1 Mini",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 1047576,
        "maxTokens": 32768,
        "reasoning": False,
        "cost": {"input": 0.40, "output": 1.60},
    },
    {
        "id": "gpt-4.1-nano",
        "name": "GPT-4.1 Nano",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 1047576,
        "maxTokens": 32768,
        "reasoning": False,
        "cost": {"input": 0.10, "output": 0.40},
    },
    {
        "id": "gpt-4-turbo",
        "name": "GPT-4 Turbo",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 4096,
        "reasoning": False,
        "cost": {"input": 10.0, "output": 30.0},
    },
    {
        "id": "o3",
        "name": "OpenAI o3",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 200000,
        "maxTokens": 100000,
        "reasoning": True,
        "cost": {"input": 2.0, "output": 8.0},
    },
    {
        "id": "o3-mini",
        "name": "OpenAI o3-mini",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 200000,
        "maxTokens": 100000,
        "reasoning": True,
        "cost": {"input": 1.1, "output": 4.4},
    },
    {
        "id": "o4-mini",
        "name": "OpenAI o4-mini",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 200000,
        "maxTokens": 100000,
        "reasoning": True,
        "cost": {"input": 1.1, "output": 4.4},
    },
    {
        "id": "o1",
        "name": "OpenAI o1",
        "provider": "openai",
        "api": "openai-completions",
        "contextWindow": 200000,
        "maxTokens": 100000,
        "reasoning": True,
        "cost": {"input": 15.0, "output": 60.0},
    },
    # Groq
    {
        "id": "llama-3.3-70b-versatile",
        "name": "Llama 3.3 70B Versatile",
        "provider": "groq",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 32768,
        "reasoning": False,
        "cost": {"input": 0.59, "output": 0.79},
    },
    {
        "id": "llama-3.1-8b-instant",
        "name": "Llama 3.1 8B Instant",
        "provider": "groq",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.05, "output": 0.08},
    },
    {
        "id": "deepseek-r1-distill-llama-70b",
        "name": "DeepSeek R1 Distill Llama 70B",
        "provider": "groq",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": True,
        "cost": {"input": 0.75, "output": 0.99},
    },
    {
        "id": "qwen-qwq-32b",
        "name": "Qwen QwQ 32B",
        "provider": "groq",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": True,
        "cost": {"input": 0.29, "output": 0.39},
    },
    {
        "id": "gemma2-9b-it",
        "name": "Gemma 2 9B IT",
        "provider": "groq",
        "api": "openai-completions",
        "contextWindow": 8192,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.20, "output": 0.20},
    },
    # OpenRouter
    {
        "id": "anthropic/claude-3.7-sonnet",
        "name": "Claude 3.7 Sonnet (OpenRouter)",
        "provider": "openrouter",
        "api": "openai-completions",
        "contextWindow": 200000,
        "maxTokens": 64000,
        "reasoning": True,
    },
    {
        "id": "google/gemini-2.5-pro",
        "name": "Gemini 2.5 Pro (OpenRouter)",
        "provider": "openrouter",
        "api": "openai-completions",
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "reasoning": True,
    },
    {
        "id": "google/gemini-2.5-flash",
        "name": "Gemini 2.5 Flash (OpenRouter)",
        "provider": "openrouter",
        "api": "openai-completions",
        "contextWindow": 1048576,
        "maxTokens": 65536,
        "reasoning": True,
    },
    {
        "id": "openai/gpt-4o",
        "name": "GPT-4o (OpenRouter)",
        "provider": "openrouter",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": False,
    },
    {
        "id": "deepseek/deepseek-r1",
        "name": "DeepSeek R1 (OpenRouter)",
        "provider": "openrouter",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 32768,
        "reasoning": True,
    },
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "name": "Llama 3.3 70B Instruct (OpenRouter)",
        "provider": "openrouter",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 32768,
        "reasoning": False,
    },
    # DeepSeek
    {
        "id": "deepseek-chat",
        "name": "DeepSeek V3 Chat",
        "provider": "deepseek",
        "api": "openai-completions",
        "contextWindow": 64000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.27, "output": 1.10},
    },
    {
        "id": "deepseek-reasoner",
        "name": "DeepSeek R1 Reasoner",
        "provider": "deepseek",
        "api": "openai-completions",
        "contextWindow": 64000,
        "maxTokens": 8192,
        "reasoning": True,
        "cost": {"input": 0.55, "output": 2.19},
    },
    # Mistral
    {
        "id": "mistral-large-latest",
        "name": "Mistral Large",
        "provider": "mistral",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 2.0, "output": 6.0},
    },
    {
        "id": "mistral-small-latest",
        "name": "Mistral Small",
        "provider": "mistral",
        "api": "openai-completions",
        "contextWindow": 32000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.1, "output": 0.3},
    },
    {
        "id": "codestral-latest",
        "name": "Codestral",
        "provider": "mistral",
        "api": "openai-completions",
        "contextWindow": 256000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.3, "output": 0.9},
    },
    {
        "id": "ministral-8b-latest",
        "name": "Ministral 8B",
        "provider": "mistral",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 8192,
        "reasoning": False,
        "cost": {"input": 0.1, "output": 0.1},
    },
    # xAI
    {
        "id": "grok-3",
        "name": "Grok 3",
        "provider": "xai",
        "api": "openai-completions",
        "contextWindow": 131072,
        "maxTokens": 16384,
        "reasoning": False,
        "cost": {"input": 3.0, "output": 15.0},
    },
    {
        "id": "grok-3-mini",
        "name": "Grok 3 Mini",
        "provider": "xai",
        "api": "openai-completions",
        "contextWindow": 131072,
        "maxTokens": 16384,
        "reasoning": True,
        "cost": {"input": 0.3, "output": 0.5},
    },
    {
        "id": "grok-2-1212",
        "name": "Grok 2",
        "provider": "xai",
        "api": "openai-completions",
        "contextWindow": 131072,
        "maxTokens": 16384,
        "reasoning": False,
        "cost": {"input": 2.0, "output": 10.0},
    },
    # Cerebras
    {
        "id": "llama-3.3-70b",
        "name": "Llama 3.3 70B (Cerebras)",
        "provider": "cerebras",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 8192,
        "reasoning": False,
    },
    {
        "id": "llama3.1-8b",
        "name": "Llama 3.1 8B (Cerebras)",
        "provider": "cerebras",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 8192,
        "reasoning": False,
    },
    {
        "id": "qwen-3-32b",
        "name": "Qwen 3 32B (Cerebras)",
        "provider": "cerebras",
        "api": "openai-completions",
        "contextWindow": 32768,
        "maxTokens": 8192,
        "reasoning": True,
    },
    # Together
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "name": "Llama 3.3 70B Instruct Turbo",
        "provider": "together",
        "api": "openai-completions",
        "contextWindow": 131072,
        "maxTokens": 8192,
        "reasoning": False,
    },
    {
        "id": "deepseek-ai/DeepSeek-R1",
        "name": "DeepSeek R1 (Together)",
        "provider": "together",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": True,
    },
    {
        "id": "Qwen/Qwen2.5-72B-Instruct-Turbo",
        "name": "Qwen 2.5 72B Instruct Turbo",
        "provider": "together",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 8192,
        "reasoning": False,
    },
    # Fireworks
    {
        "id": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "name": "Llama 3.3 70B Instruct (Fireworks)",
        "provider": "fireworks",
        "api": "openai-completions",
        "contextWindow": 131072,
        "maxTokens": 16384,
        "reasoning": False,
    },
    {
        "id": "accounts/fireworks/models/deepseek-r1",
        "name": "DeepSeek R1 (Fireworks)",
        "provider": "fireworks",
        "api": "openai-completions",
        "contextWindow": 128000,
        "maxTokens": 16384,
        "reasoning": True,
    },
]


class LlmService:
    def __init__(
        self,
        options: dict[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        config_dir: str | None = None,
    ) -> None:
        opts = options or {}
        raw_db = db or opts.get("db")
        if isinstance(raw_db, DbHandle):
            self.db: Engine = raw_db.engine
        elif isinstance(raw_db, Engine):
            self.db = raw_db
        else:
            self.db = get_db()

        self.config_dir = str(
            config_dir or opts.get("configDir") or opts.get("config_dir") or resolve_paths().config_dir
        )
        self.web_catalog: dict[str, LlmModelCatalogItem] = {}
        self._load_cached_web_catalog()

    def _web_catalog_cache_path(self) -> Path:
        return Path(self.config_dir) / WEB_CATALOG_CACHE_FILENAME

    def _load_cached_web_catalog(self) -> None:
        try:
            cache_path = self._web_catalog_cache_path()
            if cache_path.exists():
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    for entry in raw:
                        item = LlmModelCatalogItem.model_validate(entry)
                        self.web_catalog[f"{item.provider}:{item.id}"] = item
        except Exception:
            pass

    def is_web_catalog_stale(self, max_age_seconds: float = WEB_CATALOG_STALE_AFTER_SECONDS) -> bool:
        """True when the cached web catalog is missing, empty, or older than *max_age_seconds*."""
        if not self.web_catalog:
            return True
        try:
            mtime = self._web_catalog_cache_path().stat().st_mtime
        except OSError:
            return True
        return (time.time() - mtime) > max_age_seconds

    async def refresh_catalog_if_stale(
        self, max_age_seconds: float = WEB_CATALOG_STALE_AFTER_SECONDS
    ) -> dict[str, int]:
        """Refresh the web catalog only when the on-disk cache is missing or stale.

        Returns ``{"count": n, "refreshed": bool}`` so callers (e.g. server
        startup) can skip the models.dev network call on every boot.
        """
        if not self.is_web_catalog_stale(max_age_seconds):
            return {"count": len(self.web_catalog), "refreshed": False}
        result = await self.refresh_catalog()
        return {"count": result.get("count", len(self.web_catalog)), "refreshed": True}

    def _save_web_catalog(self) -> None:
        try:
            Path(self.config_dir).mkdir(parents=True, exist_ok=True)
            cache_path = self._web_catalog_cache_path()
            items = [item.model_dump(exclude_none=True) for item in self.web_catalog.values()]
            cache_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warn("Failed to persist web model cache", {"error": str(exc)})

    async def get_settings(self) -> LlmSettings:
        with self.db.connect() as conn:
            row = (
                conn.execute(text("SELECT value FROM sync_state WHERE key = 'llm_settings'"))
                .mappings()
                .first()
            )

        if row and row.get("value"):
            try:
                parsed = json.loads(row["value"])
                return LlmSettings.model_validate(parsed)
            except Exception:
                pass

        default_provider = "google"
        default_model = "gemini-2.5-flash"

        if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
            default_provider = "google"
            default_model = "gemini-2.5-flash"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            default_provider = "anthropic"
            default_model = "claude-3-7-sonnet-20250219"
        elif os.environ.get("OPENAI_API_KEY"):
            default_provider = "openai"
            default_model = "gpt-4o"
        elif os.environ.get("GROQ_API_KEY"):
            default_provider = "groq"
            default_model = "llama-3.3-70b-versatile"

        return LlmSettings(
            defaultProvider=default_provider,
            defaultModel=default_model,
            defaultThinkingLevel="off",
            providers={},
            customModels=[],
        )

    getSettings = get_settings

    async def update_settings(self, raw_input: dict[str, Any] | UpdateLlmSettingsInput) -> LlmSettings:
        raw_providers_input = (
            raw_input.get("providers")
            if isinstance(raw_input, dict) and isinstance(raw_input.get("providers"), dict)
            else None
        )
        inp = (
            raw_input
            if isinstance(raw_input, UpdateLlmSettingsInput)
            else UpdateLlmSettingsInput.model_validate(raw_input)
        )
        current = await self.get_settings()
        merged_providers: dict[str, dict[str, Any]] = {
            k: v.model_dump(exclude_none=True) for k, v in current.providers.items()
        }
        for prov_id, incoming in (inp.providers or {}).items():
            existing = dict(merged_providers.get(prov_id) or {})
            raw_prov_candidate = (
                raw_providers_input.get(prov_id) if isinstance(raw_providers_input, dict) else None
            )
            raw_prov: dict[str, Any] = raw_prov_candidate if isinstance(raw_prov_candidate, dict) else {}

            if incoming.apiKey is not None:
                trimmed_key = incoming.apiKey.strip()
                if trimmed_key and not is_masked_secret(trimmed_key):
                    existing["apiKey"] = trimmed_key

            if "baseUrl" in raw_prov or incoming.baseUrl is not None:
                trimmed_url = (incoming.baseUrl or "").strip()
                if trimmed_url:
                    existing["baseUrl"] = trimmed_url
                else:
                    existing.pop("baseUrl", None)

            if incoming.headers is not None:
                existing["headers"] = incoming.headers

            merged_providers[prov_id] = existing
        updated = LlmSettings.model_validate(
            {
                "defaultProvider": inp.defaultProvider or current.defaultProvider,
                "defaultModel": inp.defaultModel or current.defaultModel,
                "defaultThinkingLevel": inp.defaultThinkingLevel or current.defaultThinkingLevel or "off",
                "providers": merged_providers,
                "customModels": [
                    c.model_dump(exclude_none=True)
                    for c in (inp.customModels if inp.customModels is not None else current.customModels)
                ],
            }
        )
        json_val = json.dumps(updated.model_dump(exclude_none=True))
        with self.db.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO sync_state (key, value)
                    VALUES ('llm_settings', :val)
                    ON CONFLICT(key) DO UPDATE SET value = :val
                    """
                ),
                {"val": json_val},
            )

        logger.info(
            "Updated LLM settings",
            {
                "defaultProvider": updated.defaultProvider,
                "defaultModel": updated.defaultModel,
                "defaultThinkingLevel": updated.defaultThinkingLevel,
            },
        )
        return updated

    updateSettings = update_settings

    async def refresh_catalog(self) -> dict[str, int]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.get(
                    WEB_CATALOG_URL,
                    headers={"User-Agent": "Olive/0.2"},
                )
                if res.status_code == 200:
                    data = res.json()
                    if isinstance(data, dict):
                        for provider_id, pdata in data.items():
                            if not isinstance(pdata, dict) or not isinstance(pdata.get("models"), dict):
                                continue
                            for model_id, mdef in pdata["models"].items():
                                if not isinstance(mdef, dict):
                                    continue
                                limit_obj = mdef.get("limit") or {}
                                cost_obj = mdef.get("cost")
                                cost = (
                                    LlmModelCost(
                                        input=float(cost_obj.get("input") or 0),
                                        output=float(cost_obj.get("output") or 0),
                                        cacheRead=cost_obj.get("cacheRead") or cost_obj.get("cache_read"),
                                        cacheWrite=cost_obj.get("cacheWrite") or cost_obj.get("cache_write"),
                                    )
                                    if isinstance(cost_obj, dict)
                                    else None
                                )
                                reasoning = bool(mdef.get("reasoning") or mdef.get("thinking"))
                                item = LlmModelCatalogItem(
                                    id=model_id,
                                    name=mdef.get("name") or model_id,
                                    provider=provider_id,
                                    api=mdef.get("api")
                                    or (
                                        "anthropic-messages"
                                        if provider_id == "anthropic"
                                        else "openai-completions"
                                    ),
                                    baseUrl=mdef.get("baseUrl") or pdata.get("baseUrl"),
                                    contextWindow=mdef.get("contextWindow") or limit_obj.get("context"),
                                    maxTokens=mdef.get("maxTokens") or limit_obj.get("output"),
                                    reasoning=reasoning,
                                    supportedThinkingLevels=["low", "medium", "high"]
                                    if reasoning
                                    else ["off"],
                                    cost=cost,
                                    source="web",
                                )
                                self.web_catalog[f"{provider_id}:{model_id}"] = item
                        self._save_web_catalog()
                        logger.info("Refreshed web model catalog", {"modelsCount": len(self.web_catalog)})
        except Exception as exc:
            logger.warn("Failed to fetch live web catalog, using fallback", {"error": str(exc)})
        return {"count": len(self.web_catalog)}

    refreshCatalog = refresh_catalog

    refreshCatalogIfStale = refresh_catalog_if_stale
    isWebCatalogStale = is_web_catalog_stale

    def get_model_supported_thinking_levels(
        self, model_info: dict[str, Any] | LlmModelCatalogItem
    ) -> list[LlmThinkingLevel]:
        d = model_info.model_dump() if isinstance(model_info, LlmModelCatalogItem) else model_info
        if not d.get("reasoning"):
            return ["off"]
        provider = str(d.get("provider") or "")
        model_id = str(d.get("id") or "").lower()
        if provider == "google" and re.search(r"gemini-3(?:\.\d+)?-flash", model_id):
            return ["low", "medium", "high"]
        return ["off", "low", "medium", "high"]

    def resolve_effective_reasoning(
        self,
        model_info: dict[str, Any] | LlmModelCatalogItem,
        requested: LlmThinkingLevel | None = None,
        default_level: LlmThinkingLevel | None = None,
    ) -> LlmThinkingLevel | None:
        d = model_info.model_dump() if isinstance(model_info, LlmModelCatalogItem) else model_info
        if not d.get("reasoning"):
            return None
        supported = self.get_model_supported_thinking_levels(d)
        candidate: LlmThinkingLevel = requested or default_level or "off"
        if candidate == "off":
            return "off" if "off" in supported else (supported[0] if supported else "low")
        if candidate in supported:
            return candidate
        return supported[0] if supported else "low"

    async def list_models(self, filter_opts: dict[str, Any] | None = None) -> list[LlmModelCatalogItem]:
        f = filter_opts or {}
        settings = await self.get_settings()
        catalog_map: dict[str, LlmModelCatalogItem] = {}

        # 1. Built-in catalog
        for entry in BUILTIN_CATALOG:
            key = f"{entry['provider']}:{entry['id']}"
            reasoning = bool(entry.get("reasoning"))
            cost_raw = entry.get("cost")
            item = LlmModelCatalogItem(
                id=entry["id"],
                name=entry.get("name") or entry["id"],
                provider=entry["provider"],
                api=entry.get("api") or "openai-completions",
                baseUrl=entry.get("baseUrl") or DEFAULT_PROVIDER_BASE_URL.get(entry["provider"]),
                contextWindow=entry.get("contextWindow"),
                maxTokens=entry.get("maxTokens"),
                reasoning=reasoning,
                supportedThinkingLevels=self.get_model_supported_thinking_levels(entry),
                cost=LlmModelCost(**cost_raw) if isinstance(cost_raw, dict) else None,
                source="builtin",
            )
            catalog_map[key] = item

        # 2. Web catalog
        for key, item in self.web_catalog.items():
            if key not in catalog_map:
                catalog_map[key] = item

        # 3. Custom models from settings
        for custom in settings.customModels:
            key = f"{custom.provider}:{custom.id}"
            reasoning = bool(custom.reasoning)
            catalog_map[key] = LlmModelCatalogItem(
                id=custom.id,
                name=custom.name,
                provider=custom.provider,
                api=custom.api or "openai-completions",
                baseUrl=custom.baseUrl,
                contextWindow=custom.contextWindow,
                maxTokens=custom.maxTokens,
                reasoning=reasoning,
                supportedThinkingLevels=["low", "medium", "high"] if reasoning else ["off"],
                source="custom",
            )

        items = list(catalog_map.values())

        provider_filter = (f.get("provider") or "").strip().lower()
        if provider_filter:
            items = [m for m in items if m.provider.lower() == provider_filter]

        search_filter = (f.get("search") or "").strip().lower()
        if search_filter:
            items = [
                m
                for m in items
                if search_filter in m.name.lower()
                or search_filter in m.id.lower()
                or search_filter in m.provider.lower()
            ]

        items.sort(
            key=lambda m: (
                PROVIDER_PRIORITY.get(m.provider, 99),
                m.provider,
                m.name,
            )
        )

        limit = f.get("limit")
        if isinstance(limit, int) and limit > 0:
            return items[:limit]
        return items

    listModels = list_models

    async def list_providers(self) -> list[LlmProviderSummary]:
        models = await self.list_models()
        settings = await self.get_settings()

        provider_counts: dict[str, tuple[str, int]] = {}
        for m in models:
            if m.provider in provider_counts:
                name, cnt = provider_counts[m.provider]
                provider_counts[m.provider] = (name, cnt + 1)
            else:
                formatted = " ".join(part.capitalize() for part in m.provider.split("-"))
                provider_counts[m.provider] = (formatted, 1)

        summaries: list[LlmProviderSummary] = []
        for pid, (name, cnt) in provider_counts.items():
            env_keys = PROVIDER_ENV_KEY_MAP.get(pid, [])
            has_env_key = any(bool(os.environ.get(k)) for k in env_keys)
            cfg = settings.providers.get(pid)
            has_stored_key = bool(cfg and cfg.apiKey)
            has_custom_base = bool(cfg and cfg.baseUrl)
            summaries.append(
                LlmProviderSummary(
                    id=pid,
                    name=name,
                    isConfigured=has_env_key or has_stored_key,
                    hasCustomBaseUrl=has_custom_base,
                    modelCount=cnt,
                )
            )

        summaries.sort(key=lambda s: (PROVIDER_PRIORITY.get(s.id, 99), s.name))
        return summaries

    listProviders = list_providers

    def resolve_api_key(
        self,
        provider: str,
        override: str | None = None,
        settings: LlmSettings | None = None,
    ) -> str | None:
        if override and override.strip():
            return override.strip()
        if settings and provider in settings.providers and settings.providers[provider].apiKey:
            return settings.providers[provider].apiKey
        env_keys = PROVIDER_ENV_KEY_MAP.get(
            provider,
            [f"{re.sub(r'[^A-Z0-9]', '_', provider.upper())}_API_KEY"],
        )
        for k in env_keys:
            val = os.environ.get(k)
            if val:
                return val
        return None

    resolveApiKey = resolve_api_key

    async def resolve_model(
        self,
        provider: str,
        model_id: str,
        base_url_override: str | None = None,
    ) -> dict[str, Any]:
        settings = await self.get_settings()
        provider_cfg = settings.providers.get(provider)
        cfg_base = provider_cfg.baseUrl if provider_cfg else None

        # Check custom models
        custom = next(
            (c for c in settings.customModels if c.provider == provider and c.id == model_id),
            None,
        )
        if custom:
            return {
                "id": custom.id,
                "name": custom.name,
                "api": custom.api or "openai-completions",
                "provider": custom.provider,
                "baseUrl": base_url_override
                or custom.baseUrl
                or cfg_base
                or DEFAULT_PROVIDER_BASE_URL.get(provider, "https://api.openai.com/v1"),
                "reasoning": bool(custom.reasoning),
                "contextWindow": custom.contextWindow or 128000,
                "maxTokens": custom.maxTokens or 4096,
            }

        # Check builtin catalog
        builtin = next(
            (b for b in BUILTIN_CATALOG if b["provider"] == provider and b["id"] == model_id),
            None,
        )
        if builtin:
            return {
                **builtin,
                "baseUrl": base_url_override
                or cfg_base
                or builtin.get("baseUrl")
                or DEFAULT_PROVIDER_BASE_URL.get(provider, "https://api.openai.com/v1"),
            }

        # Check web catalog
        web_def = self.web_catalog.get(f"{provider}:{model_id}")
        if web_def:
            return {
                **web_def.model_dump(),
                "baseUrl": base_url_override
                or cfg_base
                or web_def.baseUrl
                or DEFAULT_PROVIDER_BASE_URL.get(provider, "https://api.openai.com/v1"),
            }

        return {
            "id": model_id,
            "name": model_id,
            "api": "anthropic-messages" if provider == "anthropic" else "openai-completions",
            "provider": provider,
            "baseUrl": base_url_override
            or cfg_base
            or DEFAULT_PROVIDER_BASE_URL.get(provider, "https://api.openai.com/v1"),
            "reasoning": False,
            "contextWindow": 128000,
            "maxTokens": 4096,
        }

    resolveModel = resolve_model

    def _map_any_llm_provider(self, provider: str) -> str:
        mapping = {
            "google": "gemini",
        }
        return mapping.get(provider.lower(), provider.lower())

    async def generate_text(self, raw_options: dict[str, Any] | LlmGenerateOptions) -> LlmGenerateResult:
        opts = (
            raw_options
            if isinstance(raw_options, LlmGenerateOptions)
            else LlmGenerateOptions.model_validate(raw_options)
        )
        start_ms = int(time.time() * 1000)
        settings = await self.get_settings()

        provider = opts.provider or settings.defaultProvider
        model_id = opts.model or settings.defaultModel

        api_key = self.resolve_api_key(provider, opts.apiKeyOverride, settings)
        if not api_key:
            raise RuntimeError(
                f'No API key configured for provider "{provider}". '
                "Please configure an API key in settings or pass one in the request."
            )

        model_info = await self.resolve_model(provider, model_id, opts.baseUrlOverride)
        messages: list[dict[str, Any]] = []
        if opts.systemPrompt:
            messages.append({"role": "system", "content": opts.systemPrompt})
        if opts.messages:
            for m in opts.messages:
                messages.append({"role": m.role, "content": m.content})
        else:
            messages.append({"role": "user", "content": opts.prompt})

        effective_reasoning = self.resolve_effective_reasoning(
            model_info, opts.thinkingLevel, settings.defaultThinkingLevel
        )

        provider_cfg = settings.providers.get(provider)
        default_base = DEFAULT_PROVIDER_BASE_URL.get(provider)
        resolved_base = model_info.get("baseUrl")
        raw_custom_base = (
            opts.baseUrlOverride
            or (provider_cfg.baseUrl if provider_cfg else None)
            or (resolved_base if resolved_base and resolved_base != default_base else None)
        )
        custom_base = normalize_provider_base_url(provider, raw_custom_base)
        any_provider = self._map_any_llm_provider(provider)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "provider": any_provider,
            "messages": messages,
            "api_key": api_key,
        }
        if custom_base:
            kwargs["api_base"] = custom_base
        if provider_cfg and provider_cfg.headers:
            kwargs["client_args"] = {"default_headers": provider_cfg.headers}
        if opts.temperature is not None:
            kwargs["temperature"] = opts.temperature
        if opts.maxTokens is not None:
            kwargs["max_tokens"] = opts.maxTokens
        if effective_reasoning and effective_reasoning != "off":
            kwargs["reasoning_effort"] = effective_reasoning

        # Self-hosted OpenAI-compatible endpoints (vLLM / llama.cpp / ollama) accept
        # ``chat_template_kwargs``. When the caller wants no reasoning, disable the
        # model's thinking at the template level so the output token budget is not
        # spent on a ``<think>`` block — qwen3.x and similar otherwise emit their
        # reasoning into ``reasoning_content`` and leave ``content`` empty, which the
        # extractor reads as a failed generation. Only ever sent to a custom base,
        # never to a default cloud provider endpoint (which would 400 on it).
        if custom_base and (not effective_reasoning or effective_reasoning == "off"):
            extra_body = dict(kwargs.get("extra_body") or {})
            ctk = dict(extra_body.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = ctk
            kwargs["extra_body"] = extra_body

        # Bound every completion so a hung or stalled self-hosted endpoint (vLLM
        # OOM, a wedged llama.cpp slot) cannot hold a worker open forever. Large
        # local models on slow GPUs legitimately take minutes per generation, so
        # the ceiling is generous and operator-tunable via OLIVE_LLM_TIMEOUT.
        kwargs.setdefault("timeout", float(os.environ.get("OLIVE_LLM_TIMEOUT", "600")))

        logger.info(
            "Executing LLM generation",
            {
                "provider": provider,
                "model": model_id,
                "baseUrl": model_info.get("baseUrl"),
                "reasoning": effective_reasoning,
            },
        )

        try:
            response: Any = await any_llm.acompletion(**kwargs)
        except Exception as exc:
            err_msg = str(exc)
            logger.error(
                "LLM completion call failed", {"provider": provider, "model": model_id, "error": err_msg}
            )
            raise RuntimeError(f"LLM call failed ({provider}/{model_id}): {err_msg}") from exc

        choice = response.choices[0] if getattr(response, "choices", None) else None
        text_content = ""
        if choice and getattr(choice, "message", None):
            # Prefer ``content``; fall back to ``reasoning_content`` for endpoints
            # that stream the answer there when thinking could not be disabled.
            text_content = (
                getattr(choice.message, "content", "")
                or getattr(choice.message, "reasoning_content", "")
                or ""
            )

        usage_obj = getattr(response, "usage", None)
        in_tok = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj else 0
        out_tok = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj else 0
        tot_tok = (
            int(getattr(usage_obj, "total_tokens", in_tok + out_tok) or (in_tok + out_tok))
            if usage_obj
            else 0
        )

        duration_ms = max(1, int(time.time() * 1000) - start_ms)
        return LlmGenerateResult(
            text=text_content,
            provider=provider,
            model=model_id,
            durationMs=duration_ms,
            usage=ChatUsage(
                inputTokens=in_tok,
                outputTokens=out_tok,
                totalTokens=tot_tok,
            ),
        )

    generateText = generate_text

    async def stream_text(
        self,
        raw_options: dict[str, Any] | LlmGenerateOptions,
        on_delta: Callable[[str], Any],
    ) -> LlmGenerateResult:
        opts = (
            raw_options
            if isinstance(raw_options, LlmGenerateOptions)
            else LlmGenerateOptions.model_validate(raw_options)
        )
        start_ms = int(time.time() * 1000)
        settings = await self.get_settings()

        provider = opts.provider or settings.defaultProvider
        model_id = opts.model or settings.defaultModel

        api_key = self.resolve_api_key(provider, opts.apiKeyOverride, settings)
        if not api_key:
            raise RuntimeError(
                f'No API key configured for provider "{provider}". '
                "Please configure an API key in settings or pass one in the request."
            )

        model_info = await self.resolve_model(provider, model_id, opts.baseUrlOverride)
        messages: list[dict[str, Any]] = []
        if opts.systemPrompt:
            messages.append({"role": "system", "content": opts.systemPrompt})
        if opts.messages:
            for m in opts.messages:
                messages.append({"role": m.role, "content": m.content})
        else:
            messages.append({"role": "user", "content": opts.prompt})

        effective_reasoning = self.resolve_effective_reasoning(
            model_info, opts.thinkingLevel, settings.defaultThinkingLevel
        )

        provider_cfg = settings.providers.get(provider)
        default_base = DEFAULT_PROVIDER_BASE_URL.get(provider)
        resolved_base = model_info.get("baseUrl")
        raw_custom_base = (
            opts.baseUrlOverride
            or (provider_cfg.baseUrl if provider_cfg else None)
            or (resolved_base if resolved_base and resolved_base != default_base else None)
        )
        custom_base = normalize_provider_base_url(provider, raw_custom_base)
        any_provider = self._map_any_llm_provider(provider)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "provider": any_provider,
            "messages": messages,
            "api_key": api_key,
            "stream": True,
        }
        if custom_base:
            kwargs["api_base"] = custom_base
        if provider_cfg and provider_cfg.headers:
            kwargs["client_args"] = {"default_headers": provider_cfg.headers}
        if opts.temperature is not None:
            kwargs["temperature"] = opts.temperature
        if opts.maxTokens is not None:
            kwargs["max_tokens"] = opts.maxTokens
        if effective_reasoning and effective_reasoning != "off":
            kwargs["reasoning_effort"] = effective_reasoning

        stream = await any_llm.acompletion(**kwargs)
        full_text = ""
        in_tok = 0
        out_tok = 0
        tot_tok = 0

        async for chunk in stream:  # type: ignore[union-attr]
            choices = getattr(chunk, "choices", None)
            if choices:
                delta_obj = getattr(choices[0], "delta", None)
                piece = getattr(delta_obj, "content", None) if delta_obj else None
                if piece:
                    full_text += piece
                    cb_res = on_delta(piece)
                    if inspect.isawaitable(cb_res):
                        await cb_res
            usage_obj = getattr(chunk, "usage", None)
            if usage_obj:
                in_tok = int(getattr(usage_obj, "prompt_tokens", in_tok) or in_tok)
                out_tok = int(getattr(usage_obj, "completion_tokens", out_tok) or out_tok)
                tot_tok = int(getattr(usage_obj, "total_tokens", in_tok + out_tok) or (in_tok + out_tok))

        duration_ms = max(1, int(time.time() * 1000) - start_ms)
        return LlmGenerateResult(
            text=full_text,
            provider=provider,
            model=model_id,
            durationMs=duration_ms,
            usage=ChatUsage(
                inputTokens=in_tok,
                outputTokens=out_tok,
                totalTokens=tot_tok,
            ),
        )

    streamText = stream_text
