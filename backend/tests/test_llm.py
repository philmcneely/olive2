"""LLM service and API tests (port of llm.test.ts)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from olive.app import create_app
from olive.db import create_db
from olive.services.llm import LlmService, normalize_provider_base_url


async def test_lists_models_across_multiple_providers_with_metadata(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))

    all_models = await service.list_models()
    assert len(all_models) >= 35

    anthropic_models = await service.list_models({"provider": "anthropic"})
    assert len(anthropic_models) > 0
    assert any("claude" in m.id for m in anthropic_models)

    google_models = await service.list_models({"provider": "google"})
    assert len(google_models) > 0
    assert any("gemini" in m.id for m in google_models)

    openai_models = await service.list_models({"provider": "openai"})
    assert len(openai_models) > 0
    assert any("gpt-4" in m.id for m in openai_models)
    handle.engine.dispose()


async def test_lists_provider_summaries_with_configuration_state(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))

    providers = await service.list_providers()
    assert len(providers) > 5

    google_prov = next((p for p in providers if p.id == "google"), None)
    assert google_prov is not None
    assert google_prov.name == "Google"
    assert google_prov.modelCount > 0
    handle.engine.dispose()


async def test_manages_settings_and_custom_model_definitions(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))

    initial = await service.get_settings()
    assert initial.defaultProvider

    updated = await service.update_settings(
        {
            "defaultProvider": "openai",
            "defaultModel": "gpt-4o",
            "providers": {"openai": {"baseUrl": "http://localhost:11434/v1"}},
            "customModels": [
                {
                    "id": "local-llama",
                    "name": "Local Llama 3.3",
                    "provider": "openai",
                    "baseUrl": "http://localhost:11434/v1",
                    "contextWindow": 64000,
                }
            ],
        }
    )

    assert updated.defaultProvider == "openai"
    assert updated.defaultModel == "gpt-4o"
    assert updated.providers["openai"].baseUrl == "http://localhost:11434/v1"
    assert len(updated.customModels) == 1

    models = await service.list_models({"provider": "openai"})
    local_llama = next((m for m in models if m.id == "local-llama"), None)
    assert local_llama is not None
    assert local_llama.source == "custom"

    # Saving an API key and then updating other settings without apiKey must preserve apiKey
    await service.update_settings(
        {
            "defaultProvider": "google",
            "defaultModel": "gemini-2.5-flash",
            "providers": {
                "google": {
                    "apiKey": "sk-secret-preserved-key",
                    "baseUrl": "https://plexus.home.cowger.us/v1",
                }
            },
        }
    )
    after_model_change = await service.update_settings(
        {
            "defaultProvider": "google",
            "defaultModel": "gemini-2.5-pro",
            "defaultThinkingLevel": "high",
            "providers": {
                "google": {
                    "baseUrl": "https://plexus.home.cowger.us/v1",
                }
            },
        }
    )
    assert after_model_change.providers["google"].apiKey == "sk-secret-preserved-key"
    assert after_model_change.providers["google"].baseUrl == "https://plexus.home.cowger.us/v1"
    assert after_model_change.defaultModel == "gemini-2.5-pro"

    # Empty or masked apiKey also preserves the stored key
    after_empty_key = await service.update_settings(
        {
            "providers": {
                "google": {
                    "apiKey": "",
                    "baseUrl": "https://plexus.home.cowger.us",
                }
            }
        }
    )
    assert after_empty_key.providers["google"].apiKey == "sk-secret-preserved-key"
    assert after_empty_key.providers["google"].baseUrl == "https://plexus.home.cowger.us"
    handle.engine.dispose()


async def test_resolves_models_with_custom_base_url_overrides(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))

    await service.update_settings(
        {"providers": {"groq": {"baseUrl": "https://custom-groq-proxy.internal/v1"}}}
    )
    model = await service.resolve_model("groq", "llama-3.3-70b-versatile")
    assert model["baseUrl"] == "https://custom-groq-proxy.internal/v1"

    for suffix in ("", "/v1", "/v1/", "/v1beta", "/v1/messages", "/v1/chat/completions"):
        url = f"https://plexus.home.cowger.us{suffix}"
        assert normalize_provider_base_url("google", url) == "https://plexus.home.cowger.us"
        assert normalize_provider_base_url("anthropic", url) == "https://plexus.home.cowger.us"
        assert normalize_provider_base_url("openai", url) == "https://plexus.home.cowger.us/v1"
        assert normalize_provider_base_url("groq", url) == "https://plexus.home.cowger.us/v1"
    assert (
        normalize_provider_base_url("openrouter", "https://openrouter.ai/api/v1/chat/completions")
        == "https://openrouter.ai/api/v1"
    )
    handle.engine.dispose()


async def test_throws_informative_error_when_api_key_missing(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))

    with pytest.raises(RuntimeError, match="No API key configured for provider"):
        await service.generate_text(
            {
                "provider": "cerebras",
                "model": "llama-3.3-70b",
                "prompt": "Hello",
            }
        )
    handle.engine.dispose()


def test_llm_api_endpoints(tmp_path: Path) -> None:
    handle = create_db(":memory:")
    app = create_app({"db": handle.engine, "configDir": str(tmp_path)})
    client = TestClient(app)

    prov_res = client.get("/api/llm/providers")
    assert prov_res.status_code == 200
    assert len(prov_res.json()["providers"]) > 5

    models_res = client.get("/api/llm/models?provider=google")
    assert models_res.status_code == 200
    assert len(models_res.json()["models"]) > 0

    cfg_res = client.get("/api/llm/config")
    assert cfg_res.status_code == 200
    assert "defaultProvider" in cfg_res.json()

    upd_res = client.post(
        "/api/llm/config",
        json={
            "defaultProvider": "anthropic",
            "defaultModel": "claude-3-7-sonnet-20250219",
            "providers": {"anthropic": {"baseUrl": "https://api.anthropic.com"}},
        },
    )
    assert upd_res.status_code == 200
    assert upd_res.json()["settings"]["defaultProvider"] == "anthropic"

    if not os.environ.get("GEMINI_API_KEY"):
        test_res = client.post(
            "/api/llm/test",
            json={"provider": "nonexistent", "model": "dummy"},
        )
        assert test_res.status_code == 400

    handle.engine.dispose()


class _FakeMessage:
    def __init__(self, content: str = "", reasoning_content: str = "") -> None:
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, message: _FakeMessage) -> None:
        self.message = message


class _FakeResponse:
    def __init__(self, message: _FakeMessage) -> None:
        self.choices = [_FakeChoice(message)]
        self.usage = None


async def test_disables_thinking_on_custom_base_and_falls_back_to_reasoning_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A self-hosted (custom base) non-reasoning model gets enable_thinking=False,
    and an empty content body falls back to reasoning_content."""
    handle = create_db(str(tmp_path / "db.sqlite"))
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))
    await service.update_settings(
        {
            "defaultProvider": "openai",
            "defaultModel": "qwen3.8-27b",
            "providers": {"openai": {"baseUrl": "http://127.0.0.1:8011/v1", "apiKey": "local"}},
        }
    )

    captured: dict = {}

    import olive.services.llm as llm_mod

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(_FakeMessage(content="", reasoning_content='{"moments": []}'))

    monkeypatch.setattr(llm_mod.any_llm, "acompletion", fake_acompletion)

    res = await service.generate_text(
        {"model": "qwen3.8-27b", "prompt": "hi", "provider": "openai"}
    )

    assert captured.get("extra_body", {}).get("chat_template_kwargs", {}).get(
        "enable_thinking"
    ) is False
    # Every completion is bounded so a wedged endpoint cannot hang a worker.
    assert captured.get("timeout") == 600.0
    assert res.text == '{"moments": []}'
    handle.engine.dispose()


async def test_llm_timeout_is_operator_tunable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OLIVE_LLM_TIMEOUT overrides the default completion timeout."""
    monkeypatch.setenv("OLIVE_LLM_TIMEOUT", "42")
    handle = create_db(str(tmp_path / "db.sqlite"))
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))
    await service.update_settings(
        {
            "defaultProvider": "openai",
            "defaultModel": "gpt-4o-mini",
            "providers": {"openai": {"apiKey": "sk-test"}},
        }
    )

    captured: dict = {}

    import olive.services.llm as llm_mod

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(_FakeMessage(content="ok"))

    monkeypatch.setattr(llm_mod.any_llm, "acompletion", fake_acompletion)

    res = await service.generate_text(
        {"model": "gpt-4o-mini", "prompt": "hi", "provider": "openai"}
    )
    assert captured.get("timeout") == 42.0
    assert res.text == "ok"
    handle.engine.dispose()


async def test_no_thinking_override_without_custom_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default cloud base (no custom baseUrl) must not receive chat_template_kwargs."""
    handle = create_db(str(tmp_path / "db.sqlite"))
    service = LlmService(db=handle.engine, config_dir=str(tmp_path))
    await service.update_settings(
        {
            "defaultProvider": "openai",
            "defaultModel": "gpt-4o-mini",
            "providers": {"openai": {"apiKey": "sk-test"}},
        }
    )

    captured: dict = {}

    import olive.services.llm as llm_mod

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(_FakeMessage(content="ok"))

    monkeypatch.setattr(llm_mod.any_llm, "acompletion", fake_acompletion)

    res = await service.generate_text(
        {"model": "gpt-4o-mini", "prompt": "hi", "provider": "openai"}
    )
    assert "extra_body" not in captured
    assert res.text == "ok"
    handle.engine.dispose()
