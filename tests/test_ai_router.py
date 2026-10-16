import asyncio

from fastapi import HTTPException

from app.ai_router import _provider_order, chat, provider_status


def test_provider_order_is_configurable(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER_ORDER", "groq,openai,cerebras,unknown")
    assert _provider_order() == ["groq", "openai", "cerebras"]


def test_ai_status_never_reports_missing_keys_as_ready(monkeypatch):
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "AI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("AI_ENDPOINT", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)
    status = provider_status()
    assert status["ready"] is False
    assert status["configured_count"] == 0
    assert all(item["configured"] is False for item in status["providers"])


def test_chat_fails_clearly_when_no_provider_is_configured(monkeypatch):
    for key in ("OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "AI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("AI_ENDPOINT", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)
    try:
        asyncio.run(chat("سلام"))
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "پیکربندی نشده" in str(exc.detail)
    else:
        raise AssertionError("Expected missing provider configuration to fail closed")
