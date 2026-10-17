from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import HTTPException

PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "key_env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
    },
    "groq": {
        "key_env": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.3-70b-versatile",
    },
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model_env": "OPENROUTER_MODEL",
        "default_model": "meta-llama/llama-3.3-70b-instruct:free",
    },
    "cerebras": {
        "key_env": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model_env": "CEREBRAS_MODEL",
        "default_model": "llama-3.3-70b",
    },
    "custom": {
        "key_env": "AI_API_KEY",
        "url_env": "AI_ENDPOINT",
        "model_env": "AI_MODEL",
        "default_model": "",
    },
}


def _provider_order() -> list[str]:
    raw = os.getenv("AI_PROVIDER_ORDER", "openai,groq,openrouter,cerebras,custom")
    requested = [name.strip().lower() for name in raw.split(",") if name.strip()]
    return [name for name in requested if name in PROVIDERS]


def provider_status() -> dict[str, Any]:
    order = _provider_order()
    items = []
    for name in order:
        spec = PROVIDERS[name]
        key = os.getenv(spec["key_env"], "").strip()
        url = os.getenv(spec.get("url_env", ""), "").strip() if spec.get("url_env") else spec["url"]
        model = os.getenv(spec.get("model_env", ""), "").strip() or spec.get("default_model", "")
        configured = bool(key and url and model)
        items.append({"provider": name, "configured": configured, "model": model if configured else None})
    return {
        "fallback_order": order,
        "configured_count": sum(1 for item in items if item["configured"]),
        "providers": items,
        "ready": any(item["configured"] for item in items),
        "note": "کلیدها فقط از متغیرهای محرمانهٔ محیط سرور خوانده می‌شوند و در پاسخ نمایش داده نمی‌شوند.",
    }


async def chat(message: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    if not message.strip():
        raise HTTPException(status_code=422, detail="پیام خالی است.")
    status = provider_status()
    configured = [item for item in status["providers"] if item["configured"]]
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="هیچ ارائه‌دهندهٔ هوش مصنوعی پیکربندی نشده است. کلیدهای OPENAI_API_KEY، GROQ_API_KEY، OPENROUTER_API_KEY یا CEREBRAS_API_KEY را در تنظیمات محرمانهٔ سرور اضافه کنید؛ برای سرویس سازگار سفارشی از AI_ENDPOINT و AI_API_KEY و AI_MODEL استفاده کنید.",
        )

    safe_history: list[dict[str, str]] = []
    for item in (history or [])[-8:]:
        if isinstance(item, dict) and item.get("role") in {"user", "assistant"} and isinstance(item.get("content"), str):
            safe_history.append({"role": item["role"], "content": item["content"][:4000]})
    messages = [
        {
            "role": "system",
            "content": (
                "You are the Persian-language Strategy Foundry assistant. Explain clearly and honestly. "
                "Treat the user's requested trading rules and app behavior as requirements to clarify and implement, not as a fixed or sacred contract. "
                "The visible settings/profile are editable defaults; do not reject a request merely because it is not in the default profile. "
                "If a requested capability is not implemented or no execution tool is available, say exactly what is missing and do not pretend it was implemented. "
                "Offer a concrete next step in plain Persian, ask only for information essential to proceed, and do not send the user searching through technical screens. "
                "This application is simulation-only; never claim to place real orders, run code, fix source code, or verify profitability unless a corresponding tool result explicitly confirms it. "
                "Do not invent backtest results. Distinguish suggestions from executed operations."
            ),
        },
        *safe_history,
        {"role": "user", "content": message[:8000]},
    ]
    failures = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(35.0, connect=8.0), follow_redirects=True) as client:
        for item in configured:
            name = item["provider"]
            spec = PROVIDERS[name]
            api_key = os.getenv(spec["key_env"], "").strip()
            url = os.getenv(spec.get("url_env", ""), "").strip() if spec.get("url_env") else spec["url"]
            model = os.getenv(spec.get("model_env", ""), "").strip() or spec.get("default_model", "")
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            if name == "openrouter":
                headers["HTTP-Referer"] = os.getenv("OPENROUTER_SITE_URL", "https://strategy-foundry.de1.netrun.io")
                headers["X-Title"] = "Strategy Foundry"
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json={"model": model, "messages": messages, "temperature": 0.2, "max_tokens": 1200},
                )
                response.raise_for_status()
                payload = response.json()
                answer = payload["choices"][0]["message"]["content"]
                if isinstance(answer, list):
                    answer = "".join(part.get("text", "") for part in answer if isinstance(part, dict))
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("Provider returned an empty response")
                return {"ok": True, "provider": name, "model": model, "fallback_used": name != configured[0]["provider"], "answer": answer.strip()}
            except Exception as exc:
                # Only provider names and exception classes are retained; credentials and response bodies are never returned.
                failures.append({"provider": name, "error": type(exc).__name__})
    raise HTTPException(
        status_code=502,
        detail={"message": "تمام ارائه‌دهندگان پیکربندی‌شده ناموفق بودند؛ کلیدها، محدودیت سهمیه و مدل‌ها را بررسی کنید.", "attempts": failures},
    )
