"""Thin, shared wrapper around the Google Gen AI SDK (``google-genai``).

* One lazily created client for the whole app.
* Friendly, user-facing error messages (missing key, quota, bad key ...).
* Automatic model fallback: if the configured model name is unknown/retired
  (HTTP 404), we try a few known-good names and finally auto-discover one.
"""
import logging
import threading
from typing import Iterator, Optional

from google import genai
from google.genai import errors, types

import config
from exceptions import AIServiceError

log = logging.getLogger("edugenie.gemini")

_FALLBACK_MODELS = ("gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash")

_lock = threading.Lock()
_client: Optional[genai.Client] = None
_working_model: Optional[str] = None


def _get_client() -> genai.Client:
    global _client
    if not config.GEMINI_API_KEY:
        raise AIServiceError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env, paste your key "
            "(get one free at https://aistudio.google.com/app/apikey) and restart the server.",
            503,
        )
    with _lock:
        if _client is None:
            _client = genai.Client(api_key=config.GEMINI_API_KEY)
        return _client


def _clean_name(name: str) -> str:
    return name.split("/", 1)[1] if name.startswith("models/") else name


def _model_names(client: genai.Client) -> Iterator[str]:
    """Yield model names to try: the known-good one, configured, fallbacks, discovered."""
    seen = set()

    def fresh(name: str) -> bool:
        if not name or name in seen:
            return False
        seen.add(name)
        return True

    candidates = [_working_model, config.GEMINI_MODEL, *_FALLBACK_MODELS]
    for name in candidates:
        if name and fresh(_clean_name(name)):
            yield _clean_name(name)

    # Last resort: ask the API which models this key can use.
    try:
        for m in client.models.list():
            name = _clean_name(getattr(m, "name", "") or "")
            actions = getattr(m, "supported_actions", None) or []
            lowered = name.lower()
            if (
                "generateContent" in actions
                and "flash" in lowered
                and not any(bad in lowered for bad in ("image", "tts", "live", "audio", "embedding"))
                and fresh(name)
            ):
                yield name
    except Exception as exc:  # discovery is best-effort
        log.warning("Could not list Gemini models: %s", exc)


def _translate(exc: errors.APIError) -> AIServiceError:
    code = getattr(exc, "code", None)
    detail = getattr(exc, "message", None) or str(exc)
    low = str(detail).lower()
    if code == 429:
        return AIServiceError(
            "Gemini rate limit / free-tier quota reached. Wait a minute and try again.", 429
        )
    if code in (401, 403) or "api key" in low:
        return AIServiceError(
            "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.", 503
        )
    return AIServiceError(f"Gemini API error ({code}): {detail}", 502)


def _extract_text(response) -> str:
    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()
    raise AIServiceError(
        "Gemini returned no text (the response may have been blocked by safety filters). "
        "Please rephrase your input and try again.",
        502,
    )


def generate_text(prompt: str, *, json_mode: bool = False, temperature: Optional[float] = None) -> str:
    """Send ``prompt`` to Gemini and return the response text."""
    global _working_model
    client = _get_client()

    cfg_kwargs = {}
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"
    if temperature is not None:
        cfg_kwargs["temperature"] = temperature
    cfg = types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None

    tried = []
    for model in _model_names(client):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=cfg)
        except errors.APIError as exc:
            if getattr(exc, "code", None) == 404:
                log.warning("Gemini model '%s' not available, trying next.", model)
                tried.append(model)
                continue
            raise _translate(exc) from exc
        except AIServiceError:
            raise
        except Exception as exc:
            raise AIServiceError(f"Could not reach Gemini: {exc}", 502) from exc

        if _working_model != model:
            _working_model = model
            log.info("Using Gemini model: %s", model)
        return _extract_text(response)

    raise AIServiceError(
        "No available Gemini model worked (tried: " + ", ".join(tried) + "). "
        "Set GEMINI_MODEL in .env to a model your key can use.",
        502,
    )


def current_model() -> str:
    return _working_model or config.GEMINI_MODEL
