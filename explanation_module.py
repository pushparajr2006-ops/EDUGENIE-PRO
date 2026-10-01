"""Concept explanation.

Primary engine: the lightweight local model ``MBZUAI/LaMini-Flan-T5-783M`` (CPU friendly).
The model is loaded lazily on first use so the server starts instantly. If the local model
cannot be used (torch/transformers missing, no internet for the first download, ...) the
module transparently falls back to Gemini when an API key is configured.
"""
import logging
import threading
from typing import Tuple

import config
import gemini_client
from exceptions import AIServiceError

log = logging.getLogger("edugenie.explain")

_lock = threading.Lock()
_tokenizer = None
_model = None
_device = "cpu"
_local_unavailable = False  # set after a failed load so we don't retry on every request


def _load_local() -> None:
    global _tokenizer, _model, _device
    with _lock:
        if _model is not None:
            return
        import torch  # imported lazily: heavy
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        log.info("Loading local model %s (first run downloads ~3 GB) ...", config.LOCAL_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.LOCAL_MODEL_NAME)
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(_device).eval()
        _model = model
        log.info("Local model ready on %s.", _device)


def _prompt(topic: str) -> str:
    return f"Explain the concept of '{topic}' in a simple and clear way for a school student."


def _explain_local(topic: str) -> str:
    import torch

    _load_local()
    inputs = _tokenizer(_prompt(topic), return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_k=50,
            top_p=0.95,
            do_sample=True,
        )
    text = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError("Local model returned an empty explanation.")
    return text


def _explain_gemini(topic: str) -> str:
    return gemini_client.generate_text(
        _prompt(topic) + " Keep it short, friendly, and use one everyday example."
    )


def explain(topic: str) -> Tuple[str, str]:
    """Return ``(explanation, source)`` where source is ``"local"`` or ``"gemini"``."""
    global _local_unavailable
    if config.EXPLAIN_BACKEND != "gemini" and not _local_unavailable:
        try:
            return _explain_local(topic), "local"
        except Exception as exc:
            _local_unavailable = True
            log.warning("Local model unavailable (%s). Falling back to Gemini.", exc)
            if not config.GEMINI_API_KEY:
                raise AIServiceError(
                    f"The local explanation model could not be loaded ({exc}) and no "
                    "GEMINI_API_KEY is configured as a fallback. Install the requirements "
                    "(torch, transformers, sentencepiece) or set up a Gemini key.",
                    503,
                ) from exc
    return _explain_gemini(topic), "gemini"


def explain_topic(topic: str) -> str:
    """Convenience wrapper that returns only the explanation text."""
    return explain(topic)[0]
