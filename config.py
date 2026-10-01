"""Central configuration, loaded from environment variables / a local .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# --- Gemini (cloud) -------------------------------------------------------
GEMINI_API_KEY: str = _get("GEMINI_API_KEY") or _get("GOOGLE_API_KEY")
# "gemini-flash-latest" is an alias that always points at the current Flash model.
# Override in .env if you want a specific model (e.g. a Pro model).
GEMINI_MODEL: str = _get("GEMINI_MODEL", "gemini-flash-latest")

# --- Local explanation model (LaMini-Flan-T5) -------------------------------
# "local"  -> use LaMini-Flan-T5-783M on this machine (falls back to Gemini on failure)
# "gemini" -> skip the local model entirely (no torch/transformers needed)
EXPLAIN_BACKEND: str = _get("EXPLAIN_BACKEND", "local").lower()
LOCAL_MODEL_NAME: str = _get("LOCAL_MODEL_NAME", "MBZUAI/LaMini-Flan-T5-783M")

# --- Limits -----------------------------------------------------------------
try:
    MAX_INPUT_CHARS: int = int(_get("MAX_INPUT_CHARS", "12000"))
except ValueError:
    MAX_INPUT_CHARS = 12000
