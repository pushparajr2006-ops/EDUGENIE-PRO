"""Quiz generation (Gemini) - returns 3 multiple-choice questions as a list of dicts."""
import json
import logging
import re
from typing import Any, List

import gemini_client
from exceptions import AIServiceError

log = logging.getLogger("edugenie.quiz")

_PROMPT = """You are a quiz generator.

From the following passage or topic, create 3 multiple-choice questions. Each question must include:
- A "question"
- A list of exactly 4 "options"
- A correct "answer" that must exactly match one of the options.

Format your output as **valid JSON only** (no commentary, no Markdown), like this:
[
  {
    "question": "What is ...?",
    "options": ["A", "B", "C", "D"],
    "answer": "A"
  }
]

Passage or topic:
{text}
"""


def clean_json_block(text: str) -> str:
    """Strip Markdown code fences (```json ... ```) and surrounding chatter."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    # Fall back to the outermost JSON array if there is still extra text around it.
    if not text.startswith(("[", "{")):
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end > start:
            text = text[start : end + 1]
    return text


def _validate(data: Any) -> List[dict]:
    if isinstance(data, dict):
        data = data.get("questions") or data.get("quiz") or []
    if not isinstance(data, list):
        raise ValueError("Quiz JSON is not a list.")

    quiz = []
    for item in data:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        options = [str(o).strip() for o in (item.get("options") or [])]
        answer = str(item.get("answer", "")).strip()
        if not question or len(options) != 4:
            continue
        if answer not in options:
            # tolerate "B" / "b" style answers or case differences
            if len(answer) == 1 and answer.upper() in "ABCD":
                answer = options["ABCD".index(answer.upper())]
            else:
                match = next((o for o in options if o.lower() == answer.lower()), None)
                if match is None:
                    continue
                answer = match
        quiz.append({"question": question, "options": options, "answer": answer})

    if not quiz:
        raise ValueError("No valid questions found in the model output.")
    return quiz[:3]


def generate_quiz(text: str) -> List[dict]:
    prompt = _PROMPT.replace("{text}", text)
    last_error = "unknown error"
    for attempt in (1, 2):
        raw = gemini_client.generate_text(prompt, json_mode=True, temperature=0.7)
        try:
            return _validate(json.loads(clean_json_block(raw)))
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            log.warning("Quiz parse failed (attempt %d): %s", attempt, exc)
    raise AIServiceError(
        f"Could not generate a valid quiz ({last_error}). Please try again or use a longer text.",
        502,
    )
