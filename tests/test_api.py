"""Offline tests: Gemini and the local model are mocked, so no API key or download is needed.

Run with:  pytest
"""
import json

import pytest
from fastapi.testclient import TestClient
from google.genai import errors

import config
import explanation_module
import gemini_client
import main
from exceptions import AIServiceError

client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    """Isolate every test from real keys / cached state."""
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "gemini")
    monkeypatch.setattr(explanation_module, "_local_unavailable", False)
    monkeypatch.setattr(gemini_client, "_working_model", None)
    monkeypatch.setattr(gemini_client, "_client", None)


def fake_gemini(monkeypatch, reply):
    """Replace the Gemini call with a function returning ``reply`` (str or callable)."""
    calls = []

    def _fake(prompt, **kwargs):
        calls.append((prompt, kwargs))
        return reply(prompt, **kwargs) if callable(reply) else reply

    monkeypatch.setattr(gemini_client, "generate_text", _fake)
    return calls


# ---------------------------------------------------------------- pages
def test_home_page_renders():
    r = client.get("/")
    assert r.status_code == 200
    assert "Welcome to EduGenie" in r.text
    assert "/static/app.js" in r.text


def test_static_files_served():
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["gemini_configured"] is True


# ----------------------------------------------------------------- Q&A
def test_qa_success(monkeypatch):
    calls = fake_gemini(monkeypatch, "The Pacific Ocean.")
    r = client.get("/qa", params={"question": "Which is the largest ocean?"})
    assert r.status_code == 200
    assert r.json() == {"answer": "The Pacific Ocean."}
    assert "largest ocean" in calls[0][0]


def test_qa_missing_question_is_400():
    assert client.get("/qa").status_code == 400
    assert client.get("/qa", params={"question": "   "}).status_code == 400


def test_qa_too_long_is_400(monkeypatch):
    monkeypatch.setattr(config, "MAX_INPUT_CHARS", 10)
    assert client.get("/qa", params={"question": "x" * 50}).status_code == 400


# ------------------------------------------------------------- explain
def test_explain_via_gemini_backend(monkeypatch):
    fake_gemini(monkeypatch, "Plants make food from sunlight.")
    r = client.post("/explain/", json={"topic": "Photosynthesis"})
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] == "Photosynthesis"
    assert body["explanation"].startswith("Plants")
    assert body["source"] == "gemini"


def test_explain_without_trailing_slash(monkeypatch):
    fake_gemini(monkeypatch, "ok")
    assert client.post("/explain", json={"topic": "Gravity"}).status_code == 200


def test_explain_local_model_used(monkeypatch):
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "local")
    monkeypatch.setattr(explanation_module, "_explain_local", lambda t: f"local: {t}")
    r = client.post("/explain", json={"topic": "Atoms"})
    assert r.json() == {"topic": "Atoms", "explanation": "local: Atoms", "source": "local"}


def test_explain_falls_back_to_gemini_when_local_fails(monkeypatch):
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "local")

    def boom(topic):
        raise RuntimeError("no torch")

    monkeypatch.setattr(explanation_module, "_explain_local", boom)
    fake_gemini(monkeypatch, "From Gemini")
    r = client.post("/explain", json={"topic": "Atoms"})
    assert r.status_code == 200
    assert r.json()["source"] == "gemini"


def test_explain_local_fails_and_no_key_gives_503(monkeypatch):
    monkeypatch.setattr(config, "EXPLAIN_BACKEND", "local")
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    monkeypatch.setattr(explanation_module, "_explain_local", lambda t: (_ for _ in ()).throw(RuntimeError("x")))
    assert client.post("/explain", json={"topic": "Atoms"}).status_code == 503


def test_explain_requires_topic():
    r = client.post("/explain", json={"topic": ""})
    assert r.status_code == 400
    assert "topic" in r.json()["error"].lower()
    assert client.post("/explain", content="not json", headers={"Content-Type": "application/json"}).status_code == 400


# ----------------------------------------------------------- summarize
def test_summarize(monkeypatch):
    fake_gemini(monkeypatch, "Short version.")
    r = client.post("/summarize/", json={"text": "A very long paragraph..."})
    assert r.json() == {"summary": "Short version."}


def test_summarize_requires_text():
    assert client.post("/summarize", json={"text": ""}).status_code == 400


# ---------------------------------------------------------------- quiz
QUIZ = [
    {"question": f"Q{i}?", "options": ["a", "b", "c", "d"], "answer": "b"} for i in range(1, 4)
]


def test_quiz_success(monkeypatch):
    fake_gemini(monkeypatch, json.dumps(QUIZ))
    r = client.post("/quiz", json={"text": "Solar System"})
    assert r.status_code == 200
    assert r.json()["quiz"] == QUIZ


def test_quiz_strips_markdown_fences(monkeypatch):
    fake_gemini(monkeypatch, "```json\n" + json.dumps(QUIZ) + "\n```")
    assert client.post("/quiz", json={"text": "x"}).json()["quiz"] == QUIZ


def test_quiz_accepts_letter_answers(monkeypatch):
    data = [{"question": "Q?", "options": ["w", "x", "y", "z"], "answer": "C"}]
    fake_gemini(monkeypatch, json.dumps(data))
    assert client.post("/quiz", json={"text": "x"}).json()["quiz"][0]["answer"] == "y"


def test_quiz_retries_once_then_succeeds(monkeypatch):
    replies = iter(["this is not json", json.dumps(QUIZ)])
    calls = fake_gemini(monkeypatch, lambda p, **k: next(replies))
    r = client.post("/quiz", json={"text": "x"})
    assert r.status_code == 200
    assert len(calls) == 2


def test_quiz_gives_502_when_model_keeps_failing(monkeypatch):
    fake_gemini(monkeypatch, "garbage")
    r = client.post("/quiz", json={"text": "x"})
    assert r.status_code == 502
    assert "quiz" in r.json()["error"].lower()


def test_quiz_requires_text():
    assert client.post("/quiz", json={"text": " "}).status_code == 400


# ------------------------------------------------------ learning path
def test_learning_recommendations(monkeypatch):
    fake_gemini(monkeypatch, "## Beginner\n- SELECT basics")
    r = client.get("/learn/recommendations", params={"topic": "SQL"})
    assert r.status_code == 200
    assert r.json()["topic"] == "SQL"
    assert "Beginner" in r.json()["recommendation"]


def test_learning_requires_topic():
    assert client.get("/learn/recommendations").status_code == 400


# ------------------------------------------- error handling & Gemini client
def test_missing_api_key_gives_helpful_503(monkeypatch):
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    r = client.get("/qa", params={"question": "hi"})
    assert r.status_code == 503
    assert "GEMINI_API_KEY" in r.json()["error"]


def test_ai_service_error_is_returned_as_json(monkeypatch):
    def raiser(prompt, **kw):
        raise AIServiceError("quota hit", 429)

    monkeypatch.setattr(gemini_client, "generate_text", raiser)
    r = client.get("/qa", params={"question": "hi"})
    assert r.status_code == 429
    assert r.json() == {"error": "quota hit"}


class _FakeResponse:
    def __init__(self, text):
        self.text = text


class _FakeModels:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.tried = []

    def generate_content(self, model, contents, config=None):
        self.tried.append(model)
        return self.behaviour(model)

    def list(self):
        return []


class _FakeClient:
    def __init__(self, behaviour):
        self.models = _FakeModels(behaviour)


def test_model_fallback_on_404(monkeypatch):
    def behaviour(model):
        if model == "retired-model":
            raise errors.ClientError(404, {"error": {"message": "model not found", "status": "NOT_FOUND"}})
        return _FakeResponse("  hello  ")

    fake = _FakeClient(behaviour)
    monkeypatch.setattr(config, "GEMINI_MODEL", "retired-model")
    monkeypatch.setattr(gemini_client, "_client", fake)

    assert gemini_client.generate_text("hi") == "hello"
    assert fake.models.tried[0] == "retired-model"
    assert len(fake.models.tried) == 2
    assert gemini_client.current_model() == fake.models.tried[1]


def test_quota_error_is_translated(monkeypatch):
    def behaviour(model):
        raise errors.ClientError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}})

    monkeypatch.setattr(gemini_client, "_client", _FakeClient(behaviour))
    with pytest.raises(AIServiceError) as exc:
        gemini_client.generate_text("hi")
    assert exc.value.status_code == 429


def test_empty_response_is_reported(monkeypatch):
    monkeypatch.setattr(gemini_client, "_client", _FakeClient(lambda m: _FakeResponse(None)))
    with pytest.raises(AIServiceError):
        gemini_client.generate_text("hi")
