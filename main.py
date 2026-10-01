"""EduGenie - FastAPI application entry point.

Run with:  uvicorn main:app --reload
"""
import logging

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import config
import gemini_client
from exceptions import AIServiceError
from explanation_module import explain
from learning_path import get_learning_recommendations
from qna import answer_question_with_gemini
from quiz_module import generate_quiz
from summary_module import summarize_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")

app = FastAPI(
    title="EduGenie",
    description="Gemini-powered learning assistant: Q&A, explanations, summaries, quizzes and learning paths.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=config.BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(config.BASE_DIR / "templates"))


# ----------------------------------------------------------------------------
# Request models & helpers
# ----------------------------------------------------------------------------
class TopicIn(BaseModel):
    topic: str = ""


class TextIn(BaseModel):
    text: str = ""


def _bad_request(message: str) -> JSONResponse:
    return JSONResponse(content={"error": message}, status_code=400)


def _check_length(value: str, label: str):
    if len(value) > config.MAX_INPUT_CHARS:
        return _bad_request(f"{label} is too long (max {config.MAX_INPUT_CHARS} characters).")
    return None


# ----------------------------------------------------------------------------
# Error handlers - the frontend always receives {"error": "..."}
# ----------------------------------------------------------------------------
@app.exception_handler(AIServiceError)
async def ai_error_handler(request: Request, exc: AIServiceError):
    return JSONResponse(content={"error": exc.message}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", []) if p not in ("query", "body"))
    return _bad_request(f"Invalid request{': ' + field if field else ''} - please check your input.")


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logging.getLogger("edugenie").exception("Unhandled error")
    return JSONResponse(content={"error": f"Unexpected server error: {exc}"}, status_code=500)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini_configured": bool(config.GEMINI_API_KEY),
        "gemini_model": gemini_client.current_model(),
        "explain_backend": config.EXPLAIN_BACKEND,
    }


# ----------------------------------------------------------------------------
# API routes
# NOTE: these are plain `def` endpoints on purpose - FastAPI runs them in a
# thread pool, so slow AI calls never block the event loop.
# ----------------------------------------------------------------------------

# Q&A - GET API using Gemini
@app.get("/qa")
def answer_question(question: str = Query(..., description="The question to ask")):
    question = question.strip()
    if not question:
        return _bad_request("Please provide a question.")
    if (err := _check_length(question, "Question")) is not None:
        return err
    return {"answer": answer_question_with_gemini(question)}


# Explanation - POST API using the local LaMini-Flan-T5 model (Gemini fallback)
@app.post("/explain")
@app.post("/explain/", include_in_schema=False)
def explain_api(payload: TopicIn):
    topic = payload.topic.strip()
    if not topic:
        return _bad_request("Please provide a topic.")
    if (err := _check_length(topic, "Topic")) is not None:
        return err
    explanation, source = explain(topic)
    return {"topic": topic, "explanation": explanation, "source": source}


# Summarization - POST API
@app.post("/summarize")
@app.post("/summarize/", include_in_schema=False)
def summarize_api(payload: TextIn):
    text = payload.text.strip()
    if not text:
        return _bad_request("Please provide text to summarize.")
    if (err := _check_length(text, "Text")) is not None:
        return err
    return {"summary": summarize_text(text)}


# Quiz generation - POST API
@app.post("/quiz")
@app.post("/quiz/", include_in_schema=False)
def quiz_api(payload: TextIn):
    text = payload.text.strip()
    if not text:
        return _bad_request("Please provide text or a topic for the quiz.")
    if (err := _check_length(text, "Text")) is not None:
        return err
    return {"quiz": generate_quiz(text)}


# Learning recommendations - GET API
@app.get("/learn/recommendations")
def learning_recommendation_api(topic: str = Query(..., description="Topic to learn")):
    topic = topic.strip()
    if not topic:
        return _bad_request("Please provide a topic.")
    if (err := _check_length(topic, "Topic")) is not None:
        return err
    return {"topic": topic, "recommendation": get_learning_recommendations(topic)}
