"""Personalised learning-path recommendations (Gemini)."""
import gemini_client


def get_learning_recommendations(topic: str) -> str:
    prompt = (
        f"You are an AI tutor. The student wants to learn about: {topic}.\n"
        "Suggest a structured and adaptive learning path including key topics, the order of "
        "learning, an estimated timeline for each stage, and resources (links, videos, "
        "articles, or books).\n"
        "Organise it into Beginner, Intermediate, and Advanced levels, and finish with a "
        "small practice project idea. Use Markdown headings and bullet points."
    )
    return gemini_client.generate_text(prompt)
