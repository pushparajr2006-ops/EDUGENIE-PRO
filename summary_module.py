"""Summarisation (Gemini)."""
import gemini_client


def summarize_text(text: str) -> str:
    prompt = (
        "Summarize the following text in simple language. Keep the core information, "
        "remove redundancy, and use short sentences or a few bullet points suitable for "
        "quick revision.\n\n"
        f"{text}"
    )
    return gemini_client.generate_text(prompt)
