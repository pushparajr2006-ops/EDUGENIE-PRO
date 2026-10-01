"""Question answering (Gemini)."""
import gemini_client


def answer_question_with_gemini(question: str) -> str:
    prompt = (
        "You are EduGenie, a friendly and accurate tutor. Answer the student's question "
        "clearly and concisely in simple language. If helpful, add one short example.\n\n"
        f"Question: {question}"
    )
    return gemini_client.generate_text(prompt)
