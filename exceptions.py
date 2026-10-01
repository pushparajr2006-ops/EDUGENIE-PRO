"""Custom exceptions shared by all EduGenie modules."""


class AIServiceError(Exception):
    """Raised when an AI backend (Gemini or the local model) cannot fulfil a request.

    ``status_code`` is the HTTP status the API layer should answer with.
    """

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
