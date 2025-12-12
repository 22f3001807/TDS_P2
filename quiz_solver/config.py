# quiz_solver/config.py

import os
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()


# ---------------------------------------------------------
# REQUIRED VALUES -- solver.py needs these to run
# ---------------------------------------------------------

STUDENT_EMAIL = os.environ.get("STUDENT_EMAIL")
QUIZ_SECRET = os.environ.get("QUIZ_SECRET")

if not STUDENT_EMAIL:
    raise ValueError(
        "STUDENT_EMAIL is missing.\n"
        "Create a .env file in project root:\n"
        "STUDENT_EMAIL=your-email@example.com"
    )

if not QUIZ_SECRET:
    raise ValueError(
        "QUIZ_SECRET is missing.\n"
        "Create a .env file in project root:\n"
        "QUIZ_SECRET=your-secret"
    )


# ---------------------------------------------------------
# OPTIONAL VALUES (used for timing, LLM, Playwright)
# ---------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# Model to use when calling OpenAI (change via .env if you prefer)
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

PLAYWRIGHT_TIMEOUT_MS = int(os.environ.get("PLAYWRIGHT_TIMEOUT_MS", "65000"))
TOTAL_QUIZ_WINDOW_SECONDS = int(os.environ.get("TOTAL_QUIZ_WINDOW_SECONDS", "180"))
PORT = int(os.environ.get("PORT", "8000"))


# ---------------------------------------------------------
# Helper to print environment status for debugging
# ---------------------------------------------------------
def env_info():
    return {
        "STUDENT_EMAIL_set": bool(STUDENT_EMAIL),
        "QUIZ_SECRET_set": bool(QUIZ_SECRET),
        "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
        "OPENAI_MODEL": OPENAI_MODEL,
        "PLAYWRIGHT_TIMEOUT_MS": PLAYWRIGHT_TIMEOUT_MS,
        "TOTAL_QUIZ_WINDOW_SECONDS": TOTAL_QUIZ_WINDOW_SECONDS,
        "PORT": PORT,
    }
