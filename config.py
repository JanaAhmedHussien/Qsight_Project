from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

class Config:
    DB = f"sqlite:///{BASE_DIR / 'qsight.db'}"

    JSON = BASE_DIR / "json_outputs"
    REPORTS = BASE_DIR / "reports"

    JSON.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    # 🔐 API keys
    GEMINI = os.getenv("GEMINI_API_KEY")
    GROQ = os.getenv("GROQ_API_KEY")
    HF = os.getenv("HF_API_KEY")
    OPENAI = os.getenv("OPENAI_API_KEY")

