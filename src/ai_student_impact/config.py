import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = Path(r"D:\Sagar workspace\GenAI\DataSet\archive\ai_student_impact_dataset.csv")

load_dotenv(ROOT / ".env")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_dataset_path(override_path: str | None = None) -> Path:
    if override_path:
        return Path(override_path).expanduser().resolve()
    return DEFAULT_CSV_PATH
