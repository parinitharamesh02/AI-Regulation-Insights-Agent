from pathlib import Path
from typing import List, Dict

from ..config import BASE_DIR, TOPIC
from ..logging_utils import get_logger

logger = get_logger(__name__)

PROMPTS_DIR = BASE_DIR / "src" / "llm" / "prompts"


def format_system_user(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_prompt(name: str) -> str:
    """
    Load a prompt template from src/llm/prompts/{name}.txt and inject {TOPIC}.
    """
    path: Path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    text = path.read_text(encoding="utf-8")
    return text.replace("{TOPIC}", TOPIC)
