import os
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables (.env must contain OPENAI_API_KEY)
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please add it to .env or export it.")

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
#  ChatCompletion Wrapper
# ---------------------------------------------------------------------------

def chat_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Thin wrapper around OpenAI's Chat Completions API (v1+).

    This design lets you swap providers (Anthropic, Azure OpenAI, etc.)
    with minimal changes to the rest of the system.
    """
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def format_system_user(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
    """Convenience helper to create a 2-message system + user conversation."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------------
#  Prompt Loading
# ---------------------------------------------------------------------------

from .config import TOPIC, BASE_DIR

PROMPTS_DIR = BASE_DIR / "src" / "prompts"


def load_prompt(name: str) -> str:
    """
    Load a prompt template from: src/prompts/{name}.txt
    and inject the configured {TOPIC}.
    """
    path: Path = PROMPTS_DIR / f"{name}.txt"

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    text = path.read_text(encoding="utf-8")
    return text.replace("{TOPIC}", TOPIC)
