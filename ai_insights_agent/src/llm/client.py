import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from ..config import OPENAI_MODEL
from ..logging_utils import get_logger

logger = get_logger(__name__)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please add it to .env or export it.")

client = OpenAI(api_key=api_key)


def chat_completion(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL,
    temperature: float = 0.2,
) -> str:
    """
    Thin wrapper around OpenAI's chat completions API.

    Keeps the rest of the system decoupled from a specific provider.
    """
    logger.info("Calling OpenAI model=%s, messages=%d", model, len(messages))
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    return content.strip()
