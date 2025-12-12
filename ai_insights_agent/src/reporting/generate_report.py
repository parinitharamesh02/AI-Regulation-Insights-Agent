import json
import uuid
from datetime import datetime
from typing import List

from ..config import TOPIC
from ..logging_utils import get_logger
from ..models import Chunk, Report
from ..llm import chat_completion, load_prompt, format_system_user
from ..data.storage import save_report

logger = get_logger(__name__)


def _extract_json(raw: str) -> dict:
    """
    Parse JSON from model output; robust to extra text.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            return json.loads(snippet)
        raise


def build_report_from_chunks(chunks: List[Chunk]) -> Report:
    """
    Use the report_prompt to generate a structured report from chunks.
    """
    if not chunks:
        raise ValueError("No chunks provided for report generation")

    report_prompt = load_prompt("report")
    logger.info("Building report from %d chunks", len(chunks))

    # Limit how much context we send
    MAX_CHUNKS = 12
    selected = chunks[:MAX_CHUNKS]

    context_parts = []
    for i, c in enumerate(selected):
        context_parts.append(f"[CHUNK {i} | article_id={c.article_id}]\n{c.text}\n")
    context = "\n\n".join(context_parts)

    user_prompt = (
        f"You are generating a structured report on the topic: {TOPIC}.\n\n"
        f"Here are content chunks from recent articles:\n\n"
        f"{context}\n\n"
        "Follow the JSON output instructions exactly."
    )

    messages = format_system_user(report_prompt, user_prompt)
    raw = chat_completion(messages)
    data = _extract_json(raw)

    report = Report(
        id=str(uuid.uuid4()),
        created_at=datetime.utcnow(),
        topic=TOPIC,
        summary=data["summary"],
        takeaways=data["takeaways"],
        entities=data["entities"],
    )
    logger.info("Report generated: %s", report.id)
    return report


def generate_and_save_report(chunks: List[Chunk]) -> Report:
    report = build_report_from_chunks(chunks)
    save_report(report)
    return report
