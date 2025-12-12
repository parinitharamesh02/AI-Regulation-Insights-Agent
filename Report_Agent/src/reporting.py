import json
import uuid
from datetime import datetime
from typing import List

from .models import Report, Chunk
from .llm_client import chat_completion, load_prompt, format_system_user
from .config import TOPIC, EXAMPLES_DIR, REPORTS_DIR
from .logging_utils import get_logger

logger = get_logger(__name__)


def build_report_from_chunks(chunks: List[Chunk]) -> Report:
    """
    Use the report_prompt to generate a structured report from chunks.

    The report includes:
      - summary (100–150 words)
      - 3–5 key takeaways
      - extracted entities (organisations, people, locations, terms)
    """
    if not chunks:
        raise ValueError("No chunks provided for report generation.")

    logger.info("Building report from %d chunks", len(chunks))

    report_prompt = load_prompt("report_prompt")

    # Limit how many chunks we send to keep the prompt efficient
    MAX_CHUNKS = 12
    selected = chunks[:MAX_CHUNKS]

    context_parts = []
    for i, c in enumerate(selected):
        context_parts.append(
            f"[CHUNK {i} | article_id={c.article_id}]\n{c.text}\n"
        )
    context = "\n\n".join(context_parts)

    user_prompt = (
        f"You are generating a structured report on the topic: {TOPIC}.\n\n"
        f"Here are content chunks from recent articles:\n\n"
        f"{context}\n\n"
        "Follow the JSON output instructions exactly."
    )

    messages = format_system_user(
        system_prompt=report_prompt,
        user_prompt=user_prompt,
    )

    raw = chat_completion(messages)
    data = _extract_json(raw)

    report = Report(
        id=str(uuid.uuid4()),
        created_at=datetime.now(),
        topic=TOPIC,
        summary=data["summary"],
        takeaways=data["takeaways"],
        entities=data["entities"],
    )

    logger.info("Report generated: %s", report.id)
    return report


def _extract_json(raw: str) -> dict:
    """
    Try to parse a JSON object from model output.

    The prompt asks for pure JSON, but this is defensive against any
    extra text around it.
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


def save_report(report: Report) -> str:
    """
    Save a timestamped report for history and overwrite a 'latest' sample.

    - Historical reports: data/reports/report_YYYYMMDDThhmmss.json
    - Latest sample: examples/sample_report.json (for submission/demo)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    ts = report.created_at.strftime("%Y%m%dT%H%M%S")
    hist_path = REPORTS_DIR / f"report_{ts}.json"

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)

    sample_path = EXAMPLES_DIR / "sample_report.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, indent=2, default=str)

    logger.info("Report saved to %s and %s", hist_path, sample_path)
    return str(hist_path)


def main():
    """
    Entry point for a single reporting cycle.

    This simulates an hourly job that:
      - ingests the latest content
      - updates the knowledge base
      - generates a report
      - persists it for later trend analysis
    """
    from .scraper import collect_articles
    from .chunking import semantic_chunk

    logger.info("Starting reporting cycle...")
    articles = collect_articles()

    all_chunks: List[Chunk] = []
    for art in articles:
        article_chunks = semantic_chunk(art)
        logger.info(
            "Article '%s' produced %d chunks",
            art.title[:60],
            len(article_chunks),
        )
        all_chunks.extend(article_chunks)

    logger.info("Total chunks for report: %d", len(all_chunks))
    report = build_report_from_chunks(all_chunks)
    path = save_report(report)
    logger.info("Reporting cycle complete. Report path: %s", path)


if __name__ == "__main__":
    main()
