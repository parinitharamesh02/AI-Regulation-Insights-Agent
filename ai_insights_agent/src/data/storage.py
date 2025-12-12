import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from ..config import RAW_DIR, PROCESSED_DIR, REPORTS_DIR, CHAT_DIR
from ..models import Article, Chunk, Report
from ..logging_utils import get_logger

logger = get_logger(__name__)


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%S")


# -------- Articles --------

def save_articles(articles: Iterable[Article]) -> Path:
    articles = list(articles)
    if not articles:
        raise ValueError("No articles to save")

    path = RAW_DIR / f"articles_{_timestamp()}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for art in articles:
            f.write(json.dumps(art.dict(), default=str) + "\n")
    logger.info("Saved %d articles to %s", len(articles), path)
    return path


# -------- Chunks --------

def save_chunks(chunks: Iterable[Chunk]) -> Path:
    chunks = list(chunks)
    if not chunks:
        raise ValueError("No chunks to save")

    path = PROCESSED_DIR / f"chunks_{_timestamp()}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.dict(), default=str) + "\n")
    logger.info("Saved %d chunks to %s", len(chunks), path)
    return path


def load_latest_chunks() -> List[Chunk]:
    files = sorted(PROCESSED_DIR.glob("chunks_*.jsonl"))
    if not files:
        return []

    latest = files[-1]
    chunks: List[Chunk] = []
    with latest.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            chunks.append(Chunk(**data))
    logger.info("Loaded %d chunks from %s", len(chunks), latest)
    return chunks


# -------- Reports --------

def save_report(report: Report) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = report.created_at.strftime("%Y%m%dT%H%M%S")
    path = REPORTS_DIR / f"report_{ts}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(report.dict(), f, indent=2, default=str)

    logger.info("Saved report %s to %s", report.id, path)
    return path


def load_all_reports() -> List[Report]:
    if not REPORTS_DIR.exists():
        return []

    files = sorted(REPORTS_DIR.glob("report_*.json"))
    reports: List[Report] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append(Report(**data))

    reports.sort(key=lambda r: r.created_at)
    return reports


# -------- Chat history (simple persistence) --------

def save_chat_history(history: List[dict]) -> Path:
    """
    Persist the current chat history as a JSON file.
    Each entry is a dict with keys: question, answer, sources.
    """
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    path = CHAT_DIR / f"chat_{_timestamp()}.json"

    with path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, default=str)

    logger.info("Saved chat history (%d turns) to %s", len(history), path)
    return path


def load_latest_chat_history() -> List[dict]:
    """
    Load the most recent chat history, if any.
    Returns an empty list if nothing is stored yet.
    """
    if not CHAT_DIR.exists():
        return []

    files = sorted(CHAT_DIR.glob("chat_*.json"))
    if not files:
        return []

    latest = files[-1]
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        if isinstance(data, list):
            logger.info("Loaded chat history (%d turns) from %s", len(data), latest)
            return data
    except Exception as e:
        logger.warning("Failed to load chat history from %s: %s", latest, e)

    return []
