import sys
from pathlib import Path
from typing import List

# Make project root importable so `src` works when running this script directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_utils import get_logger
from src.scraping import collect_articles
from src.processing.chunking import semantic_chunk
from src.models import Chunk
from src.data.storage import save_articles, save_chunks
from src.reporting.generate_report import generate_and_save_report

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting reporting cycle...")
    articles = collect_articles()
    if not articles:
        logger.warning("No articles collected; aborting.")
        return

    save_articles(articles)

    all_chunks: List[Chunk] = []
    for art in articles:
        chs = semantic_chunk(art)
        all_chunks.extend(chs)

    save_chunks(all_chunks)

    report = generate_and_save_report(all_chunks)
    logger.info("Reporting cycle complete. Report id=%s", report.id)


if __name__ == "__main__":
    main()
