import sys
from pathlib import Path

# Make project root importable so `src` works when running this script directly
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logging_utils import get_logger
from src.reporting.trend_analysis import save_trend_analysis_to_examples

logger = get_logger(__name__)


def main() -> None:
    path = save_trend_analysis_to_examples()
    if path:
        logger.info("Trend analysis written to %s", path)
    else:
        logger.info("Not enough reports for trend analysis.")


if __name__ == "__main__":
    main()
