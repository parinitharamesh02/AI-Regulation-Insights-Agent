import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Central logger factory.

    In a production deployment this could be wired to JSON logging and
    shipped to CloudWatch, Datadog, etc. For the challenge, a simple
    console logger is sufficient, but keeping this abstraction makes
    it easy to extend.
    """
    logger_name = name or "ai_report_agent"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        # Configure root logger only once
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    return logger
