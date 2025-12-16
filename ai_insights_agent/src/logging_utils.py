import logging
from typing import Optional


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Central logger factory.

    For production, this could be wired to JSON logging / CloudWatch.
    Here we keep it simple but consistent.
    """
    logger_name = name or "ai_regulation_agent"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )

    return logger
