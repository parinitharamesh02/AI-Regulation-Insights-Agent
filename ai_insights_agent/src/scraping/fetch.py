from typing import Optional
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from ..logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def fetch_html(url: str, timeout: int = 20) -> str:
    """
    Fetch HTML from a URL with basic headers.
    Raises for 4xx/5xx to make failures explicit.
    """
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
    if resp.status_code >= 400:
        logger.warning("HTTP %s when fetching %s", resp.status_code, url)
        resp.raise_for_status()
    return resp.text


def parse_iso_datetime_maybe(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


def safe_get_text(tag) -> str:
    if tag is None:
        return ""
    return tag.get_text(" ", strip=True)
