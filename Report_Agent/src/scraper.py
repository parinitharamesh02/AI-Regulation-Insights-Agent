import uuid
from datetime import datetime
from typing import List, Optional

import requests
from bs4 import BeautifulSoup

from .config import (
    BBC_ARTICLE_URLS,
    GOVUK_ARTICLE_URLS,
    BBC_SOURCE_NAME,
    GOVUK_SOURCE_NAME,
)
from .models import Article
from .logging_utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


def _fetch_html(url: str) -> str:
    """
    Fetch raw HTML from a URL with a browser-like User-Agent.

    In a production environment we would:
      - add retry/backoff
      - respect robots.txt
      - route through a trusted egress / proxy
    """
    resp = requests.get(url, timeout=20, headers=DEFAULT_HEADERS)
    if resp.status_code >= 400:
        # We log and let caller decide how to handle it
        logger.warning("HTTP %s for %s", resp.status_code, url)
        resp.raise_for_status()
    return resp.text


def _parse_datetime_maybe(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        # GOV.UK / BBC often use ISO 8601 with optional Z
        return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-source cleaners
# ---------------------------------------------------------------------------

def _clean_bbc_article(url: str, html: str) -> Article:
    """
    Extract title and main text from a BBC article.

    BBC layouts change often, so this uses conservative parsing:
      - <h1> for title
      - <article> wrapper if present, otherwise the whole document
    """
    soup = BeautifulSoup(html, "html.parser")

    # Title
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Published time (if available)
    published_at = None
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        published_at = _parse_datetime_maybe(time_tag["datetime"])

    # Main article content
    article_body = soup.find("article") or soup
    paragraphs = [
        p.get_text(" ", strip=True) for p in article_body.find_all("p")
    ]
    clean_text = "\n".join(p for p in paragraphs if p)

    return Article(
        id=str(uuid.uuid4()),
        source=BBC_SOURCE_NAME,
        url=url,
        title=title,
        published_at=published_at,
        raw_html=html,
        clean_text=clean_text,
    )


def _clean_govuk_article(url: str, html: str) -> Article:
    """
    Extract title and main text from a GOV.UK guidance / policy page.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else url

    published_at = None
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        published_at = _parse_datetime_maybe(time_tag["datetime"])

    main = (
        soup.find(class_="gem-c-govspeak")
        or soup.find("main")
        or soup
    )

    paragraphs = [
        p.get_text(" ", strip=True) for p in main.find_all("p")
    ]
    clean_text = "\n".join(p for p in paragraphs if p)

    return Article(
        id=str(uuid.uuid4()),
        source=GOVUK_SOURCE_NAME,
        url=url,
        title=title,
        published_at=published_at,
        raw_html=html,
        clean_text=clean_text,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def collect_articles() -> List[Article]:
    """
    Fetch and clean all configured source URLs.

    Currently:
      - GOV.UK policy/guidance pages on AI regulation and safety
      - BBC technology / AI article (if available; errors are logged & skipped)

    The design is deliberately extensible: new trusted sources can be
    added in config.py and wired here with dedicated cleaners, without
    changing downstream components.
    """
    articles: List[Article] = []

    # BBC
    for url in BBC_ARTICLE_URLS:
        try:
            logger.info("Fetching BBC article: %s", url)
            html = _fetch_html(url)
            art = _clean_bbc_article(url, html)
            if art.clean_text.strip():
                articles.append(art)
                logger.info("BBC OK: %s", art.title[:80])
            else:
                logger.warning("BBC empty text for %s", url)
        except Exception as e:
            logger.warning("Skipping BBC URL due to error: %s | %s", url, e)

    # GOV.UK
    for url in GOVUK_ARTICLE_URLS:
        try:
            logger.info("Fetching GOV.UK article: %s", url)
            html = _fetch_html(url)
            art = _clean_govuk_article(url, html)
            if art.clean_text.strip():
                articles.append(art)
                logger.info("GOV.UK OK: %s", art.title[:80])
            else:
                logger.warning("GOV.UK empty text for %s", url)
        except Exception as e:
            logger.warning("Skipping GOV.UK URL due to error: %s | %s", url, e)

    logger.info("Collected %d articles total.", len(articles))
    return articles
