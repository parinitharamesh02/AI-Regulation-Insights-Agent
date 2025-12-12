from typing import List

from ..config import (
    BBC_ARTICLE_URLS,
    GOVUK_ARTICLE_URLS,
)
from ..models import Article
from ..logging_utils import get_logger
from .fetch import fetch_html
from .parse_bbc import parse_bbc_article
from .parse_govuk import parse_govuk_article

logger = get_logger(__name__)


def collect_articles() -> List[Article]:
    """
    Fetch and parse all configured sources into Article models.

    Currently:
    - GOV.UK AI regulation and safety policy documents
    - Optional BBC articles (if URLs configured)
    """
    articles: List[Article] = []

    # BBC
    for url in BBC_ARTICLE_URLS:
        try:
            logger.info("Fetching BBC article: %s", url)
            html = fetch_html(url)
            art = parse_bbc_article(url, html)
            if art.clean_text.strip():
                articles.append(art)
            else:
                logger.warning("BBC article had empty text: %s", url)
        except Exception as e:
            logger.warning("Skipping BBC URL due to error: %s | %s", url, e)

    # GOV.UK
    for url in GOVUK_ARTICLE_URLS:
        try:
            logger.info("Fetching GOV.UK article: %s", url)
            html = fetch_html(url)
            art = parse_govuk_article(url, html)
            if art.clean_text.strip():
                articles.append(art)
            else:
                logger.warning("GOV.UK article had empty text: %s", url)
        except Exception as e:
            logger.warning("Skipping GOV.UK URL due to error: %s | %s", url, e)

    logger.info("Collected %d articles in total", len(articles))
    return articles
