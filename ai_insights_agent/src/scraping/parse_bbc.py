import uuid

from bs4 import BeautifulSoup

from ..config import BBC_SOURCE_NAME
from ..models import Article
from ..logging_utils import get_logger
from .fetch import parse_iso_datetime_maybe, safe_get_text

logger = get_logger(__name__)


def parse_bbc_article(url: str, html: str) -> Article:
    """
    Extract title and main text from a BBC article.

    BBC layouts change fairly often, so this is deliberately conservative:
    - <h1> as title
    - <article> wrapper when present, otherwise the full document
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("h1")
    title = safe_get_text(title_tag) or url

    published_at = None
    time_tag = soup.find("time")
    if time_tag and time_tag.has_attr("datetime"):
        published_at = parse_iso_datetime_maybe(time_tag["datetime"])

    article_body = soup.find("article") or soup
    paragraphs = [safe_get_text(p) for p in article_body.find_all("p")]
    clean_text = "\n".join(p for p in paragraphs if p)

    logger.info("Parsed BBC article '%s' (%s)", title[:80], url)

    return Article(
        id=str(uuid.uuid4()),
        source=BBC_SOURCE_NAME,
        url=url,
        title=title,
        published_at=published_at,
        raw_html=html,
        clean_text=clean_text,
    )
