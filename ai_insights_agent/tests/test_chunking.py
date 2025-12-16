import sys
from pathlib import Path

# Make project root importable so we can "import src"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.processing.chunking import semantic_chunk
from src.models import Article
from datetime import datetime


def test_semantic_chunk_produces_non_empty_chunks():
    article = Article(
        id="a1",
        source="TEST",
        url="https://example.com",
        title="Test",
        published_at=None,
        raw_html="",
        clean_text=(
            "AI regulation focuses on safety and accountability. "
            "The UK government emphasises a pro-innovation approach to AI. "
            "Regulators coordinate through a hub-and-spoke model."
        ),
    )

    chunks = semantic_chunk(article, sim_threshold=0.5, max_sentences=3)
    assert len(chunks) >= 1
    assert all(c.text.strip() for c in chunks)
