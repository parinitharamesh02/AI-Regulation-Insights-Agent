import sys
from pathlib import Path

# Make project root importable so we can "import src"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datetime import datetime
from src.models import Chunk
from src.retrieval.index import ChunkIndex


def _chunk(id_, text):
    return Chunk(
        id=id_,
        article_id="a1",
        order=0,
        text=text,
        section=None,
        topic_label=None,
        created_at=datetime.utcnow(),
    )


def test_query_returns_relevant_chunk():
    chunks = [
        _chunk("c1", "The UK AI regulation white paper proposes a pro-innovation approach."),
        _chunk("c2", "Football transfer news and scores."),
    ]
    index = ChunkIndex()
    index.build(chunks)

    results = index.query("What is the UK AI regulation approach?", k=1)
    (best_chunk, score) = results[0]
    assert best_chunk.id == "c1"
    assert score > 0.0
