from datetime import datetime
from typing import List
import uuid

import numpy as np
import nltk
from nltk.tokenize import sent_tokenize

from ..models import Article, Chunk
from ..logging_utils import get_logger
from .embeddings import embed_texts

logger = get_logger(__name__)

# Ensure sentence tokeniser is available
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def semantic_chunk(
    article: Article,
    sim_threshold: float = 0.75,
    max_sentences: int = 6,
    overlap: int = 1,
) -> List[Chunk]:
    """
    Embedding-based semantic chunking:
    - Split into sentences
    - Use sentence embeddings to group semantically similar sentences
    - Limit chunk size and add overlapping sentences for continuity
    """
    sentences = [s.strip() for s in sent_tokenize(article.clean_text) if s.strip()]
    if not sentences:
        return []

    sent_embs = embed_texts(sentences)
    chunks: List[Chunk] = []

    current_ids = [0]
    current_embs = [sent_embs[0]]

    def make_chunk(sent_ids: List[int], order: int) -> Chunk:
        text = " ".join(sentences[i] for i in sent_ids)
        return Chunk(
            id=str(uuid.uuid4()),
            article_id=article.id,
            order=order,
            text=text,
            section=None,
            topic_label=None,
            created_at=datetime.utcnow(),
        )

    order = 0
    for i in range(1, len(sentences)):
        centroid = np.mean(np.stack(current_embs, axis=0), axis=0)
        sim = _cosine_sim(centroid, sent_embs[i])

        if sim >= sim_threshold and len(current_ids) < max_sentences:
            current_ids.append(i)
            current_embs.append(sent_embs[i])
        else:
            chunks.append(make_chunk(current_ids, order))
            order += 1

            # start new chunk with overlap
            start = max(0, current_ids[-1] - overlap + 1)
            current_ids = list(range(start, i + 1))
            current_embs = [sent_embs[j] for j in current_ids]

    # Final chunk
    chunks.append(make_chunk(current_ids, order))
    logger.info(
        "Semantic chunking produced %d chunks for article '%s'",
        len(chunks),
        article.title[:80],
    )
    return chunks
