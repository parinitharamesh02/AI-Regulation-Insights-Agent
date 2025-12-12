from typing import List, Tuple, Optional

import faiss
import numpy as np

from ..models import Chunk
from ..logging_utils import get_logger
from ..processing.embeddings import embed_texts

logger = get_logger(__name__)


class ChunkIndex:
    """
    FAISS-based index over semantic chunks.

    - Normalises embeddings to use inner product as cosine similarity.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[str] = []
        self.chunks_by_id: dict[str, Chunk] = {}

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided to build index")

        texts = [c.text for c in chunks]
        emb = embed_texts(texts)
        if emb.shape[0] == 0:
            raise ValueError("Failed to compute embeddings")

        faiss.normalize_L2(emb)
        dim = emb.shape[1]

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)

        self.chunk_ids = [c.id for c in chunks]
        self.chunks_by_id = {c.id: c for c in chunks}

        logger.info("Index built with %d chunks (dim=%d)", len(chunks), dim)

    def query(self, question: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not built")

        q_emb = embed_texts([question])
        faiss.normalize_L2(q_emb)
        scores, indices = self.index.search(q_emb, k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunk_ids):
                continue
            chunk_id = self.chunk_ids[int(idx)]
            results.append((self.chunks_by_id[chunk_id], float(score)))

        logger.info("Index query returned %d chunks", len(results))
        return results
