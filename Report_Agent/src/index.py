from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME
from .models import Chunk

_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


class ChunkIndex:
    def __init__(self) -> None:
        self.index = None
        self.chunk_ids: List[str] = []
        self.chunks_by_id = {}

    def build(self, chunks: List[Chunk]) -> None:
        if not chunks:
            raise ValueError("No chunks provided to build index")

        texts = [c.text for c in chunks]
        embs = _embedder.encode(texts, convert_to_numpy=True)
        # Normalize for cosine similarity using inner product
        faiss.normalize_L2(embs)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

        self.chunk_ids = [c.id for c in chunks]
        self.chunks_by_id = {c.id: c for c in chunks}

    def query(self, question: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index not built")

        q_emb = _embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(D[0], I[0]):
            chunk_id = self.chunk_ids[int(idx)]
            results.append((self.chunks_by_id[chunk_id], float(score)))
        return results
