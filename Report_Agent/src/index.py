from typing import List, Tuple, Optional

import faiss
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME
from .models import Chunk

# Global embedder reused across the index
_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)


class ChunkIndex:
    """
    Simple FAISS-based vector index over semantic chunks.

    - Embeds chunk texts using a sentence-transformer.
    - Normalises embeddings to use inner-product as cosine similarity.
    - Provides top-k similarity search over chunks.
    """

    def __init__(self) -> None:
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[str] = []
        self.chunks_by_id: dict[str, Chunk] = {}

    def build(self, chunks: List[Chunk]) -> None:
        """
        Build the index from a list of Chunk objects.
        """
        if not chunks:
            raise ValueError("No chunks provided to build index")

        texts = [c.text for c in chunks]
        embeddings = _embedder.encode(texts, convert_to_numpy=True)

        # Normalise for cosine similarity with inner product
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self.chunk_ids = [c.id for c in chunks]
        self.chunks_by_id = {c.id: c for c in chunks}

    def query(self, question: str, k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Query the index with a natural language question and return top-k
        (Chunk, score) pairs.
        """
        if self.index is None:
            raise RuntimeError("Index not built")

        q_emb = _embedder.encode([question], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, k)

        results: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            chunk_id = self.chunk_ids[int(idx)]
            results.append((self.chunks_by_id[chunk_id], float(score)))

        return results
