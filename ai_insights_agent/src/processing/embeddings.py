from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import EMBEDDING_MODEL_NAME
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Load once per process
_model = SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Encode a list of texts into a 2D numpy array of embeddings.
    """
    if not texts:
        return np.zeros((0, _model.get_sentence_embedding_dimension()), dtype="float32")
    logger.info("Embedding %d texts using %s", len(texts), EMBEDDING_MODEL_NAME)
    emb = _model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return emb.astype("float32")
