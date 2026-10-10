"""Embeddings with multilingual-e5-small. e5 needs 'query: ' / 'passage: ' prefixes."""
import logging
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)

EMBED_MODEL = "intfloat/multilingual-e5-small"
BATCH_SIZE = 32


@lru_cache(maxsize=1)
def _model():
    """The model, on the GPU when there is one: a whole textbook is a thousand passages."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("loading the embedding model on %s", device)
    return SentenceTransformer(EMBED_MODEL, device=device)


def embed_passages(texts: list[str]) -> np.ndarray:
    vectors = _model().encode(
        [f"passage: {t}" for t in texts], normalize_embeddings=True, batch_size=BATCH_SIZE, show_progress_bar=False
    )
    release_gpu_cache()
    return vectors


def release_gpu_cache() -> None:
    """Return GPU memory that PyTorch is holding but not using.

    Indexing a textbook leaves over a gigabyte reserved. On a 6 GB card shared with the
    voice model that filled the GPU, and the voice became twice as slow.
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def embed_query(text: str) -> np.ndarray:
    return _model().encode(f"query: {text}", normalize_embeddings=True, show_progress_bar=False)
