"""Embeddings module using BGE-micro from HuggingFace."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Text embedder using BGE-micro model."""

    MODEL_NAME = "TaylorAI/bge-micro-v2"

    def __init__(self):
        """Initialize the embedding model."""
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text into a vector.

        Args:
            text: Input text to encode
            normalize: If True, normalize to unit vector (default True)

        Returns:
            Embedding vector as numpy array
        """
        vec = self.model.encode(text, convert_to_numpy=True)
        if normalize:
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
        return vec

    def encode_batch(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """Encode multiple texts into vectors.

        Args:
            texts: List of input texts
            normalize: If True, normalize to unit vectors (default True)

        Returns:
            2D numpy array of shape (n_texts, dimension)
        """
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            vecs = vecs / norms
        return vecs
