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

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text into a vector.

        Args:
            text: Input text to encode

        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts into vectors.

        Args:
            texts: List of input texts

        Returns:
            2D numpy array of shape (n_texts, dimension)
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
