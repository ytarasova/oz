"""Custom Vector Database implementation with flat index."""

from enum import Enum
from typing import Any

import numpy as np


class DistanceMetric(str, Enum):
    COSINE = "cosine"
    DOT_PRODUCT = "dot"
    EUCLIDEAN = "euclidean"


class VectorDB:
    """A simple flat-index vector database.

    Supports insert, search with configurable distance metrics.
    """

    def __init__(self, dimension: int | None = None):
        """Initialize the vector database.

        Args:
            dimension: Vector dimension. If None, inferred from first insert.
        """
        self.dimension = dimension
        self.vectors: np.ndarray | None = None
        self.ids: list[str] = []
        self.metadata: dict[str, dict[str, Any]] = {}

    def insert(self, id: str, vector: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        """Insert a vector with its ID and optional metadata.

        Args:
            id: Unique identifier for the vector
            vector: The embedding vector
            metadata: Optional metadata dict (e.g., {"text": "..."})
        """
        vector = np.asarray(vector, dtype=np.float32).flatten()

        if self.dimension is None:
            self.dimension = len(vector)
        elif len(vector) != self.dimension:
            raise ValueError(f"Vector dimension {len(vector)} != expected {self.dimension}")

        if self.vectors is None:
            self.vectors = vector.reshape(1, -1)
        else:
            self.vectors = np.vstack([self.vectors, vector])

        self.ids.append(id)
        self.metadata[id] = metadata or {}

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> list[dict[str, Any]]:
        """Search for the top-k most similar vectors.

        Args:
            query_vector: The query embedding
            k: Number of results to return
            metric: Distance metric to use

        Returns:
            List of dicts with id, score, and metadata
        """
        if self.vectors is None or len(self.ids) == 0:
            return []

        query = np.asarray(query_vector, dtype=np.float32).flatten()

        if len(query) != self.dimension:
            raise ValueError(f"Query dimension {len(query)} != database dimension {self.dimension}")

        scores = self._compute_scores(query, metric)

        # For euclidean, lower is better; for cosine/dot, higher is better
        if metric == DistanceMetric.EUCLIDEAN:
            top_indices = np.argsort(scores)[:k]
        else:
            top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            doc_id = self.ids[idx]
            results.append({
                "id": doc_id,
                "score": float(scores[idx]),
                "metadata": self.metadata.get(doc_id, {}),
            })

        return results

    def _compute_scores(self, query: np.ndarray, metric: DistanceMetric) -> np.ndarray:
        """Compute similarity/distance scores between query and all vectors."""
        if metric == DistanceMetric.COSINE:
            return self._cosine_similarity(query)
        elif metric == DistanceMetric.DOT_PRODUCT:
            return self._dot_product(query)
        elif metric == DistanceMetric.EUCLIDEAN:
            return self._euclidean_distance(query)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _cosine_similarity(self, query: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all vectors."""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return np.zeros(len(self.ids))

        vector_norms = np.linalg.norm(self.vectors, axis=1)
        # Avoid division by zero
        vector_norms = np.where(vector_norms == 0, 1, vector_norms)

        dot_products = self.vectors @ query
        return dot_products / (vector_norms * query_norm)

    def _dot_product(self, query: np.ndarray) -> np.ndarray:
        """Compute dot product between query and all vectors."""
        return self.vectors @ query

    def _euclidean_distance(self, query: np.ndarray) -> np.ndarray:
        """Compute euclidean distance between query and all vectors."""
        return np.linalg.norm(self.vectors - query, axis=1)

    def __len__(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.ids)
