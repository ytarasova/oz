"""Tests for the Embedder with normalization."""

import numpy as np
import pytest

from oz.embeddings import Embedder
from oz.vectordb import DistanceMetric, VectorDB


@pytest.fixture(scope="module")
def embedder():
    """Shared embedder instance for all tests."""
    return Embedder()


class TestEmbedderNormalization:
    """Tests for embedding normalization."""

    def test_encode_normalized_by_default(self, embedder):
        """Test that encode() returns normalized vector by default."""
        vec = embedder.encode("test sentence")
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-6)

    def test_encode_not_normalized_when_disabled(self, embedder):
        """Test that encode() returns raw vector when normalize=False."""
        vec = embedder.encode("test sentence", normalize=False)
        norm = np.linalg.norm(vec)
        # BGE-micro typically has norms > 1
        assert norm > 1.0

    def test_encode_batch_normalized_by_default(self, embedder):
        """Test that encode_batch() returns normalized vectors by default."""
        texts = ["first sentence", "second sentence", "third sentence"]
        vecs = embedder.encode_batch(texts)

        norms = np.linalg.norm(vecs, axis=1)
        assert all(np.isclose(n, 1.0, atol=1e-6) for n in norms)

    def test_encode_batch_not_normalized_when_disabled(self, embedder):
        """Test that encode_batch() returns raw vectors when normalize=False."""
        texts = ["first sentence", "second sentence"]
        vecs = embedder.encode_batch(texts, normalize=False)

        norms = np.linalg.norm(vecs, axis=1)
        assert all(n > 1.0 for n in norms)

    def test_normalized_vectors_have_correct_dimension(self, embedder):
        """Test that normalized vectors maintain correct dimension."""
        vec = embedder.encode("test")
        assert len(vec) == embedder.dimension


class TestMetricsWithNormalizedVectors:
    """Tests verifying all metrics produce same ranking with normalized vectors."""

    def test_all_metrics_same_ranking(self, embedder):
        """Test that cosine, dot product, and euclidean produce same ranking."""
        # Create test documents
        texts = [
            "Machine learning and artificial intelligence",
            "Deep neural networks for image recognition",
            "Natural language processing with transformers",
            "Database systems and SQL queries",
            "Web development with JavaScript",
        ]

        query = "AI and machine learning applications"

        # Embed with normalization (default)
        doc_vecs = embedder.encode_batch(texts)
        query_vec = embedder.encode(query)

        # Create DB and insert
        db = VectorDB(dimension=embedder.dimension)
        for i, (text, vec) in enumerate(zip(texts, doc_vecs)):
            db.insert(f"doc_{i}", vec, {"text": text})

        # Search with all three metrics
        results_cosine = db.search(query_vec, k=5, metric=DistanceMetric.COSINE)
        results_dot = db.search(query_vec, k=5, metric=DistanceMetric.DOT_PRODUCT)
        results_euclidean = db.search(query_vec, k=5, metric=DistanceMetric.EUCLIDEAN)

        # Extract ranking order
        order_cosine = [r["id"] for r in results_cosine]
        order_dot = [r["id"] for r in results_dot]
        order_euclidean = [r["id"] for r in results_euclidean]

        # All rankings should be identical for normalized vectors
        assert order_cosine == order_dot, "Cosine and Dot Product should have same ranking"
        assert order_cosine == order_euclidean, "Cosine and Euclidean should have same ranking"

    def test_cosine_equals_dot_for_normalized(self, embedder):
        """Test that cosine similarity equals dot product for unit vectors."""
        vec1 = embedder.encode("artificial intelligence")
        vec2 = embedder.encode("machine learning")

        # For unit vectors: cosine = dot product
        cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        dot = np.dot(vec1, vec2)

        assert np.isclose(cosine, dot, atol=1e-6)

    def test_euclidean_is_monotonic_transform_of_cosine(self, embedder):
        """Test that euclidean distance is sqrt(2 - 2*cosine) for unit vectors."""
        vec1 = embedder.encode("artificial intelligence")
        vec2 = embedder.encode("machine learning")

        cosine = np.dot(vec1, vec2)
        euclidean = np.linalg.norm(vec1 - vec2)
        expected_euclidean = np.sqrt(2 - 2 * cosine)

        assert np.isclose(euclidean, expected_euclidean, atol=1e-6)
