"""Tests for the VectorDB implementation."""

import numpy as np
import pytest

from oz.vectordb import DistanceMetric, VectorDB


class TestVectorDBBasic:
    """Basic functionality tests."""

    def test_insert_and_len(self):
        """Test inserting vectors and checking length."""
        db = VectorDB(dimension=3)
        db.insert("a", np.array([1.0, 0.0, 0.0]), {"text": "first"})
        db.insert("b", np.array([0.0, 1.0, 0.0]), {"text": "second"})

        assert len(db) == 2

    def test_dimension_inference(self):
        """Test that dimension is inferred from first insert."""
        db = VectorDB()
        db.insert("a", np.array([1.0, 2.0, 3.0, 4.0]))

        assert db.dimension == 4

    def test_dimension_mismatch_on_insert(self):
        """Test that mismatched dimensions raise error on insert."""
        db = VectorDB(dimension=3)
        db.insert("a", np.array([1.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="dimension"):
            db.insert("b", np.array([1.0, 0.0]))

    def test_dimension_mismatch_on_search(self):
        """Test that mismatched query dimensions raise error."""
        db = VectorDB(dimension=3)
        db.insert("a", np.array([1.0, 0.0, 0.0]))

        with pytest.raises(ValueError, match="dimension"):
            db.search(np.array([1.0, 0.0]), k=1)

    def test_insert_without_metadata(self):
        """Test inserting without metadata."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))

        results = db.search(np.array([1.0, 0.0]), k=1)
        assert results[0]["metadata"] == {}

    def test_insert_with_list(self):
        """Test inserting with Python list instead of numpy array."""
        db = VectorDB(dimension=3)
        db.insert("a", [1.0, 0.0, 0.0], {"text": "from list"})

        results = db.search([1.0, 0.0, 0.0], k=1)
        assert results[0]["id"] == "a"

    def test_metadata_preserved(self):
        """Test that metadata is preserved correctly."""
        db = VectorDB(dimension=2)
        metadata = {"text": "hello", "extra": 123, "nested": {"key": "value"}}
        db.insert("a", np.array([1.0, 0.0]), metadata)

        results = db.search(np.array([1.0, 0.0]), k=1)
        assert results[0]["metadata"] == metadata


class TestVectorDBSearch:
    """Search functionality tests."""

    def test_search_empty_db(self):
        """Test search on empty database."""
        db = VectorDB(dimension=3)
        results = db.search(np.array([1.0, 0.0, 0.0]), k=5)
        assert results == []

    def test_search_empty_db_no_dimension(self):
        """Test search on uninitialized database."""
        db = VectorDB()
        results = db.search(np.array([1.0, 0.0, 0.0]), k=5)
        assert results == []

    def test_search_k_larger_than_db(self):
        """Test search with k larger than database size."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))
        db.insert("b", np.array([0.0, 1.0]))

        results = db.search(np.array([1.0, 0.0]), k=10)
        assert len(results) == 2

    def test_search_k_equals_one(self):
        """Test search with k=1 returns single result."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))
        db.insert("b", np.array([0.0, 1.0]))

        results = db.search(np.array([1.0, 0.0]), k=1)
        assert len(results) == 1

    def test_result_structure(self):
        """Test that results have correct structure."""
        db = VectorDB(dimension=2)
        db.insert("test-id", np.array([1.0, 0.0]), {"key": "value"})

        results = db.search(np.array([1.0, 0.0]), k=1)

        assert len(results) == 1
        assert "id" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]
        assert results[0]["id"] == "test-id"
        assert isinstance(results[0]["score"], float)


class TestCosineSimlarity:
    """Cosine similarity specific tests."""

    def test_cosine_basic(self):
        """Test basic cosine similarity search."""
        db = VectorDB(dimension=3)
        db.insert("a", np.array([1.0, 0.0, 0.0]), {"text": "x-axis"})
        db.insert("b", np.array([0.0, 1.0, 0.0]), {"text": "y-axis"})
        db.insert("c", np.array([0.707, 0.707, 0.0]), {"text": "diagonal"})

        results = db.search(np.array([0.9, 0.1, 0.0]), k=2, metric=DistanceMetric.COSINE)

        assert len(results) == 2
        assert results[0]["id"] == "a"

    def test_cosine_identical_vectors(self):
        """Test cosine similarity with identical vectors returns 1.0."""
        db = VectorDB(dimension=3)
        db.insert("a", np.array([1.0, 2.0, 3.0]))

        results = db.search(np.array([1.0, 2.0, 3.0]), k=1, metric=DistanceMetric.COSINE)

        assert abs(results[0]["score"] - 1.0) < 1e-6

    def test_cosine_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors returns ~0."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))

        results = db.search(np.array([0.0, 1.0]), k=1, metric=DistanceMetric.COSINE)

        assert abs(results[0]["score"]) < 1e-6

    def test_cosine_opposite_vectors(self):
        """Test cosine similarity with opposite vectors returns -1.0."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))

        results = db.search(np.array([-1.0, 0.0]), k=1, metric=DistanceMetric.COSINE)

        assert abs(results[0]["score"] - (-1.0)) < 1e-6

    def test_cosine_zero_query_vector(self):
        """Test cosine similarity with zero query vector."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))

        results = db.search(np.array([0.0, 0.0]), k=1, metric=DistanceMetric.COSINE)

        # Zero vector should have 0 similarity
        assert results[0]["score"] == 0.0

    def test_cosine_scores_descending(self):
        """Test that cosine scores are in descending order."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))
        db.insert("b", np.array([0.7, 0.7]))
        db.insert("c", np.array([0.0, 1.0]))

        results = db.search(np.array([1.0, 0.0]), k=3, metric=DistanceMetric.COSINE)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestDotProduct:
    """Dot product specific tests."""

    def test_dot_product_basic(self):
        """Test basic dot product search."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1.0, 0.0]))
        db.insert("b", np.array([0.0, 1.0]))

        results = db.search(np.array([1.0, 0.0]), k=1, metric=DistanceMetric.DOT_PRODUCT)

        assert results[0]["id"] == "a"

    def test_dot_product_magnitude_matters(self):
        """Test that dot product considers magnitude."""
        db = VectorDB(dimension=2)
        db.insert("small", np.array([1.0, 0.0]))
        db.insert("large", np.array([10.0, 0.0]))

        results = db.search(np.array([1.0, 0.0]), k=2, metric=DistanceMetric.DOT_PRODUCT)

        # Larger magnitude vector should have higher dot product
        assert results[0]["id"] == "large"
        assert results[0]["score"] > results[1]["score"]

    def test_dot_product_scores_descending(self):
        """Test that dot product scores are in descending order."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([3.0, 0.0]))
        db.insert("b", np.array([2.0, 0.0]))
        db.insert("c", np.array([1.0, 0.0]))

        results = db.search(np.array([1.0, 0.0]), k=3, metric=DistanceMetric.DOT_PRODUCT)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestEuclideanDistance:
    """Euclidean distance specific tests."""

    def test_euclidean_basic(self):
        """Test basic euclidean distance search."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([0.0, 0.0]))
        db.insert("b", np.array([10.0, 10.0]))

        results = db.search(np.array([0.1, 0.1]), k=1, metric=DistanceMetric.EUCLIDEAN)

        assert results[0]["id"] == "a"

    def test_euclidean_identical_vectors(self):
        """Test euclidean distance with identical vectors returns 0."""
        db = VectorDB(dimension=3)
        db.insert("a", np.array([1.0, 2.0, 3.0]))

        results = db.search(np.array([1.0, 2.0, 3.0]), k=1, metric=DistanceMetric.EUCLIDEAN)

        assert abs(results[0]["score"]) < 1e-6

    def test_euclidean_scores_ascending(self):
        """Test that euclidean distances are in ascending order (lower is better)."""
        db = VectorDB(dimension=2)
        db.insert("near", np.array([1.0, 0.0]))
        db.insert("mid", np.array([5.0, 0.0]))
        db.insert("far", np.array([10.0, 0.0]))

        results = db.search(np.array([0.0, 0.0]), k=3, metric=DistanceMetric.EUCLIDEAN)

        scores = [r["score"] for r in results]
        assert scores == sorted(scores)
        assert results[0]["id"] == "near"
        assert results[2]["id"] == "far"

    def test_euclidean_known_distance(self):
        """Test euclidean distance with known value."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([3.0, 4.0]))

        results = db.search(np.array([0.0, 0.0]), k=1, metric=DistanceMetric.EUCLIDEAN)

        # Distance from origin to (3,4) is 5
        assert abs(results[0]["score"] - 5.0) < 1e-6


class TestEdgeCases:
    """Edge case tests."""

    def test_negative_values(self):
        """Test vectors with negative values."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([-1.0, -1.0]))
        db.insert("b", np.array([1.0, 1.0]))

        results = db.search(np.array([-1.0, -1.0]), k=1, metric=DistanceMetric.COSINE)

        assert results[0]["id"] == "a"

    def test_large_number_of_vectors(self):
        """Test with a larger number of vectors."""
        db = VectorDB(dimension=10)
        n_vectors = 100

        for i in range(n_vectors):
            vec = np.random.randn(10)
            db.insert(f"vec_{i}", vec)

        assert len(db) == n_vectors

        results = db.search(np.random.randn(10), k=10)
        assert len(results) == 10

    def test_high_dimensional_vectors(self):
        """Test with high dimensional vectors."""
        dim = 384  # Similar to BGE-micro
        db = VectorDB(dimension=dim)

        db.insert("a", np.random.randn(dim))
        db.insert("b", np.random.randn(dim))

        results = db.search(np.random.randn(dim), k=2)
        assert len(results) == 2

    def test_very_small_values(self):
        """Test with very small floating point values."""
        db = VectorDB(dimension=2)
        db.insert("a", np.array([1e-10, 1e-10]))

        results = db.search(np.array([1e-10, 1e-10]), k=1, metric=DistanceMetric.COSINE)

        assert len(results) == 1

    def test_mixed_positive_negative(self):
        """Test vectors with mixed positive and negative values."""
        db = VectorDB(dimension=4)
        db.insert("a", np.array([1.0, -1.0, 1.0, -1.0]))
        db.insert("b", np.array([-1.0, 1.0, -1.0, 1.0]))

        results = db.search(np.array([1.0, -1.0, 1.0, -1.0]), k=2, metric=DistanceMetric.COSINE)

        assert results[0]["id"] == "a"
        assert results[1]["id"] == "b"
