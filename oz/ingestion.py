"""URL ingestion module for crawling and processing web content."""

import uuid
from typing import Any

from crawl4ai import AsyncWebCrawler

from .embeddings import Embedder
from .vectordb import VectorDB


async def crawl_url(url: str) -> tuple[str, str]:
    """Crawl a URL and extract clean markdown content.

    Args:
        url: The URL to crawl

    Returns:
        Tuple of (markdown_content, page_title)

    Raises:
        ValueError: If crawling fails or returns no content
    """
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)

        if not result.success:
            raise ValueError(f"Failed to crawl {url}: {result.error_message}")

        markdown = result.markdown or ""
        title = result.metadata.get("title", "") if result.metadata else ""

        if not markdown.strip():
            raise ValueError(f"No content extracted from {url}")

        return markdown, title


def chunk_by_paragraphs(text: str, min_length: int = 100) -> list[str]:
    """Split text into chunks by paragraphs.

    Args:
        text: The text to chunk
        min_length: Minimum chunk length to include (filters noise)

    Returns:
        List of paragraph chunks
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = text.split("\n\n")

    # Clean up and filter
    chunks = []
    for para in paragraphs:
        # Strip whitespace and normalize
        cleaned = para.strip()
        # Filter out short chunks (likely headers, navigation, etc.)
        if len(cleaned) >= min_length:
            chunks.append(cleaned)

    return chunks


async def ingest_url(
    url: str,
    db: VectorDB,
    embedder: Embedder,
    normalize: bool = True,
) -> dict[str, Any]:
    """Full URL ingestion pipeline.

    1. Crawl the URL
    2. Chunk the content
    3. Delete existing chunks from this URL
    4. Embed new chunks
    5. Store with metadata

    Args:
        url: URL to ingest
        db: Vector database instance
        embedder: Embedding model instance
        normalize: Whether to normalize embeddings

    Returns:
        Dict with url, title, chunks_added, chunks_deleted
    """
    # Step 1: Crawl
    markdown, title = await crawl_url(url)

    # Step 2: Chunk
    chunks = chunk_by_paragraphs(markdown)

    if not chunks:
        raise ValueError(f"No valid chunks extracted from {url}")

    # Step 3: Delete existing chunks from this URL
    deleted_count = db.delete_by_source(url)

    # Step 4: Embed chunks
    vectors = embedder.encode_batch(chunks, normalize=normalize)

    # Step 5: Store with metadata
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        chunk_id = str(uuid.uuid4())
        metadata = {
            "text": chunk,
            "source_url": url,
            "title": title,
            "chunk_index": i,
        }
        db.insert(chunk_id, vector, metadata)

    return {
        "url": url,
        "title": title or url,
        "chunks_added": len(chunks),
        "chunks_deleted": deleted_count,
    }
