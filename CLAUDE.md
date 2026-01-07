# Oz - Vector Search System

A DIY vector database with RAG chat capabilities using Python and FastAPI.

## Commands

- `uv sync` - Install dependencies
- `uv run pytest` - Run tests with coverage
- `uv run python main.py` - Run the server (http://localhost:8000)
- `uv add <package>` - Add a runtime dependency
- `uv add --dev <package>` - Add a dev dependency
- `docker-compose up --build` - Run with Docker

## Project Structure

- `oz/` - Core package
  - `vectordb.py` - Custom flat-index vector database
  - `embeddings.py` - BGE-micro embedding model
  - `api.py` - FastAPI server with /search and /chat endpoints
- `static/index.html` - Frontend UI (search + chat tabs)
- `specs/blog.json` - Seed data (22 blog entries)
- `tests/` - Test files (pytest)
- `.env` - Environment variables (ANTHROPIC_API_KEY)

## API Endpoints

- `GET /` - Serve frontend
- `GET /search?query=...&k=5&metric=cosine` - Vector search
- `POST /chat` - RAG chat with Claude

## Distance Metrics

- `cosine` - Cosine similarity (default)
- `dot` - Dot product
- `euclidean` - Euclidean distance

## Code Style

- Python 3.11+
- Type hints for all functions
- Keep functions small and focused
- Write tests for new functionality
