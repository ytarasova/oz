# Vector Search System Implementation Plan

## Overview
Build a DIY vector database with search API and frontend UI using the blog.json seed data.

## Components

### 1. Vector Database (`oz/vectordb.py`)
Custom flat-index implementation:
- `VectorDB` class with:
  - `insert(id, vector, metadata)` - add vectors
  - `search(query_vector, k, metric)` - return top-k results
- Distance metrics (numpy-based):
  - Cosine similarity
  - Dot product
  - Euclidean distance
- Store vectors in numpy array, metadata in dict

### 2. Embeddings (`oz/embeddings.py`)
- Use `TaylorAI/bge-micro-v2` from HuggingFace
- `encode(text) -> np.array` function
- Batch encoding for initial data load

### 3. RAG Chat (`oz/chat.py`)
- `RAGChat` class with externalized prompt template
- Methods: `retrieve()`, `build_context()`, `generate()`, `chat()`

### 4. Backend API (`oz/api.py`)
FastAPI server with proper dependency injection:
- `GET /search?query=...&k=5&metric=cosine`
- `POST /chat` - RAG chat with Claude
- Lifespan-based initialization
- Dependencies: `get_db`, `get_embedder`, `get_rag_chat`

### 5. Frontend (`static/index.html`)
Simple HTML/JS:
- Search input box with k and metric controls
- Results display area
- Chat tab with RAG interface

## Project Structure
```
oz/
├── oz/
│   ├── __init__.py
│   ├── vectordb.py      # Vector database
│   ├── embeddings.py    # BGE-micro encoder
│   ├── chat.py          # RAG chat service
│   └── api.py           # FastAPI server
├── static/
│   └── index.html       # Frontend UI
├── specs/
│   └── blog.json        # Seed data
├── tests/
│   ├── test_main.py
│   └── test_vectordb.py
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── main.py
```

## Dependencies
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `sentence-transformers` - For BGE-micro embeddings
- `numpy` - Vector operations
- `anthropic` - Claude API for RAG chat
- `python-dotenv` - Environment variable loading
- `pydantic-settings` - Settings management

## Implementation Progress
- [x] Add dependencies
- [x] Implement VectorDB class with distance metrics
- [x] Implement embeddings module
- [x] Build FastAPI backend with /search
- [x] Create frontend HTML/JS
- [x] Add tests (30 tests, 98% VectorDB coverage)
- [x] Add RAG chat endpoint
- [x] Dockerize application
- [x] Add dotenv support
- [x] Refactor to FastAPI dependency injection (Depends)
- [x] Use FastAPI lifespan for initialization
- [x] Extract chat logic into separate module
- [x] Engineer proper RAG prompt with system prompt and citations
- [x] Improve VectorDB test coverage with edge cases

## Running the App

### Setup
```bash
# Copy .env.example and add your Anthropic API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

### Local Development
```bash
# Run the server
uv run python main.py

# Open http://localhost:8000
```

### Docker
```bash
# Build and run with docker-compose (reads from .env)
docker-compose up --build

# Open http://localhost:8000
```

### Running Tests
```bash
uv run pytest
```
