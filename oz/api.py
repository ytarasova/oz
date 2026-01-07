"""FastAPI backend for vector search and RAG chat."""

import json
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from .chat import RAGChat
from .embeddings import Embedder
from .vectordb import DistanceMetric, VectorDB


# Settings with environment variable support
class Settings(BaseSettings):
    anthropic_api_key: str = ""
    normalize_embeddings: bool = True

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Application state - initialized via lifespan
class AppState:
    """Application state container."""

    def __init__(self):
        self.db: VectorDB | None = None
        self.embedder: Embedder | None = None
        self.normalize: bool = True


# Global state instance - populated by lifespan
app_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup, cleanup on shutdown."""
    settings = get_settings()

    # Startup
    print("Loading embedder...")
    app_state.embedder = Embedder()
    app_state.normalize = settings.normalize_embeddings
    print(f"Normalization: {'enabled' if app_state.normalize else 'disabled'}")

    print("Initializing vector database...")
    app_state.db = VectorDB(dimension=app_state.embedder.dimension)

    # Load blog data
    blog_path = Path(__file__).parent.parent / "specs" / "blog.json"
    if blog_path.exists():
        print(f"Loading blog data from {blog_path}...")
        with open(blog_path) as f:
            blogs = json.load(f)

        # Extract texts and embed
        texts = [entry["metadata"]["text"] for entry in blogs]
        print(f"Embedding {len(texts)} blog entries...")
        vectors = app_state.embedder.encode_batch(texts, normalize=app_state.normalize)

        # Insert into database
        for entry, vector in zip(blogs, vectors):
            app_state.db.insert(
                id=entry["id"],
                vector=vector,
                metadata=entry["metadata"],
            )

        print(f"Loaded {len(app_state.db)} entries into vector database")
    else:
        print(f"Warning: {blog_path} not found")

    yield  # Application runs here

    # Shutdown (cleanup if needed)
    print("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Oz Vector Search",
    description="Vector search and RAG chat API",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependencies
def get_db() -> VectorDB:
    """Dependency to get the vector database."""
    if app_state.db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return app_state.db


def get_embedder() -> Embedder:
    """Dependency to get the embedder."""
    if app_state.embedder is None:
        raise HTTPException(status_code=500, detail="Embedder not initialized")
    return app_state.embedder


def get_normalize() -> bool:
    """Dependency to get the normalization setting."""
    return app_state.normalize


def get_anthropic_client(
    settings: Annotated[Settings, Depends(get_settings)]
) -> anthropic.Anthropic:
    """Dependency to get the Anthropic client."""
    if not settings.anthropic_api_key:
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY environment variable not set",
        )
    return anthropic.Anthropic(api_key=settings.anthropic_api_key)


def get_rag_chat(
    db: Annotated[VectorDB, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    client: Annotated[anthropic.Anthropic, Depends(get_anthropic_client)],
    normalize: Annotated[bool, Depends(get_normalize)],
) -> RAGChat:
    """Dependency to get the RAG chat service."""
    return RAGChat(client=client, db=db, embedder=embedder, normalize=normalize)


@app.get("/")
async def root():
    """Serve the frontend."""
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    return FileResponse(static_path)


@app.get("/search")
async def search(
    db: Annotated[VectorDB, Depends(get_db)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
    normalize: Annotated[bool, Depends(get_normalize)],
    query: str = Query(..., description="Search query text"),
    k: int = Query(5, ge=1, le=100, description="Number of results to return"),
    metric: DistanceMetric = Query(DistanceMetric.COSINE, description="Distance metric"),
):
    """Search for similar blog entries.

    Args:
        db: Vector database (injected)
        embedder: Embedding model (injected)
        normalize: Whether to normalize embeddings (injected from config)
        query: The search query text
        k: Number of results to return (default 5)
        metric: Distance metric (cosine, dot, euclidean)

    Returns:
        List of matching blog entries with scores
    """
    query_vector = embedder.encode(query, normalize=normalize)
    results = db.search(query_vector, k=k, metric=metric)

    return {
        "query": query,
        "k": k,
        "metric": metric.value,
        "results": results,
    }


class ChatRequest(BaseModel):
    query: str
    k: int = 5


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict]


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_chat: Annotated[RAGChat, Depends(get_rag_chat)],
):
    """RAG chat endpoint - retrieves context and generates response with Claude.

    Args:
        request: Chat request with query and optional k
        rag_chat: RAG chat service (injected)

    Returns:
        Generated answer with source documents
    """
    answer, sources = rag_chat.chat(request.query, k=request.k)

    return ChatResponse(
        query=request.query,
        answer=answer,
        sources=sources,
    )


# Mount static files (after routes so / doesn't conflict)
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
