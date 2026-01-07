"""RAG chat module with externalized prompt."""

import anthropic

from .vectordb import DistanceMetric, VectorDB
from .embeddings import Embedder


# System prompt that sets the assistant's role and behavior
SYSTEM_PROMPT = """You are a helpful assistant for Contextual AI, a company building enterprise-focused language models using RAG (Retrieval Augmented Generation) technology.

Your knowledge comes from Contextual AI's blog posts about:
- The company's $20M seed funding round
- The founders' background (from Facebook AI Research, Hugging Face, Microsoft Research)
- Enterprise AI challenges (hallucination, data privacy, customizability, compliance, staleness, latency)
- RAG technology and its benefits for enterprise use cases
- The company's mission to build "AI that works for work"

Guidelines:
- Base your answers strictly on the provided context
- Cite sources using [1], [2], etc. when referencing specific passages
- Be concise but thorough
- If the context doesn't contain enough information, say so clearly
- Maintain a professional, knowledgeable tone"""

# User prompt template with context and question
RAG_PROMPT_TEMPLATE = """Here are relevant passages from Contextual AI's blog:

{context}

---

Based on the above context, please answer this question:
{question}"""


class RAGChat:
    """RAG (Retrieval Augmented Generation) chat service."""

    def __init__(
        self,
        client: anthropic.Anthropic,
        db: VectorDB,
        embedder: Embedder,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        system_prompt: str = SYSTEM_PROMPT,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
        normalize: bool = True,
    ):
        """Initialize RAG chat service.

        Args:
            client: Anthropic API client
            db: Vector database for retrieval
            embedder: Text embedding model
            model: Claude model to use
            max_tokens: Maximum response tokens
            system_prompt: System prompt setting assistant behavior
            prompt_template: Template with {context} and {question} placeholders
            normalize: Whether to normalize embeddings
        """
        self.client = client
        self.db = db
        self.embedder = embedder
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.prompt_template = prompt_template
        self.normalize = normalize

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve relevant documents for a query.

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        query_vector = self.embedder.encode(query, normalize=self.normalize)
        return self.db.search(query_vector, k=k, metric=DistanceMetric.COSINE)

    def build_context(self, results: list[dict]) -> str:
        """Build context string from retrieved documents.

        Args:
            results: Retrieved documents from vector search

        Returns:
            Formatted context string with numbered citations
        """
        context_parts = []
        for i, result in enumerate(results, 1):
            text = result["metadata"].get("text", "")
            score = result.get("score", 0)
            context_parts.append(f"[{i}] (relevance: {score:.2f})\n{text}")
        return "\n\n".join(context_parts)

    def generate(self, query: str, context: str) -> str:
        """Generate response using Claude.

        Args:
            query: User question
            context: Retrieved context

        Returns:
            Generated answer
        """
        user_message = self.prompt_template.format(context=context, question=query)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        return message.content[0].text

    def chat(self, query: str, k: int = 5) -> tuple[str, list[dict]]:
        """Full RAG pipeline: retrieve, build context, generate.

        Args:
            query: User question
            k: Number of documents to retrieve

        Returns:
            Tuple of (answer, sources)
        """
        results = self.retrieve(query, k=k)
        context = self.build_context(results)
        answer = self.generate(query, context)
        return answer, results
