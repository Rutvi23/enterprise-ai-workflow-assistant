"""Dependency container shared by the API and the MCP server."""
from __future__ import annotations

import httpx

from .config import Settings
from .llm import MockProvider, OpenAIProvider
from .retrieval import EmbeddingRetriever, TfidfRetriever
from .store import Store


class Services:
    def __init__(self, settings: Settings, llm=None, retriever=None, http_client_factory=None):
        from .agent import AgentService
        from .tools import build_registry

        self.settings = settings
        self.store = Store(settings.data_dir)
        self.llm = llm or self._make_llm(settings)
        self.retriever = retriever or self._make_retriever(settings)
        self.http_client_factory = http_client_factory or (lambda: httpx.Client())
        self.registry = build_registry(self)
        self.agent = AgentService(self)

    @staticmethod
    def _make_llm(s: Settings):
        if s.llm_provider == "openai":
            return OpenAIProvider(s.llm_base_url, s.llm_api_key, s.llm_model)
        if s.llm_provider == "mock":
            return MockProvider()
        raise ValueError(f"Unknown LLM_PROVIDER: {s.llm_provider}")

    @staticmethod
    def _make_retriever(s: Settings):
        if s.retriever == "embeddings":
            return EmbeddingRetriever(s.llm_base_url, s.llm_api_key, s.embedding_model, s.min_score)
        return TfidfRetriever(s.min_score)

    def search(self, query: str, k: int | None = None, doc_ids: list[str] | None = None) -> list[dict]:
        docs = self.store.snapshot()["documents"]
        hits = self.retriever.search(docs, query, k or self.settings.top_k, doc_ids)
        for h in hits:
            h["filename"] = docs[h["document_id"]]["filename"]
        return hits
