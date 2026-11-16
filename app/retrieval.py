"""Retrievers. Both return hits shaped like:
{chunk_id, document_id, page, text, score}
"""
from __future__ import annotations

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _flatten(docs: dict, doc_ids: list[str] | None):
    chunks = []
    for doc in docs.values():
        if doc_ids and doc["id"] not in doc_ids:
            continue
        chunks.extend(doc["chunks"])
    return chunks


def _hit(chunk: dict, score: float) -> dict:
    return {
        "chunk_id": chunk["id"],
        "document_id": chunk["document_id"],
        "page": chunk["page"],
        "text": chunk["text"],
        "score": round(float(score), 4),
    }


class TfidfRetriever:
    """Lexical retrieval: no API calls, fully offline."""

    name = "tfidf"

    def __init__(self, min_score: float = 0.05):
        self.min_score = min_score
        self._sig: tuple | None = None
        self._vec: TfidfVectorizer | None = None
        self._matrix = None

    def _build(self, chunks: list[dict]) -> None:
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            self._matrix = self._vec.fit_transform([c["text"] for c in chunks])
        except ValueError:  # empty vocabulary
            self._vec, self._matrix = None, None
        self._sig = tuple(c["id"] for c in chunks)

    def search(self, docs: dict, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        chunks = _flatten(docs, doc_ids)
        if not chunks or not query.strip():
            return []
        if self._sig != tuple(c["id"] for c in chunks):
            self._build(chunks)
        if self._vec is None:
            return []
        try:
            q = self._vec.transform([query])
        except ValueError:
            return []
        scores = (self._matrix @ q.T).toarray().ravel()  # vectors are L2-normalised → cosine
        order = np.argsort(-scores)[:k]
        return [_hit(chunks[i], scores[i]) for i in order if scores[i] >= self.min_score]


class EmbeddingRetriever:
    """Dense retrieval via an OpenAI-compatible /embeddings endpoint."""

    name = "embeddings"

    def __init__(self, base_url: str, api_key: str, model: str, min_score: float = 0.25,
                 client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.min_score = min_score
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=60)
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), 64):
            r = self.client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts[i : i + 64]},
                headers=self.headers,
            )
            r.raise_for_status()
            out.extend(np.array(d["embedding"], dtype=float) for d in r.json()["data"])
        return out

    def search(self, docs: dict, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[dict]:
        chunks = _flatten(docs, doc_ids)
        if not chunks or not query.strip():
            return []
        missing = [c for c in chunks if c["id"] not in self._cache]
        if missing:
            for c, vec in zip(missing, self._embed([c["text"] for c in missing])):
                self._cache[c["id"]] = vec / (np.linalg.norm(vec) or 1.0)
        qv = self._embed([query])[0]
        qv = qv / (np.linalg.norm(qv) or 1.0)
        scores = np.array([self._cache[c["id"]] @ qv for c in chunks])
        order = np.argsort(-scores)[:k]
        return [_hit(chunks[i], scores[i]) for i in order if scores[i] >= self.min_score]
