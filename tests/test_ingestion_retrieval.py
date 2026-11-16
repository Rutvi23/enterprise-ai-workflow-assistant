import pytest

from app.ingestion import IngestionError, chunk_pages, extract_pages, split_text
from app.retrieval import EmbeddingRetriever, TfidfRetriever


def test_split_text_overlap_and_progress():
    text = ("Sentence number one is here. " * 60).strip()
    chunks = split_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)


def test_split_text_never_loops_forever_on_unbroken_text():
    assert len(split_text("x" * 5000, size=100, overlap=99)) > 1


def test_extract_pdf_pages_and_page_numbers(vendor_pdf):
    pages = extract_pages("a.pdf", vendor_pdf)
    assert len(pages) == 2
    chunks = chunk_pages("d1", pages, 800, 100)
    assert {c["page"] for c in chunks} == {1, 2}


def test_extract_rejects_bad_inputs():
    with pytest.raises(IngestionError):
        extract_pages("a.exe", b"x")
    with pytest.raises(IngestionError):
        extract_pages("a.pdf", b"not a pdf")
    with pytest.raises(IngestionError):
        extract_pages("a.txt", b"   ")


def _docs(vendor_pdf, leave_pdf):
    docs = {}
    for i, (name, data) in enumerate((("vendor.pdf", vendor_pdf), ("leave.pdf", leave_pdf))):
        pages = extract_pages(name, data)
        docs[f"d{i}"] = {"id": f"d{i}", "filename": name, "chunks": chunk_pages(f"d{i}", pages, 400, 50)}
    return docs


def test_tfidf_ranks_relevant_doc_first(vendor_pdf, leave_pdf):
    docs = _docs(vendor_pdf, leave_pdf)
    r = TfidfRetriever()
    top = r.search(docs, "how many days of sick leave do employees get", k=3)
    assert top and top[0]["document_id"] == "d1"
    top = r.search(docs, "termination notice period", k=3)
    assert top and top[0]["document_id"] == "d0"


def test_tfidf_returns_nothing_for_irrelevant_query(vendor_pdf, leave_pdf):
    assert TfidfRetriever().search(_docs(vendor_pdf, leave_pdf), "quantum chromodynamics lattice", k=3) == []


def test_tfidf_doc_filter(vendor_pdf, leave_pdf):
    docs = _docs(vendor_pdf, leave_pdf)
    hits = TfidfRetriever().search(docs, "payment leave termination", k=10, doc_ids=["d1"])
    assert hits and all(h["document_id"] == "d1" for h in hits)


def test_embedding_retriever_with_mock_endpoint(vendor_pdf, leave_pdf):
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        import json
        inputs = json.loads(request.content)["input"]
        # toy embedding: [mentions leave, mentions terminat]
        vecs = [{"embedding": [float("leave" in t.lower()), float("terminat" in t.lower()) + 0.01]} for t in inputs]
        return httpx.Response(200, json={"data": vecs})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    r = EmbeddingRetriever("http://x/v1", "k", "m", min_score=0.5, client=client)
    docs = _docs(vendor_pdf, leave_pdf)
    hits = r.search(docs, "termination", k=2)
    assert hits and hits[0]["document_id"] == "d0"
