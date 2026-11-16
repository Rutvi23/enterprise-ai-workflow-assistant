def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_list_delete_and_duplicate(client, vendor_pdf):
    r = client.post("/api/documents", files={"file": ("v.pdf", vendor_pdf, "application/pdf")})
    assert r.status_code == 201 and r.json()["pages"] == 2
    dup = client.post("/api/documents", files={"file": ("v2.pdf", vendor_pdf, "application/pdf")})
    assert dup.json()["duplicate"] is True
    assert len(client.get("/api/documents").json()) == 1
    assert client.delete(f"/api/documents/{r.json()['id']}").status_code == 204
    assert client.delete(f"/api/documents/{r.json()['id']}").status_code == 404


def test_upload_validation(client):
    assert client.post("/api/documents", files={"file": ("x.exe", b"x", "a/b")}).status_code == 415
    assert client.post("/api/documents", files={"file": ("x.pdf", b"junk", "a/b")}).status_code == 422


def test_ask_returns_grounded_answer_with_citations(loaded):
    r = loaded.post("/api/ask", json={"question": "What is the notice period to terminate for convenience?"})
    body = r.json()
    assert body["grounded"] is True
    assert "ninety" in body["answer"].lower() or "90" in body["answer"]
    assert body["sources"][0]["filename"] == "vendor_agreement.pdf"
    assert body["sources"][0]["page"] == 2
    assert "[1]" in body["answer"]


def test_ask_picks_the_right_document(loaded):
    body = loaded.post("/api/ask", json={"question": "How many sick leave days do employees get?"}).json()
    assert body["sources"][0]["filename"] == "leave_policy.pdf"


def test_ask_without_relevant_context_does_not_hallucinate(loaded):
    body = loaded.post("/api/ask", json={"question": "quantum chromodynamics lattice gauge"}).json()
    assert body["grounded"] is False and body["sources"] == []


def test_ask_with_no_documents(client):
    assert client.post("/api/ask", json={"question": "anything at all"}).json()["grounded"] is False


def test_api_key_enforced(tmp_path):
    from fastapi.testclient import TestClient

    from app.config import Settings
    from app.main import create_app

    c = TestClient(create_app(Settings(data_dir=str(tmp_path), api_key="secret", _env_file=None)))
    assert c.get("/api/documents").status_code == 401
    assert c.get("/api/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/api/documents", headers={"X-API-Key": "secret"}).status_code == 200
    assert c.get("/health").status_code == 200
