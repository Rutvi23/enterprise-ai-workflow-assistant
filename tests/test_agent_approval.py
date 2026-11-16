def _outbox(client):
    return client.get("/api/outbox").json()


def test_read_only_tool_runs_without_approval(loaded):
    run = loaded.post("/api/agent/runs", json={"instruction": "List documents"}).json()
    assert run["status"] == "completed"
    assert "vendor_agreement.pdf" in run["final_answer"]
    assert loaded.get("/api/approvals?status=pending").json() == []


def test_email_requires_approval_and_does_not_send_early(loaded):
    run = loaded.post("/api/agent/runs", json={
        "instruction": "Email alice@example.com a summary of the termination notice period"}).json()
    assert run["status"] == "awaiting_approval"
    assert run["pending_approval"]["tool"] == "send_email"
    assert run["pending_approval"]["risk"] == "high"
    assert _outbox(loaded)["emails"] == []          # nothing sent yet


def test_approve_executes_and_resumes(loaded):
    run = loaded.post("/api/agent/runs", json={
        "instruction": "Email alice@example.com a summary of the termination notice period"}).json()
    aid = run["pending_approval"]["id"]
    r = loaded.post(f"/api/approvals/{aid}/approve", json={"reviewer": "qa"}).json()
    assert r["run"]["status"] == "completed"
    emails = _outbox(loaded)["emails"]
    assert len(emails) == 1 and emails[0]["to"] == "alice@example.com"
    assert "vendor_agreement.pdf" in emails[0]["body"]   # grounded in retrieved passages
    assert r["approval"]["reviewer"] == "qa"


def test_reject_never_executes(loaded):
    run = loaded.post("/api/agent/runs", json={
        "instruction": "Email alice@example.com about the leave policy"}).json()
    aid = run["pending_approval"]["id"]
    r = loaded.post(f"/api/approvals/{aid}/reject", json={"note": "not now"}).json()
    assert r["run"]["status"] == "completed"
    assert "did not approve" in r["run"]["final_answer"]
    assert _outbox(loaded)["emails"] == []


def test_double_resolution_conflicts(loaded):
    run = loaded.post("/api/agent/runs", json={"instruction": "Create a task to review the contract by Friday"}).json()
    aid = run["pending_approval"]["id"]
    assert loaded.post(f"/api/approvals/{aid}/approve").status_code == 200
    assert loaded.post(f"/api/approvals/{aid}/approve").status_code == 409
    assert loaded.post("/api/approvals/nope/approve").status_code == 404


def test_reviewer_can_edit_arguments_before_approval(loaded):
    run = loaded.post("/api/agent/runs", json={"instruction": "Create a task to review the contract by Friday"}).json()
    appr = run["pending_approval"]
    assert appr["arguments"]["title"] == "review the contract" and appr["arguments"]["due"] == "Friday"
    edited = {"title": "Legal review of MSA", "due": "Monday"}
    loaded.post(f"/api/approvals/{appr['id']}/approve", json={"arguments": edited})
    task = _outbox(loaded)["tasks"][0]
    assert task["title"] == "Legal review of MSA" and task["due"] == "Monday"


def test_invalid_edited_arguments_rejected_and_not_executed(loaded):
    run = loaded.post("/api/agent/runs", json={
        "instruction": "Email alice@example.com about leave"}).json()
    aid = run["pending_approval"]["id"]
    bad = {"to": "not-an-email", "subject": "s", "body": "b"}
    assert loaded.post(f"/api/approvals/{aid}/approve", json={"arguments": bad}).status_code == 422
    assert _outbox(loaded)["emails"] == []
    assert loaded.get("/api/approvals?status=pending").json()[0]["id"] == aid   # still pending


def test_audit_trail_records_lifecycle(loaded):
    run = loaded.post("/api/agent/runs", json={"instruction": "Create a task to follow up"}).json()
    loaded.post(f"/api/approvals/{run['pending_approval']['id']}/approve")
    events = [e["event"] for e in loaded.get("/api/audit").json()]
    for expected in ("run_started", "approval_requested", "approval_approved", "run_completed"):
        assert expected in events


def test_state_survives_restart(settings, loaded):
    from fastapi.testclient import TestClient

    from app.main import create_app
    run = loaded.post("/api/agent/runs", json={"instruction": "Create a task to follow up"}).json()
    fresh = TestClient(create_app(settings))   # new process simulation: reloads from disk
    assert fresh.get(f"/api/agent/runs/{run['id']}").json()["status"] == "awaiting_approval"
    aid = run["pending_approval"]["id"]
    assert fresh.post(f"/api/approvals/{aid}/approve").json()["run"]["status"] == "completed"
