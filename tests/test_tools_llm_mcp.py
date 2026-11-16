import json

import httpx
import pytest

from app.llm import OpenAIProvider
from app.mcp_server import handle_message
from app.services import Services
from app.tools import ToolError


def test_fetch_url_policy(svc):
    tool = svc.registry.get("fetch_url")
    with pytest.raises(ToolError, match="https"):
        svc.registry.validate(tool, {"url": "http://httpbin.org/get"})
    with pytest.raises(ToolError, match="allowlist"):
        svc.registry.validate(tool, {"url": "https://evil.example.com/x"})
    svc.registry.validate(tool, {"url": "https://httpbin.org/get"})


def test_fetch_url_runs_via_agent_after_approval(settings):
    transport = httpx.MockTransport(lambda r: httpx.Response(200, text="hello from api"))
    svc = Services(settings, http_client_factory=lambda: httpx.Client(transport=transport))
    run = svc.agent.start("Fetch https://httpbin.org/get please")
    assert run["status"] == "awaiting_approval"
    out = svc.agent.resolve(run["pending_approval"]["id"], True)
    assert out["run"]["status"] == "completed"
    assert any(s.get("result", {}).get("body") == "hello from api" for s in out["run"]["steps"] if s["kind"] == "tool_result")


def test_agent_reports_policy_violation_without_asking_for_approval(svc):
    run = svc.agent.start("Fetch https://evil.example.com/steal")
    assert run["status"] == "completed" and "allowlist" in run["final_answer"]


def test_tool_argument_validation(svc):
    send = svc.registry.get("send_email")
    with pytest.raises(ToolError, match="Missing"):
        svc.registry.validate(send, {"to": "a@b.co"})
    with pytest.raises(ToolError, match="Unexpected"):
        svc.registry.validate(send, {"to": "a@b.co", "subject": "s", "body": "b", "cc": "x"})
    with pytest.raises(ToolError, match="type"):
        svc.registry.validate(svc.registry.get("search_documents"), {"query": 5})


def test_openai_provider_parses_tool_calls_and_sends_tools():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content), auth=request.headers["authorization"])
        return httpx.Response(200, json={"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_documents", "arguments": "{\"query\": \"x\"}"}},
            {"id": "c2", "type": "function", "function": {"name": "list_documents", "arguments": "not json"}}]}}]})

    p = OpenAIProvider("http://llm/v1", "sk-test", "m", client=httpx.Client(transport=httpx.MockTransport(handler)))
    resp = p.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function", "function": {"name": "x"}}])
    assert resp.tool_calls[0].arguments == {"query": "x"}
    assert "__invalid_json__" in resp.tool_calls[1].arguments
    assert seen["auth"] == "Bearer sk-test" and seen["tool_choice"] == "auto"


def test_llm_failure_marks_run_failed(settings):
    class Boom:
        name = "boom"

        def chat(self, *a, **k):
            raise RuntimeError("model down")

    run = Services(settings, llm=Boom()).agent.start("do something")
    assert run["status"] == "failed" and "model down" in run["error"]


def test_step_limit_prevents_infinite_loops(settings):
    from app.llm import LLMResponse, ToolCall

    class Looper:
        name = "loop"

        def chat(self, *a, **k):
            return LLMResponse(tool_calls=[ToolCall("c", "list_documents", {})])

    run = Services(settings, llm=Looper()).agent.start("loop forever")
    assert run["status"] == "failed" and "Stopped after" in run["error"]


# ------------------------------------------------------------------- MCP
def rpc(svc, method, params=None, id=1):
    return handle_message(svc, {"jsonrpc": "2.0", "id": id, "method": method, "params": params or {}})


def test_mcp_handshake_and_tool_listing(svc):
    init = rpc(svc, "initialize", {"protocolVersion": "2024-11-05"})
    assert init["result"]["capabilities"] == {"tools": {}}
    assert handle_message(svc, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    names = {t["name"] for t in rpc(svc, "tools/list")["result"]["tools"]}
    assert {"search_documents", "send_email", "create_task"} <= names
    assert rpc(svc, "nope")["error"]["code"] == -32601


def test_mcp_safe_tool_executes(svc, vendor_pdf):
    from app.ingestion import chunk_pages, extract_pages
    pages = extract_pages("v.pdf", vendor_pdf)
    with svc.store.transaction() as d:
        d["documents"]["d"] = {"id": "d", "filename": "v.pdf", "pages": 2, "chunks": chunk_pages("d", pages, 800, 100)}
    res = rpc(svc, "tools/call", {"name": "search_documents", "arguments": {"query": "termination notice"}})["result"]
    assert res["isError"] is False and "v.pdf" in res["content"][0]["text"]


def test_mcp_gated_tool_is_not_executed_it_creates_pending_approval(svc, client):
    res = rpc(svc, "tools/call", {"name": "send_email",
                                  "arguments": {"to": "a@b.co", "subject": "s", "body": "b"}})["result"]
    assert "Approval required" in res["content"][0]["text"]
    assert svc.store.snapshot()["outbox"] == []
    pending = client.get("/api/approvals?status=pending").json()
    assert pending and pending[0]["tool"] == "send_email" and pending[0]["run_id"] is None
    # a human approves via the API → now it runs
    assert client.post(f"/api/approvals/{pending[0]['id']}/approve").status_code == 200
    assert len(svc.store.snapshot()["outbox"]) == 1


def test_mcp_invalid_arguments_reported_as_tool_error(svc):
    res = rpc(svc, "tools/call", {"name": "send_email", "arguments": {"to": "bad"}})["result"]
    assert res["isError"] is True
