"""Tool registry. Each tool declares a JSON schema, a risk level, and whether it
needs human approval. The agent can never bypass `requires_approval`: the policy
is enforced in AgentService, not in the model prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .store import now

EMAIL_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")
_TYPES = {"string": str, "integer": int}


class ToolError(Exception):
    """Raised for invalid arguments or policy violations; reported back to the agent."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    handler: Callable[[dict], Any]
    requires_approval: bool = False
    risk: str = "low"
    precheck: Callable[[dict], None] | None = None

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    @staticmethod
    def validate(tool: Tool, args: dict) -> None:
        props = tool.parameters.get("properties", {})
        extra = set(args) - set(props)
        if extra:
            raise ToolError(f"Unexpected arguments: {sorted(extra)}")
        for req in tool.parameters.get("required", []):
            if args.get(req) in (None, ""):
                raise ToolError(f"Missing required argument: {req}")
        for key, value in args.items():
            expected = _TYPES.get(props[key].get("type"))
            if expected and not isinstance(value, expected):
                raise ToolError(f"Argument '{key}' must be of type {props[key]['type']}")
        if tool.precheck:
            tool.precheck(args)

    def execute(self, name: str, args: dict) -> Any:
        tool = self.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        self.validate(tool, args)
        return tool.handler(args)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]


def build_registry(svc) -> ToolRegistry:
    """`svc` is the Services container (store, settings, search, http client factory)."""
    reg = ToolRegistry()

    def search_documents(a):
        hits = svc.search(a["query"], k=int(a.get("top_k", svc.settings.top_k)))
        return {"results": [
            {"document_id": h["document_id"], "filename": h["filename"], "page": h["page"],
             "score": h["score"], "text": h["text"]} for h in hits]}

    def list_documents(_a):
        docs = svc.store.snapshot()["documents"].values()
        return {"documents": [
            {"id": d["id"], "filename": d["filename"], "pages": d["pages"]} for d in docs]}

    def summarize_document(a):
        doc = svc.store.snapshot()["documents"].get(a["document_id"])
        if not doc:
            raise ToolError("Document not found")
        text = " ".join(c["text"] for c in doc["chunks"])
        sents = _sentences(text)
        freq: dict[str, int] = {}
        for w in re.findall(r"[a-z]{3,}", text.lower()):
            if w not in ENGLISH_STOP_WORDS:
                freq[w] = freq.get(w, 0) + 1
        scored = sorted(
            ((sum(freq.get(w, 0) for w in re.findall(r"[a-z]{3,}", s.lower())) / (len(s) ** 0.5), i)
             for i, s in enumerate(sents)), reverse=True)
        keep = sorted(i for _, i in scored[: int(a.get("sentences", 3))])
        return {"document_id": doc["id"], "filename": doc["filename"], "summary": " ".join(sents[i] for i in keep)}

    def create_task(a):
        task = {"id": f"T-{len(svc.store.snapshot()['tasks']) + 1:03d}", "title": a["title"],
                "due": a.get("due"), "assignee": a.get("assignee"), "created_at": now()}
        with svc.store.transaction() as d:
            d["tasks"].append(task)
        return task

    def _check_email(a):
        if not EMAIL_RE.match(a["to"]):
            raise ToolError("Invalid recipient email address")

    def send_email(a):
        # Simulated delivery: writes to an outbox. Swap this handler for SMTP / an email API.
        mail = {"to": a["to"], "subject": a["subject"], "body": a["body"], "sent_at": now()}
        with svc.store.transaction() as d:
            d["outbox"].append(mail)
        return mail

    def _check_url(a):
        u = urlparse(a["url"])
        if u.scheme != "https":
            raise ToolError("Only https URLs are allowed")
        if (u.hostname or "").lower() not in svc.settings.allowed_hosts:
            raise ToolError(f"Host not in allowlist: {u.hostname}")

    def fetch_url(a):
        with svc.http_client_factory() as client:
            r = client.get(a["url"], timeout=10, follow_redirects=False)
        return {"url": a["url"], "status_code": r.status_code, "body": r.text[:2000]}

    def obj(props: dict, required: list[str]) -> dict:
        return {"type": "object", "properties": props, "required": required}

    reg.register(Tool("search_documents", "Semantic/keyword search over uploaded documents. Returns passages with file and page.",
                      obj({"query": {"type": "string"}, "top_k": {"type": "integer"}}, ["query"]), search_documents))
    reg.register(Tool("list_documents", "List uploaded documents.", obj({}, []), list_documents))
    reg.register(Tool("summarize_document", "Extractive summary of one document by id.",
                      obj({"document_id": {"type": "string"}, "sentences": {"type": "integer"}}, ["document_id"]),
                      summarize_document))
    reg.register(Tool("create_task", "Create a follow-up task. Requires human approval.",
                      obj({"title": {"type": "string"}, "due": {"type": "string"}, "assignee": {"type": "string"}}, ["title"]),
                      create_task, requires_approval=True, risk="medium"))
    reg.register(Tool("send_email", "Send an email. Requires human approval.",
                      obj({"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                          ["to", "subject", "body"]),
                      send_email, requires_approval=True, risk="high", precheck=_check_email))
    reg.register(Tool("fetch_url", "HTTP GET an allow-listed external API. Requires human approval.",
                      obj({"url": {"type": "string"}}, ["url"]),
                      fetch_url, requires_approval=True, risk="medium", precheck=_check_url))
    return reg
