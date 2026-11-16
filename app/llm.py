"""LLM provider abstraction.

* `OpenAIProvider` talks to any OpenAI-compatible /chat/completions endpoint
  (OpenAI, Azure OpenAI-compatible gateways, Ollama, vLLM, ...), including tool calling.
* `MockProvider` is a deterministic, rule-based stand-in so the whole app (and the
  test-suite / CI) runs offline with no API key. It is NOT a language model.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import httpx
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMProvider(Protocol):
    name: str

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...


class OpenAIProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str,
                 client: httpx.Client | None = None, timeout: float = 60):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout)

    def chat(self, messages, tools=None) -> LLMResponse:
        payload: dict = {"model": self.model, "messages": messages, "temperature": 0.1}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        r = self.client.post(f"{self.base_url}/chat/completions", json=payload, headers=self.headers)
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"__invalid_json__": raw}  # rejected later by argument validation
            calls.append(ToolCall(id=tc["id"], name=tc["function"]["name"], arguments=args))
        return LLMResponse(content=msg.get("content"), tool_calls=calls)


# --------------------------------------------------------------------------- mock

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL = re.compile(r"https?://\S+")


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS and len(w) > 2}


class MockProvider:
    name = "mock"

    def chat(self, messages, tools=None) -> LLMResponse:
        if not tools:
            return LLMResponse(content=self._grounded_answer(messages))
        return self._agent_step(messages)

    # -- RAG answer mode ---------------------------------------------------
    def _grounded_answer(self, messages: list[dict]) -> str:
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        question = prompt.split("QUESTION:", 1)[-1].strip()
        context = prompt.split("CONTEXT:", 1)[-1].split("QUESTION:", 1)[0]
        blocks = re.findall(r"\[(\d+)\] \([^)]*\) (.*?)(?=\n\[\d+\] \(|\Z)", context.strip(), re.S)
        q = _tokens(question)
        scored = []
        for n, text in blocks:
            for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
                scored.append((len(q & _tokens(sent)), len(scored), n, sent.strip()))
        if not scored:
            return "I could not find this in the uploaded documents."
        best = sorted([s for s in scored if s[0] > 0], key=lambda s: (-s[0], s[1]))[:2]
        best = best or scored[:1]
        best.sort(key=lambda s: s[1])
        return "Based on the documents: " + " ".join(f"{s[3]} [{s[2]}]" for s in best)

    # -- agent mode --------------------------------------------------------
    @staticmethod
    def _call(name: str, args: dict) -> LLMResponse:
        return LLMResponse(tool_calls=[ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=name, arguments=args)])

    @staticmethod
    def _history(messages: list[dict]):
        names, results = {}, []
        for m in messages:
            for tc in m.get("tool_calls") or []:
                names[tc["id"]] = tc["function"]["name"]
            if m["role"] == "tool":
                try:
                    content = json.loads(m["content"])
                except json.JSONDecodeError:
                    content = {"raw": m["content"]}
                results.append((names.get(m["tool_call_id"], "?"), content))
        called = list(names.values())
        return called, results

    def _agent_step(self, messages: list[dict]) -> LLMResponse:
        user = next(m["content"] for m in messages if m["role"] == "user")
        text = user.lower()
        called, results = self._history(messages)
        email = _EMAIL.search(user)
        url = _URL.search(user)

        if email and re.search(r"\b(email|e-mail|mail|send)\b", text):
            topic = self._topic(user)
            if "search_documents" not in called:
                return self._call("search_documents", {"query": topic})
            if "send_email" not in called:
                hits = next((r["results"] for n, r in results if n == "search_documents" and "results" in r), [])
                lines = [f"- [{h['filename']} p.{h['page']}] {h['text'][:300].strip()}" for h in hits[:2]]
                body = f"Summary of relevant document excerpts regarding '{topic}':\n\n" + (
                    "\n".join(lines) or "(no relevant passages found)"
                )
                return self._call("send_email", {"to": email.group(0), "subject": f"Summary: {topic[:60]}", "body": body})
        elif re.search(r"\btask\b", text):
            if "create_task" not in called:
                m = re.search(r"task\s*(?:to|for|:|-)?\s*(.+)", user, re.I)
                title = (m.group(1) if m else user).strip().rstrip(".")
                due = None
                dm = re.search(r"\b(?:by|due)\s+(.+)$", title, re.I)
                if dm:
                    due, title = dm.group(1).strip(), title[: dm.start()].strip()
                args = {"title": title}
                if due:
                    args["due"] = due
                am = re.search(r"assign(?:ed)?\s+to\s+(\w+)", user, re.I)
                if am:
                    args["assignee"] = am.group(1)
                return self._call("create_task", args)
        elif url and re.search(r"\b(fetch|get|call|open|check)\b", text):
            if "fetch_url" not in called:
                return self._call("fetch_url", {"url": url.group(0).rstrip(".,)")})
        elif "list" in text and "document" in text:
            if "list_documents" not in called:
                return self._call("list_documents", {})
        elif not called:
            return self._call("search_documents", {"query": user})

        return LLMResponse(content=self._final(results))

    @staticmethod
    def _topic(user: str) -> str:
        cleaned = _EMAIL.sub("", user)
        m = re.search(r"(?:about|regarding|summary of|summarize|on)\s+(.+)", cleaned, re.I)
        topic = (m.group(1) if m else cleaned).strip(" .")
        return re.sub(r"^(send|email|mail)\s+(an?\s+)?(email\s+)?(to\s+)?", "", topic, flags=re.I) or user

    @staticmethod
    def _final(results: list[tuple[str, dict]]) -> str:
        if not results:
            return "I did not need any tools for that request."
        name, res = results[-1]
        if res.get("status") == "rejected":
            return f"The reviewer did not approve `{name}`, so nothing was executed."
        if "error" in res:
            return f"`{name}` failed: {res['error']}"
        if name == "send_email":
            return f"Email to {res.get('to')} was sent (subject: {res.get('subject')})."
        if name == "create_task":
            return f"Task created: {res.get('title')} (id {res.get('id')})."
        if name == "fetch_url":
            return f"Fetched {res.get('url')} (HTTP {res.get('status_code')})."
        if name == "list_documents":
            docs = res.get("documents", [])
            return "Documents: " + (", ".join(d["filename"] for d in docs) or "none uploaded yet.")
        if name == "search_documents":
            hits = res.get("results", [])
            if not hits:
                return "I found no relevant passages in the uploaded documents."
            h = hits[0]
            return f"Top match in {h['filename']} (page {h['page']}): {h['text'][:300]}"
        return "Done."
