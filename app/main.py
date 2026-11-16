from __future__ import annotations

import hashlib
import secrets
import uuid
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .agent import Conflict, NotFound
from .config import Settings, get_settings
from .ingestion import SUPPORTED, IngestionError, chunk_pages, extract_pages
from .rag import answer_question
from .services import Services
from .store import now
from .tools import ToolError

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=4, ge=1, le=10)
    document_ids: list[str] | None = None


class RunRequest(BaseModel):
    instruction: str = Field(min_length=3, max_length=4000)


class ResolveRequest(BaseModel):
    reviewer: str = Field(default="reviewer", max_length=80)
    note: str = Field(default="", max_length=1000)
    arguments: dict | None = None  # reviewer may edit arguments before approving


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    svc = services or Services(settings)
    app = FastAPI(title="Enterprise AI Workflow Assistant", version="1.0.0")
    app.state.services = svc

    def require_key(x_api_key: str | None = Header(default=None)):
        if settings.api_key and not (x_api_key and secrets.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(401, "Invalid or missing X-API-Key")

    auth = [Depends(require_key)]

    @app.get("/health")
    def health():
        return {"status": "ok", "llm": svc.llm.name, "retriever": svc.retriever.name}

    # ---------------------------------------------------------------- documents
    @app.post("/api/documents", dependencies=auth, status_code=201)
    async def upload_document(file: UploadFile = File(...)):
        filename = Path(file.filename or "upload").name
        if Path(filename).suffix.lower() not in SUPPORTED:
            raise HTTPException(415, "Unsupported file type. Use .pdf, .txt or .md")
        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
        digest = hashlib.sha256(content).hexdigest()
        for d in svc.store.snapshot()["documents"].values():
            if d["sha256"] == digest:
                return {k: v for k, v in d.items() if k != "chunks"} | {"duplicate": True}
        try:
            pages = extract_pages(filename, content)
        except IngestionError as exc:
            raise HTTPException(422, str(exc)) from exc
        doc_id = uuid.uuid4().hex[:12]
        doc = {"id": doc_id, "filename": filename, "sha256": digest, "size": len(content),
               "pages": len(pages), "created_at": now(),
               "chunks": chunk_pages(doc_id, pages, settings.chunk_size, settings.chunk_overlap)}
        with svc.store.transaction() as d:
            d["documents"][doc_id] = doc
        svc.store.audit("document_uploaded", document_id=doc_id, filename=filename, chunks=len(doc["chunks"]))
        return {k: v for k, v in doc.items() if k != "chunks"} | {"chunk_count": len(doc["chunks"])}

    @app.get("/api/documents", dependencies=auth)
    def list_documents():
        return [{k: v for k, v in d.items() if k != "chunks"} | {"chunk_count": len(d["chunks"])}
                for d in svc.store.snapshot()["documents"].values()]

    @app.delete("/api/documents/{doc_id}", dependencies=auth, status_code=204)
    def delete_document(doc_id: str):
        with svc.store.transaction() as d:
            if doc_id not in d["documents"]:
                raise HTTPException(404, "Document not found")
            del d["documents"][doc_id]
        svc.store.audit("document_deleted", document_id=doc_id)

    # ---------------------------------------------------------------------- RAG
    @app.post("/api/ask", dependencies=auth)
    def ask(req: AskRequest):
        try:
            return answer_question(svc, req.question, req.top_k, req.document_ids)
        except httpx.HTTPError as exc:
            raise HTTPException(502, f"Upstream model/embedding error: {exc}") from exc

    # -------------------------------------------------------------------- agent
    @app.get("/api/tools", dependencies=auth)
    def tools():
        return [{"name": t.name, "description": t.description, "risk": t.risk,
                 "requires_approval": t.requires_approval, "parameters": t.parameters}
                for t in svc.registry.all()]

    @app.post("/api/agent/runs", dependencies=auth, status_code=201)
    def start_run(req: RunRequest):
        return svc.agent.start(req.instruction)

    @app.get("/api/agent/runs", dependencies=auth)
    def list_runs():
        return svc.agent.list_runs()

    @app.get("/api/agent/runs/{run_id}", dependencies=auth)
    def get_run(run_id: str):
        try:
            return svc.agent.get_run(run_id)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc

    # ---------------------------------------------------------------- approvals
    @app.get("/api/approvals", dependencies=auth)
    def approvals(status: str | None = None):
        items = sorted(svc.store.snapshot()["approvals"].values(), key=lambda a: a["created_at"], reverse=True)
        return [a for a in items if status is None or a["status"] == status]

    def _resolve(approval_id: str, approve: bool, body: ResolveRequest):
        try:
            return svc.agent.resolve(approval_id, approve, body.reviewer, body.note, body.arguments)
        except NotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except ToolError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/approvals/{approval_id}/approve", dependencies=auth)
    def approve(approval_id: str, body: ResolveRequest = ResolveRequest()):
        return _resolve(approval_id, True, body)

    @app.post("/api/approvals/{approval_id}/reject", dependencies=auth)
    def reject(approval_id: str, body: ResolveRequest = ResolveRequest()):
        return _resolve(approval_id, False, body)

    # ------------------------------------------------------------ audit & state
    @app.get("/api/audit", dependencies=auth)
    def audit(limit: int = 100):
        return svc.store.read_audit(min(limit, 500))

    @app.get("/api/outbox", dependencies=auth)
    def outbox():
        d = svc.store.snapshot()
        return {"emails": d["outbox"], "tasks": d["tasks"]}

    # ---------------------------------------------------------------------- UI
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    return app

