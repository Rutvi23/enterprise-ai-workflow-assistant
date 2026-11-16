"""Tiny JSON-file persistence layer plus an append-only audit log.

Deliberately simple so the project runs with zero infrastructure. The `Store`
interface (snapshot / transaction / audit) is the seam where a real database
(e.g. Oracle AI Database, PostgreSQL) would be plugged in.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "state.json"
        self.audit_path = self.dir / "audit.jsonl"
        self._lock = threading.RLock()
        self._mtime: int | None = None
        self._data = self._empty()
        self._refresh()

    @staticmethod
    def _empty() -> dict:
        return {"documents": {}, "runs": {}, "approvals": {}, "tasks": [], "outbox": []}

    def _refresh(self) -> None:
        """Reload from disk if another process (e.g. the MCP server) changed the file."""
        if not self.path.exists():
            return
        mtime = self.path.stat().st_mtime_ns
        if mtime != self._mtime:
            self._data = {**self._empty(), **json.loads(self.path.read_text())}
            self._mtime = mtime

    def snapshot(self) -> dict:
        """Read-only view. Callers must not mutate the returned dict."""
        with self._lock:
            self._refresh()
            return self._data

    @contextmanager
    def transaction(self):
        with self._lock:
            self._refresh()
            try:
                yield self._data
            except Exception:
                self._mtime = None  # discard partial in-memory changes
                self._data = self._empty()
                self._refresh()
                raise
            self._flush()

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data))
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime_ns

    def audit(self, event: str, **details) -> None:
        record = {"ts": now(), "event": event, **details}
        with self._lock, self.audit_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def read_audit(self, limit: int = 100) -> list[dict]:
        if not self.audit_path.exists():
            return []
        lines = self.audit_path.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in reversed(lines)]
