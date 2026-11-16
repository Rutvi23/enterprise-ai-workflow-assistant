"""Minimal Model Context Protocol (MCP) server over stdio (newline-delimited JSON-RPC 2.0).

Exposes the same tool registry as the web agent. Tools flagged `requires_approval` do NOT
execute when called over MCP: they create a pending approval that a human resolves in the web UI
(POST /api/approvals/{id}/approve), so human-in-the-loop holds for every client.

Run:  python -m app.mcp_server
Claude Desktop / any MCP client config:
  {"mcpServers": {"workflow-assistant": {"command": "python", "args": ["-m", "app.mcp_server"],
                                         "cwd": "/path/to/enterprise-ai-workflow-assistant"}}}
"""
from __future__ import annotations

import json
import sys

from .config import get_settings
from .services import Services
from .tools import ToolError

PROTOCOL_VERSION = "2024-11-05"


def _result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _text(payload, is_error=False):
    return {"content": [{"type": "text", "text": payload if isinstance(payload, str) else json.dumps(payload)}],
            "isError": is_error}


def handle_message(svc: Services, msg: dict) -> dict | None:
    """Return a JSON-RPC response, or None for notifications."""
    method, msg_id, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
    if msg_id is None:  # notification (e.g. notifications/initialized)
        return None

    if method == "initialize":
        return _result(msg_id, {
            "protocolVersion": params.get("protocolVersion", PROTOCOL_VERSION),
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "enterprise-ai-workflow-assistant", "version": "1.0.0"},
        })
    if method == "ping":
        return _result(msg_id, {})
    if method == "tools/list":
        return _result(msg_id, {"tools": [
            {"name": t.name,
             "description": t.description + (" [needs human approval]" if t.requires_approval else ""),
             "inputSchema": t.parameters} for t in svc.registry.all()]})
    if method == "tools/call":
        name, args = params.get("name"), params.get("arguments") or {}
        tool = svc.registry.get(name)
        if tool is None:
            return _error(msg_id, -32602, f"Unknown tool: {name}")
        try:
            svc.registry.validate(tool, args)
            if tool.requires_approval:
                approval = svc.agent.create_approval(None, {"id": "mcp-call", "arguments": args}, tool)
                return _result(msg_id, _text(
                    f"Approval required before '{name}' can run. Pending approval id: {approval['id']}. "
                    "A human must approve it in the web UI."))
            result = tool.handler(args)
            svc.store.audit("tool_executed", source="mcp", tool=name, arguments=args, ok=True)
            return _result(msg_id, _text(result))
        except ToolError as exc:
            return _result(msg_id, _text(str(exc), is_error=True))
        except Exception as exc:  # tool crashed
            return _result(msg_id, _text(f"{type(exc).__name__}: {exc}", is_error=True))
    return _error(msg_id, -32601, f"Method not found: {method}")


def main() -> None:
    svc = Services(get_settings())
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            sys.stdout.write(json.dumps(_error(None, -32700, "Parse error")) + "\n")
            sys.stdout.flush()
            continue
        response = handle_message(svc, msg)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
