"""Convert ToolSpecSet to an MCP server instance.

Soft-depends on ``mcp``; raises an informative ``ImportError`` if missing.
"""
from __future__ import annotations

from typing import Any

from seekvfs.tools.spec import ToolSpecSet


def _require_mcp():
    try:
        from mcp.server import Server
        from mcp.types import Tool
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "to_mcp requires `mcp`. Install with: pip install 'seekvfs[mcp]'"
        ) from e
    return Server, Tool


def to_mcp(specs: ToolSpecSet, server_name: str = "seekvfs") -> Any:
    Server, Tool = _require_mcp()
    server = Server(server_name)

    tools = [
        Tool(
            name=s.name,
            description=s.description,
            inputSchema=s.parameters_schema,
        )
        for s in specs.specs
    ]

    @server.list_tools()
    async def _list_tools() -> list[Any]:
        return tools

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict) -> Any:
        spec = specs.by_name(name)
        return spec.callable(**(arguments or {}))

    return server


__all__ = ["to_mcp"]
