"""Convert ToolSpecSet to LangGraph / LangChain StructuredTool list.

Soft-depends on ``langchain-core``; raises an informative ``ImportError`` if
missing, directing the user to the ``[langgraph]`` extra.
"""
from __future__ import annotations

from typing import Any

from seekvfs.tools.spec import ToolSpec, ToolSpecSet


def _require_langchain():
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "to_langgraph requires `langchain-core`. "
            "Install with: pip install 'seekvfs[langgraph]'"
        ) from e
    return StructuredTool


def _to_tool(StructuredTool: Any, spec: ToolSpec) -> Any:  # noqa: N803
    return StructuredTool.from_function(
        func=spec.callable,
        name=spec.name,
        description=spec.description,
        args_schema=None,  # rely on description + native param coercion
    )


def to_langgraph(specs: ToolSpecSet) -> list[Any]:
    StructuredTool = _require_langchain()
    return [_to_tool(StructuredTool, s) for s in specs.specs]


__all__ = ["to_langgraph"]
