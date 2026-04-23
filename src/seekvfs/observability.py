"""OpenTelemetry hooks — soft dependency.

When ``opentelemetry-api`` is not installed, ``trace_span`` degrades to a
no-op so users never pay the instrumentation tax unless they opt in.
"""
from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

_tracer: Any | None = None
_otel_available: bool | None = None


def _have_otel() -> bool:
    global _otel_available
    if _otel_available is None:
        try:
            import opentelemetry.trace  # noqa: F401
            _otel_available = True
        except ImportError:
            _otel_available = False
    return _otel_available


def get_tracer() -> Any | None:
    """Lazily fetch the process-wide tracer; returns None if OT is absent."""
    global _tracer
    if _tracer is not None:
        return _tracer
    if not _have_otel():
        return None
    from opentelemetry import trace

    _tracer = trace.get_tracer("seekvfs")
    return _tracer


def trace_span(name: str) -> Callable[[F], F]:
    """Wrap a sync or async function in an OpenTelemetry span when OT is available."""

    def decorator(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                if tracer is None:
                    return await fn(*args, **kwargs)
                with tracer.start_as_current_span(name):
                    return await fn(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                tracer = get_tracer()
                if tracer is None:
                    return fn(*args, **kwargs)
                with tracer.start_as_current_span(name):
                    return fn(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["trace_span", "get_tracer"]
