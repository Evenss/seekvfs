from __future__ import annotations

import pytest

from seekvfs.vfs import VFS
from tests.conftest import _StubBackend

pytest.importorskip("opentelemetry")


@pytest.fixture
def tracer_provider():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    from seekvfs import observability

    observability._tracer = None
    observability._otel_available = None
    return exporter


def test_write_emits_span(tracer_provider) -> None:
    vfs = VFS(routes={"seekvfs://a/": {"backend": _StubBackend()}})
    vfs.write("seekvfs://a/p", "hello")
    spans = tracer_provider.get_finished_spans()
    names = {s.name for s in spans}
    assert "vfs.write" in names
