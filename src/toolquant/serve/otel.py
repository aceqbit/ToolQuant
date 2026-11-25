"""OpenTelemetry setup. OTLP/HTTP export if OTEL_EXPORTER_OTLP_ENDPOINT is set, else in-memory."""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_memory = InMemorySpanExporter()


def setup(service_name: str = "toolquant") -> trace.Tracer:
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    provider.add_span_processor(SimpleSpanProcessor(_memory))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("toolquant")


def finished_spans() -> list:
    return list(_memory.get_finished_spans())
