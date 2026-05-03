from contextlib import contextmanager
from typing import Generator

from dotenv import load_dotenv
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import OTEL_SERVICE_NAME


def setup_tracing() -> None:
    """Initialize TracerProvider with OTLP gRPC exporter. Call once at process start."""
    # load_dotenv is a no-op when vars are already set (Modal secret injection)
    load_dotenv()
    resource = Resource.create({SERVICE_NAME: OTEL_SERVICE_NAME})
    # OTLPSpanExporter reads OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS from env
    exporter = OTLPSpanExporter()
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def get_tracer(name: str = OTEL_SERVICE_NAME) -> trace.Tracer:
    return trace.get_tracer(name)


def inject_context() -> dict[str, str]:
    """Serialize current W3C trace context into a dict for passing across Modal boundaries."""
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier


def extract_context(carrier: dict[str, str]) -> context.Context:
    """Deserialize a carrier dict from inject_context() back into a Context."""
    return extract(carrier)


@contextmanager
def restored_context(carrier: dict[str, str]) -> Generator[None, None, None]:
    """Attach a remote trace context in this process. Use at the top of every Modal function entry point.

    Example:
        with restored_context(carrier):
            with get_tracer().start_as_current_span("my_span"):
                ...
    """
    ctx = extract(carrier)
    token = context.attach(ctx)
    try:
        yield
    finally:
        context.detach(token)
