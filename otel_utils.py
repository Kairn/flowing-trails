"""Centralized observability: tracing, structured logging, and OTel export.

Call setup_tracing() and setup_logging() once at process start.
Use get_tracer() for spans and get_logger() for structured logs.
Both export to Grafana Cloud via OTLP when endpoint env vars are set.
"""

import logging
import sys
from contextlib import contextmanager
from typing import Generator

import structlog
from dotenv import load_dotenv
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.propagate import extract, inject
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from config import OTEL_SERVICE_NAME

load_dotenv()

_tracing_initialized = False
_logging_initialized = False
_log_provider: LoggerProvider | None = None


def _make_resource() -> Resource:
    return Resource.create({SERVICE_NAME: OTEL_SERVICE_NAME})


def _add_otel_context(logger, method_name, event_dict):
    """Structlog processor: inject trace_id/span_id from the active OTel span."""
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx.trace_id != 0:
        event_dict["trace_id"] = format(ctx.trace_id, "032x")
        event_dict["span_id"] = format(ctx.span_id, "016x")
    return event_dict


def setup_tracing() -> None:
    """Initialize TracerProvider with OTLP HTTP exporter. Safe to call multiple times."""
    global _tracing_initialized
    if _tracing_initialized:
        return
    _tracing_initialized = True
    provider = TracerProvider(resource=_make_resource())
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structlog + stdlib logging with OTel log export and trace correlation. Safe to call multiple times."""
    global _log_provider, _logging_initialized
    if _logging_initialized:
        return
    _logging_initialized = True

    # OTel log export — sends to Grafana Cloud Logs via same OTLP endpoint as traces
    _log_provider = LoggerProvider(resource=_make_resource())
    _log_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))

    shared_processors = [
        structlog.stdlib.add_log_level,
        _add_otel_context,
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Console: pretty in terminals, JSON otherwise (Modal, CI, piped output)
    console = logging.StreamHandler(sys.stderr)
    renderer = (
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )
    console.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(LoggingHandler(level=level, logger_provider=_log_provider))
    root.setLevel(level)

    for name in ("urllib3", "httpx", "httpcore", "opentelemetry"):
        logging.getLogger(name).setLevel(logging.WARNING)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger with trace-correlated, structured output."""
    return structlog.get_logger(name)


def get_tracer(name: str = OTEL_SERVICE_NAME) -> trace.Tracer:
    return trace.get_tracer(name)


def flush_telemetry() -> None:
    """Force-flush traces and logs. Call before process exit."""
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
    if _log_provider is not None:
        _log_provider.force_flush()


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
    """Attach a remote trace context in this process."""
    ctx = extract(carrier)
    token = context.attach(ctx)
    try:
        yield
    finally:
        context.detach(token)
