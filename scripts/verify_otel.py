"""Verify Grafana Cloud OTLP connection.

Creates a test trace with two spans and flushes them.
Search for the logged trace ID in Grafana Tempo to confirm export works.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from otel_utils import (
    flush_telemetry,
    get_logger,
    get_tracer,
    setup_logging,
    setup_tracing,
)

setup_tracing()
setup_logging()
log = get_logger("verify-otel")
tracer = get_tracer("verify-otel")

with tracer.start_as_current_span("verify-connection") as parent:
    parent.set_attribute("test.purpose", "m0-t7-otel-verify")
    parent.set_attribute("test.timestamp", time.time())

    with tracer.start_as_current_span("child-operation") as child:
        child.set_attribute("test.step", "child")
        child.add_event("verification-event", {"detail": "hello from FlowingTrails"})
        time.sleep(0.05)

    trace_id = format(parent.get_span_context().trace_id, "032x")

flush_telemetry()
log.info("Trace exported — search Grafana Tempo for this trace", trace_id=trace_id)
