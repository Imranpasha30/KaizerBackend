"""OpenTelemetry tracing — opt-in via KAIZER_OTEL_ENABLED=true.

Why opt-in?
  - The OTel stack is heavy (~3 packages, hundreds of MB of deps via
    transitive grpc/protobuf code paths).
  - Local development doesn't benefit unless the dev runs a Tempo /
    Jaeger / Honeycomb backend, which most people don't.
  - The existing ``[metrics] stage=… ms=…`` log lines emitted by
    :mod:`youtube.worker` are always on and grep-able, so we never
    lose visibility just because OTel is off.

When OTel is on:
  - Auto-instruments FastAPI (one span per HTTP request, with
    method/path/status as attributes).
  - Auto-instruments redis-py (one span per Redis command — XADD,
    XREADGROUP, GET, SETEX, …). Useful for catching slow Redis ops
    that look like worker stalls.
  - Manual spans in ``youtube/worker.py`` wrap the per-job claim /
    process / ack so a job lifecycle is one trace tree.

Backends:
  Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` to a collector URL (e.g.
  ``http://otel-collector.internal:4318``). Honeycomb, Datadog,
  Grafana Tempo, and AWS X-Ray all speak OTLP/HTTP. If the env var
  is unset but ``KAIZER_OTEL_ENABLED=true``, we fall back to the
  console exporter — spans dump to stdout for local debugging.

Sampling:
  Default: parent-based, with 5% root sampling (``KAIZER_OTEL_SAMPLE_RATE``
  env var, range 0.0-1.0). At enterprise volume, full-trace sampling
  bankrupts the trace backend; 5% is the canonical SaaS default.

Public surface (everything else is internal):

    init_tracing(service_name, service_version)
        Call ONCE on FastAPI startup. Idempotent.

    span(name, **attrs) -> ContextManager
        Whether OTel is on or off, returns a usable context manager.
        When off, it's a no-op cheap class — adds <1µs.

    is_enabled() -> bool
        For callers that want to gate expensive attribute computation
        on whether anyone will see it.
"""
from __future__ import annotations

import contextlib
import logging
import os
from typing import Iterator, Optional

logger = logging.getLogger("kaizer.tracing")

_initialised = False
_enabled = False
_tracer = None  # type: ignore[assignment]


def is_enabled() -> bool:
    return _enabled


def _otel_enabled_env() -> bool:
    return (os.environ.get("KAIZER_OTEL_ENABLED", "false").strip().lower()
            in ("true", "1", "yes", "on"))


def init_tracing(
    service_name: str = "kaizer-backend",
    service_version: str = "1.0",
) -> bool:
    """Set up OTel tracer + auto-instrumentation. Idempotent.

    Returns True iff tracing is now active. Always safe to call.
    """
    global _initialised, _enabled, _tracer
    if _initialised:
        return _enabled
    _initialised = True

    if not _otel_enabled_env():
        logger.info("tracing: KAIZER_OTEL_ENABLED is false — running with no-op spans")
        _enabled = False
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor, ConsoleSpanExporter,
        )
        from opentelemetry.sdk.trace.sampling import (
            ParentBased, TraceIdRatioBased,
        )
    except Exception as exc:
        logger.error("tracing: opentelemetry SDK import failed: %s", exc)
        _enabled = False
        return False

    sample_rate = float(os.environ.get("KAIZER_OTEL_SAMPLE_RATE", "0.05"))
    sample_rate = max(0.0, min(1.0, sample_rate))

    resource = Resource.create({
        SERVICE_NAME:    service_name,
        SERVICE_VERSION: service_version,
        "deployment.environment": os.environ.get("KAIZER_ENV", "dev"),
        "host.name":     os.environ.get("HOSTNAME", "") or _safe_hostname(),
    })
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sample_rate)),
    )

    # Pick exporter — OTLP/HTTP if configured, else console (dev).
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter(endpoint=otlp_endpoint.rstrip("/") + "/v1/traces")
            logger.info("tracing: OTLP exporter → %s", otlp_endpoint)
        except Exception as exc:
            logger.warning("tracing: OTLP exporter failed (%s) — falling back to console", exc)
            exporter = ConsoleSpanExporter()
    else:
        logger.info("tracing: OTEL_EXPORTER_OTLP_ENDPOINT unset — using ConsoleSpanExporter")
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name, service_version)

    # Best-effort auto-instrumentation. Each one is wrapped because a
    # missing extra (e.g. redis instrumentation not installed) shouldn't
    # tear down tracing for the bits that DID install.
    _try_instrument_redis()

    _enabled = True
    logger.info("tracing: enabled (service=%s sample=%.2f)", service_name, sample_rate)
    return True


def instrument_fastapi(app) -> None:
    """Add FastAPI auto-instrumentation. Call AFTER ``init_tracing()``
    and AFTER FastAPI() instance is constructed.

    Kept separate from init_tracing because the app object is not
    available where init_tracing typically runs.
    """
    if not _enabled:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("tracing: FastAPI instrumented")
    except Exception as exc:
        logger.warning("tracing: FastAPI instrumentation failed: %s", exc)


def _try_instrument_redis() -> None:
    try:
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        RedisInstrumentor().instrument()
        logger.info("tracing: redis-py instrumented")
    except Exception as exc:
        logger.warning("tracing: redis instrumentation failed: %s", exc)


def _safe_hostname() -> str:
    try:
        import socket
        return socket.gethostname()
    except Exception:
        return ""


# ── Manual span helper ──────────────────────────────────────────
class _NoopSpan:
    """Stand-in span when OTel is disabled — same interface, zero cost."""
    def set_attribute(self, *_args, **_kwargs) -> None: pass
    def set_status(self,    *_args, **_kwargs) -> None: pass
    def add_event(self,     *_args, **_kwargs) -> None: pass
    def record_exception(self, *_args, **_kwargs) -> None: pass


@contextlib.contextmanager
def span(name: str, **attrs) -> Iterator[object]:
    """Open a span. Use as ``with span("name", k=v): ...``.

    When OTel is off, yields a no-op span object — no allocation cost
    beyond the contextmanager wrapper.
    """
    if not _enabled or _tracer is None:
        yield _NoopSpan()
        return
    with _tracer.start_as_current_span(name) as sp:
        for k, v in attrs.items():
            try:
                sp.set_attribute(k, v)
            except Exception:
                pass
        try:
            yield sp
        except Exception as exc:
            try:
                sp.record_exception(exc)
                # OK status remains unless caller sets ERROR explicitly;
                # the recorded exception is what shows on the trace.
            except Exception:
                pass
            raise
