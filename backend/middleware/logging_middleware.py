
# BaseHTTPMiddleware wraps the ASGI app and provides a clean interface
# for intercepting requests and responses
from starlette.middleware.base import BaseHTTPMiddleware

# starlette.requests: Request object with headers, body, query params
from starlette.requests import Request

# starlette.responses: Response type for type hints
from starlette.responses import Response

# uuid: generates unique request identifiers
import uuid

# time: high-resolution timing using monotonic clock
import time

# logging: structured log output
import logging

# typing: type annotations
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    Attaches a unique UUID to every request as X-Request-ID header.

    Header behavior:
    - If client sends X-Request-ID: use it (client-initiated tracing)
    - If client sends no X-Request-ID: generate a new UUID4

    This enables end-to-end request tracing:
    Client → Load Balancer → FastAPI → MongoDB → Response
    All log lines include the same request_id for correlation.

    The request_id is stored in request.state — accessible by
    route handlers, dependencies, and other middleware.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Middleware dispatch method called for every request.

        Args:
            request:   Incoming HTTP request object
            call_next: Callable that passes request to next layer
                       (next middleware or route handler)

        Returns:
            HTTP Response with X-Request-ID header added
        """

        # Check if client provided a request ID (distributed tracing)
        # If the client already has a trace ID (e.g., from a gateway),
        # we preserve it rather than generating a new one
        request_id = request.headers.get(
            "X-Request-ID",
            str(uuid.uuid4())  # Generate new UUID4 if not provided
        )

        # Store in request.state — accessible throughout request lifecycle
        # request.state is a SimpleNamespace — arbitrary attribute storage
        request.state.request_id = request_id

        # Pass request to the next middleware or route handler
        # await is required — call_next is an async callable
        response = await call_next(request)

        # Add request ID to response headers
        # This lets clients correlate their request with server logs
        response.headers["X-Request-ID"] = request_id

        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """
    Measures and logs HTTP request processing time.

    Adds X-Process-Time header (in milliseconds) to every response.
    Logs structured request details: method, path, status, duration.

    Log format designed for structured log parsing:
        INFO: GET /api/v1/upload/resume 201 45.32ms [req_id=abc123]

    Monitoring systems parse these structured logs to build:
    - Request rate graphs
    - Latency percentile distributions (P50, P95, P99)
    - Error rate alerts
    - Slow endpoint identification
    """

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Records start time, processes request, logs timing on completion.

        Uses time.monotonic() instead of time.time():
        - monotonic: always increases, not affected by system clock changes
        - time():    can jump backward if NTP adjusts the system clock
        For measuring durations, monotonic is always correct.
        """

        # Record start time using monotonic clock
        start_time = time.monotonic()

        # Get request metadata for logging
        method = request.method         # GET, POST, PUT, DELETE
        path   = request.url.path       # /api/v1/upload/resume
        client = request.client         # (host, port) tuple

        # Get request ID (set by RequestIDMiddleware if ordered correctly)
        # Fallback to "unknown" if RequestIDMiddleware hasn't run yet
        request_id = getattr(request.state, "request_id", "unknown")

        # Process the request through remaining middleware and route handler
        response = await call_next(request)

        # Calculate processing time in milliseconds
        # monotonic() returns seconds as float → multiply by 1000 for ms
        process_time_ms = (time.monotonic() - start_time) * 1000

        # Add timing header to response
        # X-Process-Time: standard header name for API latency
        response.headers["X-Process-Time"] = f"{process_time_ms:.2f}ms"

        # Structured log line with all request context
        # Logging level based on status code:
        # 2xx → INFO, 4xx → WARNING, 5xx → ERROR
        status_code = response.status_code

        log_data = (
            f"{method} {path} {status_code} "
            f"{process_time_ms:.2f}ms "
            f"[req={request_id[:8]}]"
        )

        if status_code >= 500:
            logger.error(f"SERVER ERROR: {log_data}")
        elif status_code >= 400:
            logger.warning(f"CLIENT ERROR: {log_data}")
        else:
            logger.info(f"REQUEST: {log_data}")

        return response
