
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# starlette: HTTP exception base class
from starlette.exceptions import HTTPException as StarletteHTTPException

# pydantic: validation error type
from pydantic import ValidationError

# logging: structured error logging
import logging

# typing: type annotations
from typing import Union

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Registers all global exception handlers on the FastAPI app.

    Called once at app initialization. FastAPI routes all exceptions
    through these handlers before returning to the client.

    Handler priority:
    1. RequestValidationError (422) — Pydantic validation failures
    2. HTTPException (4xx/5xx)      — Explicit HTTP errors
    3. ValidationError              — Pydantic model errors
    4. Exception (catch-all)        — Any unhandled exception

    Args:
        app: FastAPI application instance to register handlers on
    """

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handles Pydantic validation errors from request body parsing.

        Triggered when:
        - Request body doesn't match the expected Pydantic model
        - Required fields are missing from request body
        - Field values fail type validation

        FastAPI's default 422 response has a complex nested structure.
        We flatten it into a human-readable format.

        Example default (complex):
        {"detail": [{"loc": ["body", "title"], "msg": "field required", "type": "missing"}]}

        Our format (clean):
        {"error": "Validation Error", "detail": "title: field required", "request_id": "..."}
        """
        request_id = getattr(request.state, "request_id", "unknown")

        # Extract error messages from Pydantic's error list
        # exc.errors() returns: [{"loc": [...], "msg": "...", "type": "..."}, ...]
        error_messages = []
        for error in exc.errors():
            # loc: tuple of field location path → join with "."
            # Example: ("body", "title") → "body.title"
            location = " → ".join(str(loc) for loc in error["loc"])
            message  = error["msg"]
            error_messages.append(f"{location}: {message}")

        # Join multiple validation errors with semicolons
        detail = "; ".join(error_messages)

        logger.warning(
            f"Validation error [req={request_id[:8]}]: {detail}"
        )

        return JSONResponse(
            # 422 Unprocessable Entity: request syntax valid but semantics invalid
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error":      "Validation Error",
                "detail":     detail,
                "request_id": request_id,
                "status_code": 422,
            }
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request,
        exc: StarletteHTTPException
    ) -> JSONResponse:
        """
        Handles explicit HTTP exceptions raised via HTTPException(status_code=N).

        Triggered when:
        - Route handler raises HTTPException(404, detail="Not found")
        - FastAPI raises 405 for wrong HTTP method
        - FastAPI raises 415 for unsupported media type

        Adds request_id to the standard HTTPException response format.
        """
        request_id = getattr(request.state, "request_id", "unknown")

        # Log at appropriate level based on status code
        if exc.status_code >= 500:
            logger.error(
                f"HTTP {exc.status_code} [req={request_id[:8]}]: {exc.detail}"
            )
        else:
            logger.warning(
                f"HTTP {exc.status_code} [req={request_id[:8]}]: {exc.detail}"
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error":      http_status_name(exc.status_code),
                "detail":     exc.detail,
                "request_id": request_id,
                "status_code": exc.status_code,
            }
        )

    @app.exception_handler(ValidationError)
    async def pydantic_validation_handler(
        request: Request,
        exc: ValidationError
    ) -> JSONResponse:
        """
        Handles Pydantic ValidationError from model instantiation in route code.

        This is different from RequestValidationError:
        - RequestValidationError: request body fails validation
        - ValidationError:        model instantiation in route code fails

        Example:
            result = ATSScoreResult(**bad_data)  # raises ValidationError
        """
        request_id = getattr(request.state, "request_id", "unknown")

        logger.error(
            f"Internal Pydantic error [req={request_id[:8]}]: {str(exc)}"
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error":      "Internal Data Validation Error",
                "detail":     "An internal data consistency error occurred.",
                "request_id": request_id,
                "status_code": 500,
            }
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """
        Catch-all handler for any unhandled exception.

        This is the last line of defense — catches everything not caught
        by more specific handlers above.

        Security: NEVER return exc details to client in production.
        Internal Python errors may contain file paths, variable names,
        database connection strings, or other sensitive information.

        We log the full traceback internally but return only a generic
        "something went wrong" message to the client.
        """
        request_id = getattr(request.state, "request_id", "unknown")

        # Log full exception with traceback for internal debugging
        # exc_info=True: includes traceback in log output
        logger.error(
            f"Unhandled exception [req={request_id[:8]}]: "
            f"{type(exc).__name__}: {str(exc)}",
            exc_info=True  # Include full Python traceback in log
        )

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error":      "Internal Server Error",
                # Safe generic message — never expose exc details
                "detail":     (
                    "An unexpected error occurred. "
                    f"Reference ID: {request_id} for support."
                ),
                "request_id": request_id,
                "status_code": 500,
            }
        )


def http_status_name(status_code: int) -> str:
    """
    Returns a human-readable name for an HTTP status code.

    Used in error responses to provide context alongside the code.
    Example: 404 → "Not Found", 422 → "Unprocessable Entity"

    Args:
        status_code: Integer HTTP status code

    Returns:
        Human-readable status name string
    """
    status_names = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        405: "Method Not Allowed",
        409: "Conflict",
        413: "Payload Too Large",
        415: "Unsupported Media Type",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        502: "Bad Gateway",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return status_names.get(status_code, f"HTTP Error {status_code}")
