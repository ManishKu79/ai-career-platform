# backend/middleware/cors.py

from fastapi.middleware.cors import CORSMiddleware

# FastAPI app type for type hints
from fastapi import FastAPI

# Our settings
from backend.config import settings

# logging
import logging

logger = logging.getLogger(__name__)


def configure_cors(app: FastAPI) -> None:
    """
    Adds CORS middleware to the FastAPI application.

    CORS (Cross-Origin Resource Sharing) is a browser security mechanism
    that restricts HTTP requests made from one origin to another.

    Our setup:
    - Streamlit runs on http://localhost:8501
    - FastAPI runs on http://localhost:8000
    - Browser blocks requests from 8501 → 8000 without CORS headers
    - Adding CORS middleware tells the browser this is allowed

    How CORS works:
    1. Browser sends OPTIONS preflight: "Can I send POST to localhost:8000?"
    2. CORS middleware responds: "Yes, from localhost:8501 you can"
    3. Browser sends actual POST request
    4. CORS middleware adds Access-Control headers to response

    allow_origins: List of domains that can make requests
    allow_methods: Which HTTP methods are permitted
    allow_headers: Which request headers are permitted
    allow_credentials: Whether cookies/auth headers are included

    Args:
        app: FastAPI application to add CORS middleware to
    """

    # Origins that are allowed to make cross-origin requests
    # In production: replace with your exact domain list
    # Never use ["*"] with allow_credentials=True — security risk
    allowed_origins = [
        "http://localhost:8501",      # Streamlit local development
        "http://localhost:3000",      # Optional: React dev server
        "http://127.0.0.1:8501",     # Streamlit (127.0.0.1 variant)
        "http://0.0.0.0:8501",       # Docker internal network
    ]

    # In development, log which origins are allowed
    logger.info(f"CORS configured for origins: {allowed_origins}")

    app.add_middleware(
        CORSMiddleware,

        # Specific origin list (not "*") for security
        # "*" would allow any website to call your API
        allow_origins=allowed_origins,

        # Allow credentials (cookies, auth headers)
        # Required if Streamlit sends auth tokens
        allow_credentials=True,

        # Allow all standard HTTP methods
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],

        # Allow all headers (including custom headers like X-Request-ID)
        # In production: restrict to known headers for security
        allow_headers=["*"],

        # How long browser can cache preflight response (in seconds)
        # 3600 = 1 hour: browser won't re-send OPTIONS every request
        max_age=3600,
    )
