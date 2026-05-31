# backend/config.py

# python-dotenv: reads key=value pairs from .env file into os.environ
from dotenv import load_dotenv

# pydantic_settings: provides BaseSettings class that auto-reads from environment
# variables and validates their types
from pydantic_settings import BaseSettings

# os: needed to construct file paths
import os

# Load the .env file into environment variables before Settings class reads them
# This must happen before BaseSettings is instantiated
load_dotenv()


class Settings(BaseSettings):
    """
    Application-wide configuration loaded from environment variables.
    
    Pydantic BaseSettings automatically reads each field from:
    1. Environment variables (highest priority)
    2. .env file (via load_dotenv above)
    3. Default values defined here (lowest priority)
    
    Type annotations enforce that values are cast to the correct type.
    For example, API_PORT="8000" from .env becomes int 8000 automatically.
    """

    # MongoDB connection string
    # Format: mongodb://host:port for local, mongodb+srv://... for Atlas
    MONGODB_URL: str = "mongodb://localhost:27017"

    # Name of the MongoDB database to use
    MONGODB_DB_NAME: str = "career_platform"

    # FastAPI server host — 0.0.0.0 means listen on all network interfaces
    API_HOST: str = "0.0.0.0"

    # FastAPI server port
    API_PORT: int = 8000

    # Hot reload: restarts server when code changes (development only)
    API_RELOAD: bool = True

    # Streamlit port for dashboard
    STREAMLIT_PORT: int = 8501

    # Base URL the Streamlit frontend uses to call the FastAPI backend
    API_BASE_URL: str = "http://localhost:8000"

    # Which spaCy model to load — en_core_web_lg is large English model
    SPACY_MODEL: str = "en_core_web_lg"

    # Maximum allowed resume file size in megabytes
    MAX_FILE_SIZE_MB: int = 10

    # Minimum ATS score (0.0 to 1.0) to consider a candidate
    ATS_SCORE_THRESHOLD: float = 0.5

    # Maximum candidates to show in dashboard rankings
    MAX_CANDIDATES_DISPLAY: int = 50

    class Config:
        # Tell Pydantic where to find the .env file
        env_file = ".env"

        # Allow extra fields in .env without raising validation errors
        extra = "ignore"


# Instantiate a single settings object to be imported across the application
# Using a single instance avoids re-reading the file on every import
settings = Settings()