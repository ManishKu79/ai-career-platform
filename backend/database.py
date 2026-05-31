# backend/database.py

# motor.motor_asyncio: async MongoDB driver built on top of PyMongo
# AsyncIOMotorClient manages a connection pool to MongoDB
from motor.motor_asyncio import AsyncIOMotorClient

# Our centralized settings object
from backend.config import settings

# typing: for type hints on return values
from typing import Optional


class DatabaseManager:
    """
    Manages the MongoDB connection lifecycle.
    
    Pattern: Single shared client instance (connection pool) created at
    application startup and closed at shutdown. This is the recommended
    Motor pattern for FastAPI applications.
    
    Motor's AsyncIOMotorClient creates a connection pool internally.
    Default pool size is 100 connections, configurable via maxPoolSize.
    """

    # Class-level variable: one client shared across all requests
    # Optional[AsyncIOMotorClient] means it can be None before initialization
    client: Optional[AsyncIOMotorClient] = None

    def connect(self):
        """
        Creates the Motor client and connection pool.
        Called once at FastAPI application startup via lifespan event.
        
        serverSelectionTimeoutMS: how long to wait for MongoDB to respond
        before raising an error — 5000ms is standard for local dev
        """
        self.client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000
        )
        # Log connection attempt (in production, use proper logging)
        print(f"Connected to MongoDB at: {settings.MONGODB_URL}")

    def disconnect(self):
        """
        Closes all connections in the pool.
        Called at FastAPI application shutdown via lifespan event.
        Prevents resource leaks when the server restarts.
        """
        if self.client is not None:
            self.client.close()
            print("MongoDB connection closed.")

    def get_db(self):
        """
        Returns the database object for the configured database name.
        
        In MongoDB, databases are created automatically on first write —
        no manual schema creation needed.
        
        Returns: AsyncIOMotorDatabase object used to access collections
        """
        return self.client[settings.MONGODB_DB_NAME]


# Single instance used application-wide
# Import this object in routers and services: from backend.database import db_manager
db_manager = DatabaseManager()


def get_database():
    """
    FastAPI dependency injection function.
    
    Usage in a route:
        async def my_route(db = Depends(get_database)):
            await db["resumes"].insert_one(doc)
    
    FastAPI calls this function and injects the return value into route handlers.
    This pattern makes it trivial to swap the real DB for a test DB in unit tests.
    """
    return db_manager.get_db()