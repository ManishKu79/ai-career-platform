# backend/database.py

# motor.motor_asyncio: async MongoDB driver for Python
# AsyncIOMotorClient: manages an async connection pool to MongoDB
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

# pymongo: sync MongoDB driver — used for index management utilities
# IndexModel: represents an index specification
# ASCENDING / DESCENDING: sort order constants (1 and -1)
from pymongo import IndexModel, ASCENDING, DESCENDING

# pymongo.errors: MongoDB-specific exception classes
from pymongo.errors import (
    ConnectionFailure,       # Cannot connect to MongoDB
    ServerSelectionTimeoutError,  # Timeout waiting for server
    OperationFailure,        # DB operation failed (auth, permissions)
)

# Our settings object
from backend.config import settings

# typing: for type hints
from typing import Optional, List

# logging: structured application logging
import logging

# asyncio: for async operations
import asyncio

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Production-grade MongoDB connection manager.

    Responsibilities:
    1. Connection pool management (connect/disconnect lifecycle)
    2. Index initialization at startup
    3. Database health verification
    4. Dependency injection interface for FastAPI

    Connection Pool Configuration:
    - maxPoolSize: max concurrent connections (100 default)
    - minPoolSize: min maintained connections (10 for warm pool)
    - maxIdleTimeMS: close idle connections after N milliseconds
    - connectTimeoutMS: timeout for initial connection
    - serverSelectionTimeoutMS: timeout for server selection
    - retryWrites: automatically retry failed write operations

    These values are calibrated for a single-server dev/staging setup.
    Production clusters use higher maxPoolSize and replica set URIs.
    """

    # Class-level client — one pool shared across all requests
    client: Optional[AsyncIOMotorClient] = None

    def connect(self):
        """
        Creates Motor client with production connection pool settings.
        Called once at FastAPI application startup via lifespan event.

        Motor creates the connection pool lazily — actual TCP connections
        are established on first database operation, not here.
        This means connect() is fast (~1ms) and the first DB operation
        pays the TCP handshake cost (~5-50ms).
        """
        logger.info(f"Initializing MongoDB connection pool → {settings.MONGODB_URL}")

        self.client = AsyncIOMotorClient(
            settings.MONGODB_URL,

            # Maximum connections in pool
            # Each concurrent async request uses one connection
            # 100 supports 100 simultaneous DB operations
            maxPoolSize=100,

            # Minimum maintained connections
            # Keep 10 warm to avoid cold-start latency on traffic spikes
            minPoolSize=10,

            # Close connections idle longer than 30 seconds
            # Prevents stale connections after MongoDB restarts
            maxIdleTimeMS=30_000,

            # Timeout for establishing a new connection
            # 10 seconds is generous for LAN; reduce to 5s for cloud
            connectTimeoutMS=10_000,

            # How long to wait for a suitable server before failing
            # 5 seconds for dev; use 30s for replica sets with failover
            serverSelectionTimeoutMS=5_000,

            # Automatically retry failed writes once
            # Handles transient network errors transparently
            retryWrites=True,

            # Automatically retry failed reads once
            retryReads=True,

            # Application name: appears in MongoDB logs and Atlas metrics
            # Makes it easy to identify which app is generating queries
            appName="ai-career-platform",
        )

        logger.info("MongoDB client initialized. Pool size: 10-100 connections.")

    def disconnect(self):
        """
        Closes all connections in the pool gracefully.
        Called at FastAPI application shutdown.

        Motor waits for in-flight operations to complete before
        closing connections — prevents data corruption on shutdown.
        """
        if self.client is not None:
            self.client.close()
            logger.info("MongoDB connection pool closed.")
        self.client = None

    def get_db(self) -> AsyncIOMotorDatabase:
        """
        Returns the database object by name.

        MongoDB creates databases lazily — if the database doesn't
        exist, it will be created on first write operation.
        No manual database creation is needed.

        Returns:
            AsyncIOMotorDatabase: Database object for collection access
        """
        if self.client is None:
            raise RuntimeError(
                "Database not connected. "
                "Ensure db_manager.connect() was called at startup."
            )
        return self.client[settings.MONGODB_DB_NAME]

    async def ping(self) -> bool:
        """
        Verifies MongoDB connectivity by sending a ping command.

        Uses the admin database ping command — the lightest possible
        operation that confirms the server is reachable and responsive.

        Returns:
            True if ping succeeds, False if MongoDB unreachable
        """
        try:
            db = self.get_db()
            # admin.command("ping") sends: { ping: 1 }
            # Server responds: { ok: 1.0 }
            await db.client.admin.command("ping")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"MongoDB ping failed: {e}")
            return False

    async def initialize_indexes(self):
        """
        Creates all required indexes for optimal query performance.

        Index creation strategy:
        - All indexes use background=True in production
          (doesn't block reads/writes during index build on existing data)
        - Unique indexes enforce data integrity at DB level
        - Compound indexes support multi-field query patterns
        - TTL indexes automatically expire old analytics data
        - Text indexes enable full-text search

        Idempotent: calling multiple times is safe —
        MongoDB skips creation if index already exists with same spec.

        Called once at application startup after connect().
        Index builds on empty collections are instant.
        Index builds on populated collections run in background.
        """
        db = self.get_db()

        logger.info("Initializing MongoDB indexes...")

        # ── resumes collection indexes ─────────────────────────────────
        await self._create_resumes_indexes(db)

        # ── jobs collection indexes ────────────────────────────────────
        await self._create_jobs_indexes(db)

        # ── scores collection indexes ──────────────────────────────────
        await self._create_scores_indexes(db)

        # ── rankings collection indexes ────────────────────────────────
        await self._create_rankings_indexes(db)

        logger.info("All indexes initialized successfully.")

    async def _create_resumes_indexes(self, db: AsyncIOMotorDatabase):
        """
        Creates indexes for the resumes collection.

        Query patterns supported:
        1. Find by file_id (primary lookup)       → unique single-field
        2. Filter by status (pipeline stage)      → single-field
        3. Sort by upload_timestamp               → single-field
        4. Full-text search on cleaned_text       → text index
        5. Find by candidate email                → single-field
        """
        collection = db["resumes"]

        indexes = [
            # Primary lookup index — unique enforces no duplicate uploads
            # unique=True: MongoDB rejects inserts with duplicate file_id
            IndexModel(
                [("file_id", ASCENDING)],
                unique=True,
                name="idx_resumes_file_id_unique",
                # background=True: build index without blocking collection
                background=True,
            ),

            # Status-based filtering for pipeline stage queries
            # "Find all resumes with status='nlp_processed'"
            IndexModel(
                [("status", ASCENDING)],
                name="idx_resumes_status",
                background=True,
            ),

            # Chronological listing — newest first in dashboard
            # DESCENDING: index stores values high-to-low for efficient DESC sort
            IndexModel(
                [("upload_timestamp", DESCENDING)],
                name="idx_resumes_upload_timestamp",
                background=True,
            ),

            # Email lookup for candidate deduplication
            # sparse=True: only indexes docs where field exists
            # (not all resumes have extractable email)
            IndexModel(
                [("candidate_email", ASCENDING)],
                name="idx_resumes_email",
                sparse=True,
                background=True,
            ),

            # Compound: status + timestamp for paginated pipeline queries
            # "Find all uploaded resumes, newest first"
            # This compound index satisfies both filter AND sort efficiently
            IndexModel(
                [("status", ASCENDING), ("upload_timestamp", DESCENDING)],
                name="idx_resumes_status_timestamp",
                background=True,
            ),

            # Text index for full-text search on resume content
            # Allows: db.resumes.find({$text: {$search: "machine learning"}})
            # MongoDB supports one text index per collection
            # weights: higher weight = higher relevance in text search ranking
            IndexModel(
                [
                    ("cleaned_text", "text"),
                    ("candidate_name", "text"),
                ],
                weights={
                    "candidate_name": 10,  # Name matches rank higher
                    "cleaned_text":   1,   # Content matches rank lower
                },
                name="idx_resumes_text_search",
                background=True,
            ),
        ]

        try:
            result = await collection.create_indexes(indexes)
            logger.info(f"Resumes indexes created: {result}")
        except OperationFailure as e:
            # OperationFailure with code 85 = index already exists with same name
            # This is expected on restart — not an error
            logger.warning(f"Resume index creation warning (may already exist): {e}")

    async def _create_jobs_indexes(self, db: AsyncIOMotorDatabase):
        """
        Creates indexes for the jobs collection.

        Query patterns:
        1. Find by job_id (primary lookup)
        2. Filter active jobs
        3. Search by title (text)
        4. Filter by company
        """
        collection = db["jobs"]

        indexes = [
            # Primary job lookup
            IndexModel(
                [("job_id", ASCENDING)],
                unique=True,
                name="idx_jobs_job_id_unique",
                background=True,
            ),

            # Active job filter — most queries use is_active=True
            IndexModel(
                [("is_active", ASCENDING)],
                name="idx_jobs_is_active",
                background=True,
            ),

            # Compound: active + created (paginated job listings)
            IndexModel(
                [("is_active", ASCENDING), ("created_at", DESCENDING)],
                name="idx_jobs_active_created",
                background=True,
            ),

            # Company filter for multi-tenant or multi-company deployments
            IndexModel(
                [("company", ASCENDING)],
                sparse=True,
                name="idx_jobs_company",
                background=True,
            ),

            # Text search on job title and description
            IndexModel(
                [("title", "text"), ("description", "text")],
                weights={"title": 5, "description": 1},
                name="idx_jobs_text_search",
                background=True,
            ),
        ]

        try:
            result = await collection.create_indexes(indexes)
            logger.info(f"Jobs indexes created: {result}")
        except OperationFailure as e:
            logger.warning(f"Jobs index creation warning: {e}")

    async def _create_scores_indexes(self, db: AsyncIOMotorDatabase):
        """
        Creates indexes for the scores collection.

        This is the most query-intensive collection.
        Query patterns:
        1. Find scores for a job (leaderboard)
        2. Find scores for a resume (candidate history)
        3. Ranked listing: job_id + final_score DESC
        4. Deduplication: (resume_id, job_id) pair
        """
        collection = db["scores"]

        indexes = [
            # Primary score lookup by score_id
            IndexModel(
                [("score_id", ASCENDING)],
                unique=True,
                name="idx_scores_score_id_unique",
                background=True,
            ),

            # All scores for a given job (leaderboard query)
            IndexModel(
                [("job_id", ASCENDING)],
                name="idx_scores_job_id",
                background=True,
            ),

            # All scores for a given resume (candidate history)
            IndexModel(
                [("resume_file_id", ASCENDING)],
                name="idx_scores_resume_file_id",
                background=True,
            ),

            # Most critical index: job leaderboard sorted by score
            # Compound: filter by job_id, sort by final_score DESC
            # MongoDB uses this single index for BOTH filter and sort
            # Without: filter uses idx_scores_job_id, sort needs in-memory sort
            # With: single B-tree traversal satisfies filter + sort
            IndexModel(
                [("job_id", ASCENDING), ("final_score", DESCENDING)],
                name="idx_scores_job_score_compound",
                background=True,
            ),

            # Deduplication check: each (resume, job) pair should be unique
            # unique=True prevents storing multiple scores for same pair
            # This enforces: one score record per resume-job combination
            IndexModel(
                [
                    ("resume_file_id", ASCENDING),
                    ("job_id", ASCENDING)
                ],
                unique=True,
                name="idx_scores_resume_job_unique",
                background=True,
            ),

            # TTL index: automatically delete scores older than 90 days
            # Prevents unbounded collection growth
            # expireAfterSeconds: MongoDB runs a background thread every 60s
            # that deletes documents where scored_at < now - 90 days
            IndexModel(
                [("scored_at", ASCENDING)],
                name="idx_scores_ttl_90days",
                expireAfterSeconds=90 * 24 * 60 * 60,  # 90 days in seconds
                background=True,
            ),
        ]

        try:
            result = await collection.create_indexes(indexes)
            logger.info(f"Scores indexes created: {result}")
        except OperationFailure as e:
            logger.warning(f"Scores index creation warning: {e}")

    async def _create_rankings_indexes(self, db: AsyncIOMotorDatabase):
        """
        Creates indexes for the rankings collection.

        Query patterns:
        1. Most recent ranking for a job
        2. Ranking by ranking_id
        """
        collection = db["rankings"]

        indexes = [
            IndexModel(
                [("ranking_id", ASCENDING)],
                unique=True,
                name="idx_rankings_ranking_id_unique",
                background=True,
            ),

            # Most common query: latest ranking for a job
            # Compound supports: filter job_id + sort generated_at DESC
            IndexModel(
                [("job_id", ASCENDING), ("generated_at", DESCENDING)],
                name="idx_rankings_job_generated",
                background=True,
            ),
        ]

        try:
            result = await collection.create_indexes(indexes)
            logger.info(f"Rankings indexes created: {result}")
        except OperationFailure as e:
            logger.warning(f"Rankings index creation warning: {e}")


# ── Singleton instance ────────────────────────────────────────────────
db_manager = DatabaseManager()


def get_database() -> AsyncIOMotorDatabase:
    """
    FastAPI dependency injection function.

    Usage in route handlers:
        async def my_route(db = Depends(get_database)):
            await db["resumes"].find_one(...)

    FastAPI calls this function before the route handler executes,
    injects the return value as the `db` parameter.
    """
    return db_manager.get_db()