

# motor type for annotations
from motor.motor_asyncio import AsyncIOMotorDatabase

# pymongo ASCENDING/DESCENDING for sort specifications
from pymongo import ASCENDING, DESCENDING

# pymongo errors for exception handling
from pymongo.errors import (
    DuplicateKeyError,
    BulkWriteError,
    OperationFailure,
)

# datetime: for TTL and age calculations
from datetime import datetime, timedelta

# typing: type annotations
from typing import List, Dict, Any, Optional, Tuple

# logging
import logging

# Our settings
from backend.config import settings

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Service layer for MongoDB operations across all collections.

    This class encapsulates all database interaction logic that
    doesn't belong in route handlers or domain services.

    Responsibilities:
    - Collection statistics and health metrics
    - Schema verification and document counting
    - Bulk write operations with error handling
    - Data cleanup and maintenance utilities
    - Index inspection and verification

    All methods receive `db` as a parameter (not as instance state)
    to support dependency injection and testability.
    """

    # Collection names — centralized to avoid string typos
    COLLECTIONS = {
        "resumes":   "resumes",
        "jobs":      "jobs",
        "scores":    "scores",
        "rankings":  "rankings",
    }

    # Expected indexes per collection for verification
    # Used by verify_indexes() to detect missing indexes
    EXPECTED_INDEXES = {
        "resumes":  [
            "idx_resumes_file_id_unique",
            "idx_resumes_status",
            "idx_resumes_upload_timestamp",
            "idx_resumes_text_search",
        ],
        "jobs": [
            "idx_jobs_job_id_unique",
            "idx_jobs_is_active",
            "idx_jobs_active_created",
        ],
        "scores": [
            "idx_scores_score_id_unique",
            "idx_scores_job_score_compound",
            "idx_scores_resume_job_unique",
        ],
        "rankings": [
            "idx_rankings_ranking_id_unique",
            "idx_rankings_job_generated",
        ],
    }

    async def get_database_stats(
        self,
        db: AsyncIOMotorDatabase
    ) -> Dict[str, Any]:
        """
        Retrieves comprehensive database statistics.

        Uses MongoDB's dbStats command which returns:
        - db:         Database name
        - collections: Number of collections
        - dataSize:    Total data size in bytes
        - storageSize: Allocated storage in bytes
        - indexes:     Total number of indexes
        - indexSize:   Total index size in bytes
        - objects:     Total document count across all collections

        These metrics are surfaced in the admin dashboard
        and used for capacity planning.

        Args:
            db: Async MongoDB database object

        Returns:
            Dict with database-level statistics
        """
        try:
            # dbStats: MongoDB server command for database statistics
            # scale=1: return sizes in bytes (default)
            # scale=1024: KB, scale=1048576: MB
            raw_stats = await db.command("dbStats", scale=1)

            return {
                "database_name":    raw_stats.get("db"),
                "collection_count": raw_stats.get("collections", 0),
                "total_documents":  raw_stats.get("objects", 0),
                "data_size_mb":     round(raw_stats.get("dataSize", 0) / 1_048_576, 2),
                "storage_size_mb":  round(raw_stats.get("storageSize", 0) / 1_048_576, 2),
                "index_count":      raw_stats.get("indexes", 0),
                "index_size_mb":    round(raw_stats.get("indexSize", 0) / 1_048_576, 2),
            }
        except OperationFailure as e:
            logger.error(f"dbStats command failed: {e}")
            return {"error": str(e)}

    async def get_collection_stats(
        self,
        db: AsyncIOMotorDatabase
    ) -> Dict[str, Dict]:
        """
        Retrieves per-collection document counts and status breakdowns.

        For resumes: counts by pipeline status
        For scores:  counts by score tier
        For jobs:    counts by active/inactive

        Uses count_documents() with different filters rather than
        aggregation — faster for simple counts, less flexible.

        Args:
            db: Async MongoDB database object

        Returns:
            Dict mapping collection_name → stats dict
        """
        stats = {}

        # ── Resumes collection stats ───────────────────────────────────
        try:
            resumes_coll = db["resumes"]

            # Total count — no filter = full collection count
            # Uses collection metadata, not a full scan
            total_resumes = await resumes_coll.count_documents({})

            # Count by pipeline status using asyncio.gather for parallelism
            # gather() runs all coroutines concurrently → 4 parallel queries
            # instead of 4 sequential queries
            import asyncio
            (uploaded, nlp_proc, skills_ext, scored) = await asyncio.gather(
                resumes_coll.count_documents({"status": "uploaded"}),
                resumes_coll.count_documents({"status": "nlp_processed"}),
                resumes_coll.count_documents({"status": "skills_extracted"}),
                resumes_coll.count_documents({"status": "scored"}),
            )

            stats["resumes"] = {
                "total": total_resumes,
                "by_status": {
                    "uploaded":         uploaded,
                    "nlp_processed":    nlp_proc,
                    "skills_extracted": skills_ext,
                    "scored":           scored,
                    "other": total_resumes - (uploaded + nlp_proc + skills_ext + scored)
                }
            }
        except Exception as e:
            stats["resumes"] = {"error": str(e)}

        # ── Jobs collection stats ──────────────────────────────────────
        try:
            jobs_coll = db["jobs"]
            total_jobs  = await jobs_coll.count_documents({})
            active_jobs = await jobs_coll.count_documents({"is_active": True})

            stats["jobs"] = {
                "total":    total_jobs,
                "active":   active_jobs,
                "inactive": total_jobs - active_jobs,
            }
        except Exception as e:
            stats["jobs"] = {"error": str(e)}

        # ── Scores collection stats ────────────────────────────────────
        try:
            scores_coll = db["scores"]
            total_scores = await scores_coll.count_documents({})

            # Scores by tier — uses indexed field final_score
            (excellent, good, fair, poor) = await asyncio.gather(
                scores_coll.count_documents({"final_score": {"$gte": 0.80}}),
                scores_coll.count_documents({"final_score": {"$gte": 0.65, "$lt": 0.80}}),
                scores_coll.count_documents({"final_score": {"$gte": 0.50, "$lt": 0.65}}),
                scores_coll.count_documents({"final_score": {"$lt": 0.50}}),
            )

            stats["scores"] = {
                "total": total_scores,
                "by_tier": {
                    "excellent": excellent,
                    "good":      good,
                    "fair":      fair,
                    "poor":      poor,
                }
            }
        except Exception as e:
            stats["scores"] = {"error": str(e)}

        # ── Rankings collection stats ──────────────────────────────────
        try:
            rankings_coll = db["rankings"]
            total_rankings = await rankings_coll.count_documents({})
            stats["rankings"] = {"total": total_rankings}
        except Exception as e:
            stats["rankings"] = {"error": str(e)}

        return stats

    async def health_check(
        self,
        db: AsyncIOMotorDatabase
    ) -> Dict[str, Any]:
        """
        Comprehensive database health check for monitoring systems.

        Checks:
        1. Ping — basic connectivity
        2. Write test — can we insert a document?
        3. Read test — can we query immediately after write?
        4. Delete test — can we clean up?
        5. Collection existence — all required collections present
        6. Index verification — all required indexes exist

        This endpoint is called by:
        - Docker health checks (HEALTHCHECK in Dockerfile)
        - Load balancer health probes
        - Monitoring systems (Datadog, Prometheus)
        - Deployment pipelines before routing traffic

        Args:
            db: Async MongoDB database object

        Returns:
            Dict with health status and check results
        """
        checks = {}
        overall_healthy = True

        # ── Check 1: Ping ─────────────────────────────────────────────
        try:
            await db.client.admin.command("ping")
            checks["ping"] = {"status": "ok", "latency_ms": None}

            # Measure ping latency
            import time
            t0 = time.monotonic()
            await db.client.admin.command("ping")
            latency_ms = round((time.monotonic() - t0) * 1000, 2)
            checks["ping"]["latency_ms"] = latency_ms

        except Exception as e:
            checks["ping"] = {"status": "failed", "error": str(e)}
            overall_healthy = False

        # ── Check 2: Write test ───────────────────────────────────────
        # Insert a small test document into a _health collection
        # Using _health avoids polluting application collections
        try:
            health_doc = {
                "_health_check": True,
                "timestamp": datetime.utcnow(),
                "check_id": "health_write_test"
            }
            result = await db["_health"].insert_one(health_doc)
            test_id = result.inserted_id
            checks["write"] = {"status": "ok", "inserted_id": str(test_id)}
        except Exception as e:
            checks["write"] = {"status": "failed", "error": str(e)}
            overall_healthy = False
            test_id = None

        # ── Check 3: Read test ────────────────────────────────────────
        if test_id:
            try:
                # Read back the document we just inserted
                # find_one by _id — uses primary key (always indexed)
                doc = await db["_health"].find_one({"_id": test_id})
                read_ok = doc is not None
                checks["read"] = {
                    "status": "ok" if read_ok else "failed",
                    "document_found": read_ok
                }
                if not read_ok:
                    overall_healthy = False
            except Exception as e:
                checks["read"] = {"status": "failed", "error": str(e)}
                overall_healthy = False

        # ── Check 4: Delete test ──────────────────────────────────────
        if test_id:
            try:
                # Clean up health check document
                await db["_health"].delete_one({"_id": test_id})
                checks["delete"] = {"status": "ok"}
            except Exception as e:
                checks["delete"] = {"status": "failed", "error": str(e)}

        # ── Check 5: Collection existence ─────────────────────────────
        try:
            existing = await db.list_collection_names()
            required  = set(self.COLLECTIONS.values())
            missing   = required - set(existing)

            checks["collections"] = {
                "status":   "ok" if not missing else "warning",
                "existing": list(existing),
                "missing":  list(missing),
            }
            # Missing collections are warnings, not failures
            # (they're created on first write)
        except Exception as e:
            checks["collections"] = {"status": "failed", "error": str(e)}

        # ── Check 6: Index verification ───────────────────────────────
        index_status = await self.verify_indexes(db)
        checks["indexes"] = index_status
        if index_status.get("missing_indexes"):
            overall_healthy = False

        return {
            "status":    "healthy" if overall_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "database":  settings.MONGODB_DB_NAME,
            "checks":    checks,
        }

    async def verify_indexes(
        self,
        db: AsyncIOMotorDatabase
    ) -> Dict[str, Any]:
        """
        Verifies all required indexes exist on each collection.

        Fetches the index list from MongoDB's index catalog for each
        collection and compares against EXPECTED_INDEXES.

        Missing indexes don't break the application but cause
        full collection scans → slow queries → degraded performance.

        Args:
            db: Async MongoDB database object

        Returns:
            Dict with verification results per collection
        """
        results = {}
        all_missing = []

        for collection_name, expected in self.EXPECTED_INDEXES.items():
            try:
                collection = db[collection_name]

                # list_indexes() returns cursor over index documents
                # Each document: {name, key, unique, sparse, ...}
                indexes_cursor = collection.list_indexes()
                existing_names = []

                async for index_doc in indexes_cursor:
                    existing_names.append(index_doc["name"])

                # Check which expected indexes are missing
                missing = [
                    idx for idx in expected
                    if idx not in existing_names
                ]

                results[collection_name] = {
                    "existing_count": len(existing_names),
                    "existing_names": existing_names,
                    "missing":        missing,
                    "status":         "ok" if not missing else "missing_indexes",
                }
                all_missing.extend(missing)

            except Exception as e:
                results[collection_name] = {
                    "status": "error",
                    "error": str(e)
                }

        return {
            "collections": results,
            "missing_indexes": all_missing,
            "all_indexes_present": len(all_missing) == 0,
        }

    async def upsert_score(
        self,
        db: AsyncIOMotorDatabase,
        score_dict: Dict[str, Any]
    ) -> bool:
        """
        Upserts an ATS score document.

        Uses update_one with upsert=True:
        - If (resume_file_id, job_id) pair exists → update in place
        - If pair doesn't exist → insert new document

        This prevents duplicate scores when the scoring endpoint
        is called multiple times for the same resume-job pair.

        $set operator: updates only specified fields
        $setOnInsert: sets fields only on INSERT (not on update)
        Combining both: always update score fields, but only set
        score_id and scored_at on first insert.

        Args:
            db:         Async MongoDB database object
            score_dict: Score document dict from ATSScoreResult

        Returns:
            True if operation succeeded, False otherwise
        """
        try:
            filter_query = {
                "resume_file_id": score_dict["resume_file_id"],
                "job_id":         score_dict["job_id"],
            }

            update_doc = {
                # Always update these fields (score may have changed)
                "$set": {
                    "final_score":         score_dict["final_score"],
                    "final_score_percent": score_dict["final_score_percent"],
                    "component_scores":    score_dict["component_scores"],
                    "skill_gap":           score_dict["skill_gap"],
                    "score_tier":          score_dict["score_tier"],
                    "recommendation":      score_dict["recommendation"],
                    "passes_threshold":    score_dict["passes_threshold"],
                    "matched_keywords":    score_dict.get("matched_keywords", []),
                    "missing_keywords":    score_dict.get("missing_keywords", []),
                    "top_matching_terms":  score_dict.get("top_matching_terms", []),
                    "scored_at":           datetime.utcnow(),
                },
                # Only set these on first insert
                "$setOnInsert": {
                    "score_id": score_dict.get(
                        "score_id",
                        str(__import__('uuid').uuid4())
                    ),
                }
            }

            result = await db["scores"].update_one(
                filter_query,
                update_doc,
                upsert=True  # Insert if not found, update if found
            )

            # upserted_id is set when a new document was inserted
            # modified_count > 0 when an existing document was updated
            was_inserted = result.upserted_id is not None
            was_updated  = result.modified_count > 0

            logger.debug(
                f"Score upsert: inserted={was_inserted}, updated={was_updated}"
            )
            return True

        except DuplicateKeyError as e:
            # Shouldn't happen with upsert=True, but handle defensively
            logger.warning(f"Duplicate score key: {e}")
            return False
        except Exception as e:
            logger.error(f"Score upsert failed: {e}")
            return False

    async def bulk_upsert_scores(
        self,
        db: AsyncIOMotorDatabase,
        score_dicts: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Efficiently upserts multiple score documents in one operation.

        Uses pymongo's bulk_write() with UpdateOne(upsert=True) operations.
        Batch size: MongoDB recommends max 1000 operations per bulk write.

        Bulk write is significantly faster than N individual upserts:
        - N=100 individual: 100 round trips × 5ms = 500ms
        - N=100 bulk:       1 round trip × 10ms  = 10ms

        Args:
            db:          Async MongoDB database object
            score_dicts: List of score document dicts

        Returns:
            Dict with counts: inserted, modified, errors
        """
        from pymongo import UpdateOne

        if not score_dicts:
            return {"inserted": 0, "modified": 0, "errors": 0}

        operations = []
        for score in score_dicts:
            filter_q = {
                "resume_file_id": score["resume_file_id"],
                "job_id":         score["job_id"],
            }
            update_doc = {
                "$set": {
                    "final_score":         score["final_score"],
                    "final_score_percent": score["final_score_percent"],
                    "component_scores":    score["component_scores"],
                    "skill_gap":           score["skill_gap"],
                    "score_tier":          score["score_tier"],
                    "passes_threshold":    score["passes_threshold"],
                    "scored_at":           datetime.utcnow().isoformat(),
                },
                "$setOnInsert": {
                    "score_id": score.get("score_id", str(__import__('uuid').uuid4()))
                }
            }
            operations.append(UpdateOne(filter_q, update_doc, upsert=True))

        try:
            # ordered=False: continue on error (don't stop at first failure)
            # This is important for bulk operations — one bad score
            # shouldn't prevent all others from being saved
            result = await db["scores"].bulk_write(
                operations,
                ordered=False
            )

            return {
                "inserted": result.upserted_count,
                "modified": result.modified_count,
                "errors":   0,
            }

        except BulkWriteError as e:
            # BulkWriteError contains details about which ops failed
            write_errors = e.details.get("writeErrors", [])
            logger.error(
                f"Bulk write partial failure: "
                f"{len(write_errors)} errors in {len(score_dicts)} operations"
            )
            return {
                "inserted": e.details.get("nUpserted", 0),
                "modified": e.details.get("nModified", 0),
                "errors":   len(write_errors),
            }

    async def cleanup_old_scores(
        self,
        db: AsyncIOMotorDatabase,
        days_old: int = 90
    ) -> int:
        """
        Deletes score documents older than `days_old` days.

        Note: The TTL index on scored_at handles this automatically.
        This method provides manual cleanup capability for:
        - Immediate cleanup before TTL runs (TTL checks every 60s)
        - Cleaning up with a different age threshold
        - Administrative data management

        $lt operator: less than (strictly)
        datetime.utcnow() - timedelta(days=days_old): cutoff timestamp

        Args:
            db:       Async MongoDB database object
            days_old: Delete scores older than this many days

        Returns:
            Number of documents deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days_old)

        result = await db["scores"].delete_many({
            # $lt: scored_at is less than (older than) cutoff
            "scored_at": {"$lt": cutoff.isoformat()}
        })

        deleted = result.deleted_count
        logger.info(
            f"Cleanup: deleted {deleted} scores older than {days_old} days "
            f"(cutoff: {cutoff.isoformat()})"
        )
        return deleted

    async def get_resume_pipeline_status(
        self,
        db: AsyncIOMotorDatabase
    ) -> List[Dict]:
        """
        Returns all resumes with their current pipeline status.
        Used for monitoring and debugging the processing pipeline.

        Returns lightweight projection — no text fields.

        Args:
            db: Async MongoDB database object

        Returns:
            List of dicts with file_id, status, candidate_name, timestamps
        """
        cursor = db["resumes"].find(
            {},
            {
                "_id": 0,
                "file_id": 1,
                "candidate_name": 1,
                "status": 1,
                "upload_timestamp": 1,
                "metadata.filename": 1,
                "metadata.file_type": 1,
                "metadata.word_count": 1,
            }
        ).sort("upload_timestamp", DESCENDING).limit(200)

        return await cursor.to_list(length=200)


# Single shared instance
db_service = DatabaseService()
