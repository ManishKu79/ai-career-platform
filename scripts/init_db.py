# scripts/init_db.py

"""
Database initialization script.
Run before first deployment:
    python scripts/init_db.py

Also called from Docker entrypoint to ensure DB is ready.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import db_manager
from backend.services.db_service import db_service


async def initialize():
    """Run full database initialization sequence."""
    print("=" * 60)
    print("AI Career Intelligence Platform — Database Initialization")
    print("=" * 60)

    # Step 1: Connect
    print("\n[1/4] Connecting to MongoDB...")
    db_manager.connect()

    # Step 2: Ping
    print("[2/4] Verifying connectivity...")
    db = db_manager.get_db()
    is_healthy = await db_manager.ping()

    if not is_healthy:
        print("ERROR: Cannot connect to MongoDB.")
        print("Ensure MongoDB is running at:", end=" ")
        from backend.config import settings
        print(settings.MONGODB_URL)
        sys.exit(1)

    print("      MongoDB connection: OK")

    # Step 3: Create indexes
    print("[3/4] Creating indexes...")
    await db_manager.initialize_indexes()
    print("      Indexes created: OK")

    # Step 4: Verify
    print("[4/4] Verifying indexes...")
    verification = await db_service.verify_indexes(db)

    if verification["all_indexes_present"]:
        print("      Index verification: OK")
    else:
        print(f"WARNING: Missing indexes: {verification['missing_indexes']}")

    # Final stats
    stats = await db_service.get_database_stats(db)
    print("\n" + "=" * 60)
    print("Database Initialization Complete")
    print(f"  Database:    {stats.get('database_name')}")
    print(f"  Collections: {stats.get('collection_count')}")
    print(f"  Documents:   {stats.get('total_documents')}")
    print(f"  Indexes:     {stats.get('index_count')}")
    print(f"  Data Size:   {stats.get('data_size_mb')} MB")
    print("=" * 60)

    db_manager.disconnect()


if __name__ == "__main__":
    asyncio.run(initialize())