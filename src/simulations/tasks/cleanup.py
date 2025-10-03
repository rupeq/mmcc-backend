import logging
from datetime import datetime, timezone, timedelta

from src.simulations.tasks.db_session import get_worker_session
from src.simulations.db_utils.simulation_reports import (
    cleanup_stale_pending_reports as cleanup_stale_pending_reports_from_db,
)
from src.simulations.worker import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(name="simulations.cleanup_stale_reports")
async def cleanup_stale_pending_reports():
    """Clean up reports stuck in PENDING > 24h"""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    async with get_worker_session() as session:
        rowcount = await cleanup_stale_pending_reports_from_db(
            session, cutoff=cutoff
        )
        logger.info("Cleaned up %d stale reports", rowcount)
