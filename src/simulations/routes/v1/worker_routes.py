"""Worker health and status endpoints.

This module provides endpoints for monitoring Celery worker health
and status.
"""

from fastapi import APIRouter, HTTPException
from starlette import status


router = APIRouter(
    tags=["v1", "simulations", "worker"], prefix="/v1/simulations/worker"
)


@router.get("/health/worker", include_in_schema=False)
async def worker_health():
    """Check if Celery workers are responsive.

    Verify that:
    1. At least one worker is active
    2. Workers can accept tasks
    3. Redis broker is accessible

    Returns:
        Dictionary containing:
            - status: "healthy" if workers are available
            - workers: Number of active workers
            - worker_info: Detailed worker statistics
            - redis_status: Redis connection status

    Raises:
        HTTPException: 503 if no workers are active or Redis is down.
    """
    from src.simulations.worker import celery_app
    from src.simulations.routes.v1.rate_limiter import redis_client

    health = {"status": "healthy", "checks": {}}

    inspect = celery_app.control.inspect()
    stats = inspect.stats()

    if not stats:
        health["status"] = "unhealthy"
        health["checks"]["celery"] = "No active workers"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No active Celery workers",
        )

    health["checks"]["celery"] = {
        "status": "healthy",
        "workers": len(stats),
        "worker_info": stats,
    }

    try:
        redis_client.ping()
        health["checks"]["redis"] = {"status": "healthy"}
    except Exception as e:
        health["status"] = "unhealthy"
        health["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Redis broker unavailable: {str(e)}",
        )

    return health
