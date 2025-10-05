"""
Celery worker application with lifecycle management.

This module initializes the Celery worker application, configures task routing,
and sets up signal handlers for graceful startup/shutdown.
"""

import asyncio
import logging
import signal
import sys
import time
from typing import Any

from celery import Celery, Task
from celery.schedules import crontab
from celery.signals import (
    worker_ready,
    worker_shutdown,
    worker_process_init,
    worker_process_shutdown,
    task_failure,
    task_success,
    task_retry,
    task_revoked,
    task_postrun,
    task_prerun,
)
from celery.utils.log import get_task_logger

from src.simulations.config import get_worker_settings
from src.simulations.tasks.db_session import (
    get_worker_engine,
    get_worker_session_factory,
    dispose_worker_engine,
)
from src.users.models.users import *  # noqa: F401
from src.simulations.models.simulations import *  # noqa: F401
from src.background_tasks.models.background_tasks import *  # noqa: F401


logger = logging.getLogger(__name__)
task_logger = get_task_logger(__name__)

_TASK_START_TIME: dict[str, tuple[float, float]] = {}
_TASK_TIMING_TTL = 60 * 60 * 2

worker_settings = get_worker_settings()
celery_app = Celery(
    "mmcc_simulations",
    broker=worker_settings.celery_broker_url,
    backend=worker_settings.celery_result_backend,
    include=[
        "src.simulations.tasks.simulations",
        "src.simulations.tasks.cleanup",
        "src.simulations.tasks.dlq",
        "src.simulations.tasks.animation",
    ],
)

celery_app.conf.update(worker_settings.get_celery_config())

celery_app.conf.task_routes = {
    "simulations.run_simulation": {
        "queue": "simulations:10",
        "routing_key": "simulations:10",
        "priority": 5,
    },
    "simulations.generate_animation": {
        "queue": "simulations:10",
        "routing_key": "simulations:10",
    },
    "simulations.dlq": {
        "queue": "simulations_dlq",
        "routing_key": "simulations.dlq",
    },
    "simulations.cleanup_stale_reports": {
        "queue": "simulations:10",
        "routing_key": "simulations:10",
    },
}

celery_app.conf.task_default_priority = 5
celery_app.conf.task_queue_max_priority = 10

celery_app.conf.update(
    {
        "task_send_sent_event": worker_settings.task_send_sent_event,
        "worker_enable_remote_control": worker_settings.worker_enable_remote_control,
    }
)
celery_app.conf.beat_schedule = {
    "cleanup-stale-reports": {
        "task": "simulations.cleanup_stale_reports",
        "schedule": crontab(hour=2, minute=0),
    },
}


# ============================================================================
# Signal Handlers: Worker Lifecycle Management
# ============================================================================


@worker_ready.connect
def on_worker_ready(**__: Any) -> None:
    """
    Handle worker ready signal.

    This signal is sent when the worker is fully initialized and ready to
    accept tasks. Initialize database connections and log worker configuration.

    Raises:
        Exception: If worker initialization fails.
    """
    logger.info("=" * 60)
    logger.info("WORKER INITIALIZATION")
    logger.info("=" * 60)

    try:
        engine = get_worker_engine()
        get_worker_session_factory()

        logger.info("✅ Database connections initialized")
        logger.info("   Pool size: %d", engine.pool.size())
        logger.info("   Max overflow: %d", engine.pool._max_overflow)

        logger.info("🚀 Worker ready with configuration:")
        logger.info("   Concurrency: %d", worker_settings.worker_concurrency)
        logger.info("   Pool type: %s", worker_settings.worker_pool_type)
        logger.info(
            "   Prefetch multiplier: %d",
            worker_settings.worker_prefetch_multiplier,
        )
        logger.info(
            "   Max tasks per child: %d",
            worker_settings.worker_max_tasks_per_child,
        )
        logger.info("   Task time limit: %ds", worker_settings.task_time_limit)
        logger.info("   Environment: %s", worker_settings.environment)

        if worker_settings.worker_max_memory_per_child:
            logger.info(
                "   Max memory per child: %d KB",
                worker_settings.worker_max_memory_per_child,
            )

        logger.info("=" * 60)

    except Exception as e:
        logger.error("❌ Failed to initialize worker: %s", e, exc_info=True)
        raise


@worker_shutdown.connect
def on_worker_shutdown(**__: Any) -> None:
    """Handle worker shutdown signal.

    Perform cleanup tasks like closing database connections and flushing logs.
    Handle different Celery pool types (prefork, eventlet, gevent).
    """
    logger.info("=" * 60)
    logger.info("WORKER SHUTDOWN")
    logger.info("=" * 60)

    try:
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Detected running event loop, using existing context")
            loop.create_task(dispose_worker_engine())
        except RuntimeError:
            logger.debug("No running event loop, creating new one")
            asyncio.run(dispose_worker_engine())

        logger.info("✅ Database connections closed successfully")
    except Exception as e:
        logger.error(
            "⚠️  Error during worker shutdown: %s",
            e,
            exc_info=True,
        )
    finally:
        logger.info("👋 Worker shutdown complete")
        logger.info("=" * 60)


@worker_process_init.connect
def on_worker_process_init(**__: Any) -> None:
    """
    Handle worker process initialization signal.

    This signal is sent when a worker subprocess is initialized (forked).
    Reset database connections to ensure each worker has its own connection pool
    with event loops that match the worker's process.
    """
    import os

    pid = os.getpid()
    logger.info("🔧 Worker process initialized: PID=%d", pid)

    from src.simulations.tasks.db_session import reset_worker_db

    reset_worker_db()
    logger.info("🔄 Database state reset for worker process PID=%d", pid)

    import numpy as np

    np.random.seed()


@worker_process_shutdown.connect
def on_worker_process_shutdown(**__: Any) -> None:
    """
    Handle worker process shutdown signal.

    This signal is sent when a worker subprocess is shutting down.
    Useful for per-process cleanup.
    """
    import os

    pid = os.getpid()
    logger.debug("Worker process shutting down: PID=%d", pid)


# ============================================================================
# Signal Handlers: Task Lifecycle Events
# ============================================================================


@task_failure.connect
def on_task_failure(
    sender: Task | None = None,
    task_id: str | None = None,
    exception: Exception | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    *_: Any,
    **__: Any,
) -> None:
    """
    Handle task failure signal.

    This signal is sent when a task fails with an exception. Log detailed
    error information for debugging and monitoring.

    Args:
        sender: Task class that failed.
        task_id: Unique task identifier.
        exception: Exception that caused failure.
        args: Task positional arguments.
        kwargs: Task keyword arguments.
    """
    task_name = sender.name if sender else "Unknown"

    logger.error(
        "❌ Task failed: %s (ID: %s)",
        task_name,
        task_id,
        exc_info=True,
        extra={
            "task_id": task_id,
            "task_name": task_name,
            "exception": str(exception),
            "task_args": args,
            "task_kwargs": kwargs,
        },
    )


@task_success.connect
def on_task_success(
    sender: Task | None = None, result: Any = None, **__: Any
) -> None:
    """
    Handle task success signal.

    Args:
        sender: Task class that succeeded.
        result: Task return value.
    """
    if sender and hasattr(result, "get"):
        metrics = result.get("metrics", {})
        if metrics:
            task_logger.info(
                "✅ Task succeeded: %s - Processed: %d, Rejected: %d",
                sender.name,
                metrics.get("processed_requests", 0),
                metrics.get("rejected_requests", 0),
            )


@task_retry.connect
def on_task_retry(
    sender: Task | None = None,
    task_id: str | None = None,
    reason: str | None = None,
    **__: Any,
) -> None:
    """
    Handle task retry signal.

    Args:
        sender: Task class being retried.
        task_id: Unique task identifier.
        reason: Reason for retry.
    """
    task_name = sender.name if sender else "Unknown"

    logger.warning(
        "🔄 Task retrying: %s (ID: %s) - Reason: %s",
        task_name,
        task_id,
        reason,
        extra={
            "task_id": task_id,
            "task_name": task_name,
            "retry_reason": reason,
        },
    )


@task_revoked.connect
def on_task_revoked(
    sender: Task | None = None,
    request: Any = None,
    terminated: bool = False,
    signum: int | None = None,
    expired: bool = False,
    **__: Any,
) -> None:
    """
    Handle task revoked signal.

    Args:
        sender: Task class that was revoked.
        request: Task request object.
        terminated: Whether task was terminated.
        signum: Signal number if terminated.
        expired: Whether task expired.
    """
    task_name = sender.name if sender else "Unknown"
    task_id = request.id if request else "Unknown"

    reason = "expired" if expired else "terminated" if terminated else "revoked"

    logger.warning(
        "🚫 Task revoked: %s (ID: %s) - Reason: %s",
        task_name,
        task_id,
        reason,
        extra={
            "task_id": task_id,
            "task_name": task_name,
            "revoke_reason": reason,
            "signal": signum,
        },
    )


# ============================================================================
# Health Check & Utilities
# ============================================================================


def check_worker_health() -> dict[str, Any]:
    """
    Check worker health status.

    Returns:
        Dictionary with health check results.

    Example:
        >>> health = check_worker_health()
        >>> print(health["status"])
        healthy
    """
    try:
        # Check Redis connection
        inspect = celery_app.control.inspect()
        stats = inspect.stats()

        if not stats:
            return {
                "status": "unhealthy",
                "reason": "No active workers",
                "workers": 0,
            }

        engine = get_worker_engine()

        return {
            "status": "healthy",
            "workers": len(stats),
            "pool_size": engine.pool.size(),
            "environment": worker_settings.environment,
        }

    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {
            "status": "unhealthy",
            "reason": str(e),
        }


def _cleanup_old_task_timings() -> None:
    """Clean up stale task timing entries.

    Remove entries older than TTL to prevent memory leaks from
    crashed/revoked tasks that never complete.

    Note:
        Called automatically during prerun/postrun hooks.
        Should also be called periodically (e.g., every 100 tasks).
    """
    now = time.time()
    stale_keys = [
        task_id
        for task_id, (start_time, last_access) in _TASK_START_TIME.items()
        if now - last_access > _TASK_TIMING_TTL
    ]

    for task_id in stale_keys:
        _TASK_START_TIME.pop(task_id, None)

    if stale_keys:
        logger.debug("Cleaned up %d stale task timing entries", len(stale_keys))


@task_prerun.connect
def on_task_prerun(
    task_id: str | None = None,
    task: Task | None = None,
    **__: Any,
) -> None:
    """Handle task pre-run signal.

    Record task start time for duration tracking and emit start metrics.
    Automatically clean up stale entries every 100 tasks.

    Args:
        task_id: Unique task identifier.
        task: Task instance.
    """
    now = time.time()

    if task_id:
        _TASK_START_TIME[task_id] = (now, now)

    if len(_TASK_START_TIME) % 100 == 0:
        _cleanup_old_task_timings()

    task_name = task.name if task else "Unknown"
    logger.debug(
        "⏱️  Task starting: %s (ID: %s)",
        task_name,
        task_id,
        extra={
            "task_id": task_id,
            "task_name": task_name,
        },
    )


@task_postrun.connect
def on_task_postrun(
    task_id: str | None = None,
    task: Task | None = None,
    state: str | None = None,
    **__: Any,
) -> None:
    """Handle task post-run signal.

    Calculate task duration and clean up tracking data.

    Args:
        task_id: Unique task identifier.
        task: Task instance.
        state: Final task state (SUCCESS, FAILURE, etc).
    """
    if task_id and task_id in _TASK_START_TIME:
        start_time, _ = _TASK_START_TIME.pop(task_id)
        duration = time.time() - start_time

        task_name = task.name if task else "Unknown"
        logger.info(
            "⏱️  Task completed: %s (ID: %s) in %.2fs - State: %s",
            task_name,
            task_id,
            duration,
            state,
            extra={
                "task_id": task_id,
                "task_name": task_name,
                "duration": duration,
                "state": state,
            },
        )


@task_revoked.connect
def on_task_revoked(
    request: Any = None,
    **__: Any,
) -> None:
    """Handle task revoked signal.

    Clean up timing data for revoked tasks to prevent memory leaks.

    Args:
        request: Task request object.
    """
    task_id = request.id if request else None
    if task_id and task_id in _TASK_START_TIME:
        _TASK_START_TIME.pop(task_id, None)
        logger.debug("Cleaned up timing data for revoked task %s", task_id)


# ============================================================================
# Graceful Shutdown Handler
# ============================================================================


def graceful_shutdown(signum: int, _: Any) -> None:
    """
    Handle graceful shutdown on SIGTERM/SIGINT.

    Args:
        signum: Signal number.
        _: Current stack frame.
    """
    logger.info(
        "Received shutdown signal %d, shutting down gracefully...", signum
    )

    celery_app.control.shutdown()
    sys.exit(0)


signal.signal(signal.SIGTERM, graceful_shutdown)
signal.signal(signal.SIGINT, graceful_shutdown)
