import logging

from src.simulations.worker import celery_app


logger = logging.getLogger(__name__)


@celery_app.task(name="simulations.dlq")
def dead_letter_handler(task_id: str, args: list, kwargs: dict, exc: str):
    """Handle permanently failed tasks"""
    logger.error(
        "Task moved to DLQ: task_id=%s, exception=%s",
        task_id,
        exc,
        extra={"task_id": task_id, "args": args, "kwargs": kwargs},
    )
