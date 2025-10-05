from enum import Enum


class TaskType(str, Enum):
    """Enumerate the types of background tasks.

    Attributes:
        SIMULATION: Simulation execution task.
        ANIMATION: Animation generation task.
        OPTIMIZATION: Channel optimization task.
    """

    SIMULATION = "simulation"
    ANIMATION = "animation"
    OPTIMIZATION = "optimization"


class TaskStatus(str, Enum):
    """Enumerate the standardized status values for background tasks.

    These statuses abstract away Celery-specific states and provide
    a consistent interface for the frontend.

    Attributes:
        PENDING: Task is queued and waiting to be executed.
        RUNNING: Task is currently being executed by a worker.
        SUCCESS: Task completed successfully.
        FAILED: Task failed with an error.
        RETRY: Task failed but is being retried.
        CANCELLED: Task was cancelled by user or system.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"
    CANCELLED = "cancelled"
