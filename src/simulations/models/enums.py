from enum import Enum


class ReportStatus(str, Enum):
    """Represent the status of a simulation report.

    These statuses are stored in the database and represent the
    final state of a simulation execution. They align with TaskStatus
    but are specific to simulation reports.

    Attributes:
        PENDING: Report created, simulation queued.
        RUNNING: Simulation is currently executing.
        COMPLETED: Simulation finished successfully.
        FAILED: Simulation encountered an error.
        CANCELLED: Simulation was cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
