from enum import Enum


class ReportStatus(str, Enum):
    """Represent the status of a simulation report."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
