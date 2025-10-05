import uuid
from typing import Any

from src.background_tasks.routes.v1.exceptions import InvalidSubjectID


def verify_subject_ids(subject_ids: list[Any] | None) -> list[uuid.UUID] | None:
    """Validate and normalize subject IDs into UUIDs.

    Convert each provided subject ID to a UUID instance. Return None if
    no IDs are provided.

    Args:
        subject_ids: Optional list of subject identifiers (UUIDs or
            UUID-like strings).

    Returns:
        List of verified UUIDs or None if input is None.

    Raises:
        InvalidSubjectID: If any subject ID is not a valid UUID.
    """
    if subject_ids is None:
        return None

    verified = []

    for subject_id in subject_ids:
        try:
            verified_subject_id = uuid.UUID(subject_id)
            verified.append(verified_subject_id)
        except ValueError:
            raise InvalidSubjectID

    return verified


def get_status_message(state: str) -> str:
    """Map a Celery task state to a human-readable message.

    Args:
        state: Celery task state (e.g., PENDING, STARTED, SUCCESS).

    Returns:
        Human-readable status message corresponding to the state.
    """
    status_messages = {
        "PENDING": "Task is queued",
        "STARTED": "Task is running",
        "PROGRESS": "Task is running",
        "SUCCESS": "Task completed successfully",
        "FAILURE": "Task failed",
        "RETRY": "Task is being retried",
        "REVOKED": "Task was cancelled",
    }
    return status_messages.get(state, f"Unknown state: {state}")
