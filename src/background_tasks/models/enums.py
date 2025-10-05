from enum import Enum


class TaskType(str, Enum):
    """Enumerate the types of background tasks."""

    SIMULATION = "simulation"
    ANIMATION = "animation"
    OPTIMIZATION = "optimization"
