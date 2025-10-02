from dataclasses import dataclass
from typing import Any

from src.simulations.core.enums import EventType


@dataclass(order=True)
class Event:
    """Discrete event for the simulation queue."""

    time: float
    event_type: EventType
    data: dict[str, Any]

    def __post_init__(self):
        if self.data is None:
            self.data = {}
