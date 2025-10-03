from enum import Enum


class DistributionType(str, Enum):
    """Enumerate the supported types of probability distributions."""

    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    GAMMA = "gamma"
    WEIBULL = "weibull"
    TRUNCATED_NORMAL = "truncated_normal"
    EMPIRICAL = "empirical"


class EventType(str, Enum):
    """Enumerate the types of events that can occur in a simulation."""

    ARRIVAL = "arrival"
    DEPARTURE = "departure"
