from dataclasses import dataclass, field
import logging

from src.simulations.core.schemas import GanttChartItem

logger = logging.getLogger(__name__)


@dataclass
class SimulationStatistics:
    """Collects simulation statistics with configurable memory limits"""

    # Basic counters
    total_requests: int = 0
    processed_requests: int = 0
    rejected_requests: int = 0
    total_busy_time: float = 0.0

    # Optional data collection with limits
    service_times: list[float] = field(default_factory=list)
    gantt_items: list[GanttChartItem] = field(default_factory=list)

    # Welford's online algorithm for mean/variance
    _count: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    # Configuration limits
    _max_service_samples: int | None = field(default=1000, init=False)
    _max_gantt_items: int | None = field(default=10000, init=False)
    _collect_service_times: bool = field(default=True, init=False)
    _collect_gantt_data: bool = field(default=True, init=False)

    # Tracking for truncation warnings
    _service_times_truncated: bool = field(default=False, init=False)
    _gantt_items_truncated: bool = field(default=False, init=False)

    def configure_limits(
        self,
        *,
        max_service_samples: int | None = 1000,
        max_gantt_items: int | None = 10000,
        collect_service_times: bool = True,
        collect_gantt_data: bool = True,
    ) -> None:
        """Configure collection limits before simulation starts"""
        self._max_service_samples = max_service_samples
        self._max_gantt_items = max_gantt_items
        self._collect_service_times = collect_service_times
        self._collect_gantt_data = collect_gantt_data

        logger.debug(
            "Statistics configured: service_times=%s (max=%s), gantt=%s (max=%s)",
            collect_service_times,
            max_service_samples,
            collect_gantt_data,
            max_gantt_items,
        )

    def record_service_time(self, service_time: float) -> None:
        """Record service time with online mean/variance and optional storage"""
        # Always update online statistics
        self._count += 1
        delta = service_time - self._mean
        self._mean += delta / self._count
        delta2 = service_time - self._mean
        self._m2 += delta * delta2

        # Optionally store raw samples
        if not self._collect_service_times:
            return

        if self._max_service_samples is None:
            # Unlimited collection
            self.service_times.append(service_time)
        elif self._max_service_samples == 0:
            # Collection disabled
            pass
        elif len(self.service_times) < self._max_service_samples:
            # Still under limit
            self.service_times.append(service_time)
        elif not self._service_times_truncated:
            # First truncation
            self._service_times_truncated = True
            logger.debug(
                "Service time collection limit reached (%d samples). "
                "Further samples will not be stored.",
                self._max_service_samples,
            )

    def record_gantt_item(self, item: GanttChartItem) -> None:
        """Record Gantt chart item with optional limit"""
        if not self._collect_gantt_data:
            return

        if self._max_gantt_items is None:
            # Unlimited collection
            self.gantt_items.append(item)
        elif self._max_gantt_items == 0:
            # Collection disabled
            pass
        elif len(self.gantt_items) < self._max_gantt_items:
            # Still under limit
            self.gantt_items.append(item)
        elif not self._gantt_items_truncated:
            # First truncation
            self._gantt_items_truncated = True
            logger.debug(
                "Gantt chart collection limit reached (%d items). "
                "Further items will not be stored.",
                self._max_gantt_items,
            )

    @property
    def service_time_mean(self) -> float:
        """Get mean service time (always available via online calculation)"""
        return self._mean

    @property
    def service_time_std(self) -> float:
        """Get standard deviation of service times"""
        if self._count < 2:
            return 0.0
        return (self._m2 / (self._count - 1)) ** 0.5

    @property
    def service_time_variance(self) -> float:
        """Get variance of service times"""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def is_service_times_truncated(self) -> bool:
        """Check if service time collection was truncated"""
        return self._service_times_truncated

    @property
    def is_gantt_truncated(self) -> bool:
        """Check if Gantt collection was truncated"""
        return self._gantt_items_truncated

    def get_collection_info(self) -> dict:
        """Get information about data collection status"""
        return {
            "service_times_collected": len(self.service_times),
            "service_times_truncated": self._service_times_truncated,
            "service_times_limit": self._max_service_samples,
            "gantt_items_collected": len(self.gantt_items),
            "gantt_items_truncated": self._gantt_items_truncated,
            "gantt_items_limit": self._max_gantt_items,
            "total_service_time_count": self._count,
        }
