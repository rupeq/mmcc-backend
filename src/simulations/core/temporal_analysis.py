"""Temporal analysis module for tracking time-series metrics."""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.simulations.core.schemas import (
    BusyIdlePeriod,
    BusyIdleStatistics,
    OccupancySnapshot,
    PeakPeriod,
    PhaseMetrics,
    TemporalMetrics,
    TemporalProfile,
    TimeWindow,
)

logger = logging.getLogger(__name__)


@dataclass
class TemporalCollector:
    """Collect temporal statistics during simulation."""

    num_channels: int
    simulation_time: float
    window_size: float = 10.0  # Time window for aggregation
    snapshot_interval: float = 1.0  # Occupancy sampling interval

    # Time-series data
    window_data: list[dict[str, Any]] = field(default_factory=list)
    occupancy_snapshots: list[tuple[float, int]] = field(default_factory=list)

    # Phase tracking (for non-stationary)
    phase_data: list[dict[str, Any]] = field(default_factory=list)
    current_phase: dict[str, Any] | None = None

    # Channel state tracking
    channel_states: dict[int, list[BusyIdlePeriod]] = field(
        default_factory=lambda: defaultdict(list)
    )
    channel_last_state_change: dict[int, tuple[float, str]] = field(
        default_factory=dict
    )

    # Current window accumulators
    _current_window_start: float = 0.0
    _current_window_arrivals: int = 0
    _current_window_processed: int = 0
    _current_window_rejected: int = 0
    _current_window_busy_time: dict[int, float] = field(default_factory=dict)

    # Last snapshot time
    _last_snapshot_time: float = 0.0

    def __post_init__(self):
        """Initialize tracking structures."""
        for channel_id in range(self.num_channels):
            self.channel_last_state_change[channel_id] = (0.0, "idle")
            self._current_window_busy_time[channel_id] = 0.0

    def record_arrival(self, time: float) -> None:
        """Record an arrival event."""
        self._check_window_boundary(time)
        self._current_window_arrivals += 1

    def record_processed(self, time: float, channel_id: int) -> None:
        """Record a processed request."""
        self._check_window_boundary(time)
        self._current_window_processed += 1
        self._update_channel_state(time, channel_id, "idle")

    def record_rejected(self, time: float) -> None:
        """Record a rejected request."""
        self._check_window_boundary(time)
        self._current_window_rejected += 1

    def record_service_start(
        self, time: float, channel_id: int, service_duration: float
    ) -> None:
        """Record service start on a channel."""
        self._check_window_boundary(time)
        self._update_channel_state(time, channel_id, "busy")

        # Accumulate busy time for current window
        window_end = min(time + service_duration, self._get_window_end())
        busy_duration = window_end - time
        self._current_window_busy_time[channel_id] += busy_duration

    def record_occupancy_snapshot(
        self, time: float, busy_channels: int
    ) -> None:
        """Record channel occupancy at a point in time."""
        if time - self._last_snapshot_time >= self.snapshot_interval:
            self.occupancy_snapshots.append((time, busy_channels))
            self._last_snapshot_time = time

    def start_phase(
        self, time: float, phase_index: int, arrival_rate: float
    ) -> None:
        """Start tracking a new phase for non-stationary arrivals."""
        # Finalize previous phase
        if self.current_phase is not None:
            self.current_phase["end_time"] = time
            self.phase_data.append(self.current_phase)

        # Start new phase
        self.current_phase = {
            "phase_index": phase_index,
            "start_time": time,
            "end_time": None,  # Will be set when phase ends
            "arrival_rate": arrival_rate,
            "arrivals": 0,
            "processed": 0,
            "rejected": 0,
            "busy_time": {ch: 0.0 for ch in range(self.num_channels)},
        }

    def record_phase_arrival(self) -> None:
        """Record arrival in current phase."""
        if self.current_phase:
            self.current_phase["arrivals"] += 1

    def record_phase_processed(
        self, channel_id: int, service_time: float
    ) -> None:
        """Record processed request in current phase."""
        if self.current_phase:
            self.current_phase["processed"] += 1
            self.current_phase["busy_time"][channel_id] += service_time

    def record_phase_rejected(self) -> None:
        """Record rejection in current phase."""
        if self.current_phase:
            self.current_phase["rejected"] += 1

    def finalize(self, final_time: float) -> None:
        """Finalize data collection."""
        # Close current window
        self._finalize_current_window(final_time)

        # Close current phase
        if self.current_phase is not None:
            self.current_phase["end_time"] = final_time
            self.phase_data.append(self.current_phase)

        # Finalize channel states
        for channel_id in range(self.num_channels):
            last_time, last_state = self.channel_last_state_change[channel_id]
            if last_time < final_time:
                period = BusyIdlePeriod(
                    channel_id=channel_id,
                    period_type=last_state,  # type: ignore
                    duration=final_time - last_time,
                    start_time=last_time,
                    end_time=final_time,
                )
                self.channel_states[channel_id].append(period)

    def _check_window_boundary(self, time: float) -> None:
        """Check if we've crossed into a new time window."""
        window_end = self._get_window_end()
        if time >= window_end:
            self._finalize_current_window(window_end)
            self._current_window_start = window_end

    def _get_window_end(self) -> float:
        """Get end time of current window."""
        return min(
            self._current_window_start + self.window_size, self.simulation_time
        )

    def _finalize_current_window(self, end_time: float) -> None:
        """Finalize current time window."""
        duration = end_time - self._current_window_start

        if duration <= 0:
            return

        # Calculate average utilization
        total_busy_time = sum(self._current_window_busy_time.values())
        total_channel_time = self.num_channels * duration
        avg_utilization = (
            total_busy_time / total_channel_time
            if total_channel_time > 0
            else 0.0
        )

        avg_busy_channels = total_busy_time / duration if duration > 0 else 0.0

        window_metrics = {
            "start_time": self._current_window_start,
            "end_time": end_time,
            "duration": duration,
            "arrivals": self._current_window_arrivals,
            "processed": self._current_window_processed,
            "rejected": self._current_window_rejected,
            "avg_utilization": avg_utilization,
            "avg_busy_channels": avg_busy_channels,
        }

        self.window_data.append(window_metrics)

        # Reset accumulators
        self._current_window_arrivals = 0
        self._current_window_processed = 0
        self._current_window_rejected = 0
        for channel_id in range(self.num_channels):
            self._current_window_busy_time[channel_id] = 0.0

    def _update_channel_state(
        self, time: float, channel_id: int, new_state: str
    ) -> None:
        """Update channel state and record period."""
        last_time, last_state = self.channel_last_state_change[channel_id]

        if last_state != new_state and time > last_time:
            # Record period
            period = BusyIdlePeriod(
                channel_id=channel_id,
                period_type=last_state,  # type: ignore
                duration=time - last_time,
                start_time=last_time,
                end_time=time,
            )
            self.channel_states[channel_id].append(period)

            # Update state
            self.channel_last_state_change[channel_id] = (time, new_state)


def analyze_temporal_profile(
    collector: TemporalCollector,
    peak_threshold: float = 0.7,
    valley_threshold: float = 0.3,
) -> TemporalProfile:
    """Analyze collected temporal data and generate profile.

    Args:
        collector: Temporal data collector.
        peak_threshold: Utilization threshold for peak detection.
        valley_threshold: Utilization threshold for valley detection.

    Returns:
        Complete temporal profile with all analyses.
    """
    # 1. Convert window data to TemporalMetrics
    temporal_metrics = []
    for window in collector.window_data:
        total = window["arrivals"]
        rejection_rate = window["rejected"] / total if total > 0 else 0.0

        metrics = TemporalMetrics(
            time_window=TimeWindow(
                start_time=window["start_time"],
                end_time=window["end_time"],
                duration=window["duration"],
            ),
            arrivals=window["arrivals"],
            processed=window["processed"],
            rejected=window["rejected"],
            rejection_rate=rejection_rate,
            avg_utilization=window["avg_utilization"],
            avg_busy_channels=window["avg_busy_channels"],
        )
        temporal_metrics.append(metrics)

    # 2. Convert occupancy snapshots
    occupancy_snapshots = [
        OccupancySnapshot(
            time=time,
            busy_channels=busy,
            utilization=busy / collector.num_channels,
        )
        for time, busy in collector.occupancy_snapshots
    ]

    # 3. Process phase metrics (if non-stationary)
    phase_metrics = None
    if collector.phase_data:
        phase_metrics = []
        for phase in collector.phase_data:
            duration = phase["end_time"] - phase["start_time"]
            total_channel_time = collector.num_channels * duration
            total_busy_time = sum(phase["busy_time"].values())

            avg_util = (
                total_busy_time / total_channel_time
                if total_channel_time > 0
                else 0.0
            )

            rejection_prob = (
                phase["rejected"] / phase["arrivals"]
                if phase["arrivals"] > 0
                else 0.0
            )

            throughput = phase["processed"] / duration if duration > 0 else 0.0

            metrics = PhaseMetrics(
                phase_index=phase["phase_index"],
                start_time=phase["start_time"],
                end_time=phase["end_time"],
                arrival_rate=phase["arrival_rate"],
                total_arrivals=phase["arrivals"],
                processed_requests=phase["processed"],
                rejected_requests=phase["rejected"],
                rejection_probability=rejection_prob,
                avg_utilization=avg_util,
                throughput=throughput,
            )
            phase_metrics.append(metrics)

    # 4. Detect peaks and valleys
    peak_periods = _detect_peaks_valleys(
        temporal_metrics, peak_threshold, valley_threshold
    )

    # 5. Analyze busy/idle periods
    busy_idle_stats = _analyze_busy_idle_periods(collector.channel_states)

    # 6. Calculate summary statistics
    utilizations = [m.avg_utilization for m in temporal_metrics]
    overall_peak = max(utilizations) if utilizations else 0.0
    overall_valley = min(utilizations) if utilizations else 0.0
    utilization_variance = float(np.var(utilizations)) if utilizations else 0.0

    return TemporalProfile(
        window_size=collector.window_size,
        temporal_metrics=temporal_metrics,
        occupancy_snapshots=occupancy_snapshots,
        phase_metrics=phase_metrics,
        peak_periods=peak_periods,
        busy_idle_stats=busy_idle_stats,
        overall_peak_utilization=overall_peak,
        overall_valley_utilization=overall_valley,
        utilization_variance=utilization_variance,
    )


def _detect_peaks_valleys(
    metrics: list[TemporalMetrics],
    peak_threshold: float,
    valley_threshold: float,
) -> list[PeakPeriod]:
    """Detect peak and valley periods in utilization."""
    if not metrics:
        return []

    periods = []
    current_period: dict[str, Any] | None = None

    for metric in metrics:
        util = metric.avg_utilization

        # Determine if this is a peak or valley
        if util >= peak_threshold:
            period_type = "peak"
        elif util <= valley_threshold:
            period_type = "valley"
        else:
            # Neither peak nor valley - finalize current period
            if current_period:
                periods.append(_finalize_period(current_period))
                current_period = None
            continue

        # Check if we're continuing or starting a new period
        if current_period is None:
            # Start new period
            current_period = {
                "type": period_type,
                "start_time": metric.time_window.start_time,
                "end_time": metric.time_window.end_time,
                "busy_channels_sum": metric.avg_busy_channels,
                "utilization_sum": metric.avg_utilization,
                "rejection_count": metric.rejected,
                "count": 1,
            }
        elif current_period["type"] == period_type:
            current_period["end_time"] = metric.time_window.end_time
            current_period["busy_channels_sum"] += metric.avg_busy_channels
            current_period["utilization_sum"] += metric.avg_utilization
            current_period["rejection_count"] += metric.rejected
            current_period["count"] += 1
        else:
            periods.append(_finalize_period(current_period))
            current_period = {
                "type": period_type,
                "start_time": metric.time_window.start_time,
                "end_time": metric.time_window.end_time,
                "busy_channels_sum": metric.avg_busy_channels,
                "utilization_sum": metric.avg_utilization,
                "rejection_count": metric.rejected,
                "count": 1,
            }

    if current_period:
        periods.append(_finalize_period(current_period))

    return periods


def _finalize_period(period_data: dict[str, Any]) -> PeakPeriod:
    """Convert period data dict to PeakPeriod."""
    count = period_data["count"]
    return PeakPeriod(
        period_type=period_data["type"],
        start_time=period_data["start_time"],
        end_time=period_data["end_time"],
        duration=period_data["end_time"] - period_data["start_time"],
        avg_busy_channels=period_data["busy_channels_sum"] / count,
        avg_utilization=period_data["utilization_sum"] / count,
        rejection_count=period_data["rejection_count"],
    )


def _analyze_busy_idle_periods(
    channel_states: dict[int, list[BusyIdlePeriod]],
) -> list[BusyIdleStatistics]:
    """Analyze busy/idle period distributions."""
    stats_list = []

    for channel_id, periods in channel_states.items():
        busy_durations = [
            p.duration for p in periods if p.period_type == "busy"
        ]
        idle_durations = [
            p.duration for p in periods if p.period_type == "idle"
        ]

        if not busy_durations:
            busy_durations = [0.0]
        if not idle_durations:
            idle_durations = [0.0]

        stats = BusyIdleStatistics(
            channel_id=channel_id,
            busy_periods=busy_durations,
            idle_periods=idle_durations,
            mean_busy_duration=float(np.mean(busy_durations)),
            mean_idle_duration=float(np.mean(idle_durations)),
            max_busy_duration=float(np.max(busy_durations)),
            max_idle_duration=float(np.max(idle_durations)),
            total_busy_time=float(np.sum(busy_durations)),
            total_idle_time=float(np.sum(idle_durations)),
        )
        stats_list.append(stats)

    return stats_list
