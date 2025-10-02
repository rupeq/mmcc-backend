import pytest

from src.simulations.core.statistics import SimulationStatistics
from src.simulations.core.schemas import GanttChartItem


class TestSimulationStatistics:
    def test_default_configuration(self):
        """Test default statistics configuration"""
        stats = SimulationStatistics()

        assert stats._max_service_samples == 1000
        assert stats._max_gantt_items == 10000
        assert stats._collect_service_times is True
        assert stats._collect_gantt_data is True

    def test_configure_limits(self):
        """Test configuring collection limits"""
        stats = SimulationStatistics()
        stats.configure_limits(
            max_service_samples=500,
            max_gantt_items=2000,
            collect_service_times=False,
            collect_gantt_data=True,
        )

        assert stats._max_service_samples == 500
        assert stats._max_gantt_items == 2000
        assert stats._collect_service_times is False
        assert stats._collect_gantt_data is True

    def test_service_time_collection_with_limit(self):
        """Test service time collection respects limit"""
        stats = SimulationStatistics()
        stats.configure_limits(max_service_samples=3)

        for i in range(10):
            stats.record_service_time(float(i))

        assert len(stats.service_times) == 3
        assert stats.service_times == [0.0, 1.0, 2.0]
        assert stats._service_times_truncated is True
        assert stats._count == 10  # Online stats still accurate

    def test_service_time_unlimited_collection(self):
        """Test unlimited service time collection"""
        stats = SimulationStatistics()
        stats.configure_limits(max_service_samples=None)

        for i in range(2000):
            stats.record_service_time(1.0)

        assert len(stats.service_times) == 2000
        assert stats._service_times_truncated is False

    def test_service_time_collection_disabled(self):
        """Test disabling service time collection"""
        stats = SimulationStatistics()
        stats.configure_limits(
            max_service_samples=0, collect_service_times=False
        )

        for i in range(100):
            stats.record_service_time(float(i))

        assert len(stats.service_times) == 0
        assert stats._count == 100  # Online stats still work
        assert stats.service_time_mean == pytest.approx(49.5)

    def test_gantt_collection_with_limit(self):
        """Test Gantt chart collection respects limit"""
        stats = SimulationStatistics()
        stats.configure_limits(max_gantt_items=5)

        for i in range(10):
            item = GanttChartItem(
                channel=0, start=float(i), end=float(i + 1), duration=1.0
            )
            stats.record_gantt_item(item)

        assert len(stats.gantt_items) == 5
        assert stats._gantt_items_truncated is True

    def test_gantt_collection_unlimited(self):
        """Test unlimited Gantt collection"""
        stats = SimulationStatistics()
        stats.configure_limits(max_gantt_items=None)

        for i in range(1000):
            item = GanttChartItem(
                channel=0, start=float(i), end=float(i + 1), duration=1.0
            )
            stats.record_gantt_item(item)

        assert len(stats.gantt_items) == 1000
        assert stats._gantt_items_truncated is False

    def test_gantt_collection_disabled(self):
        """Test disabling Gantt collection"""
        stats = SimulationStatistics()
        stats.configure_limits(max_gantt_items=0, collect_gantt_data=False)

        for i in range(100):
            item = GanttChartItem(
                channel=0, start=float(i), end=float(i + 1), duration=1.0
            )
            stats.record_gantt_item(item)

        assert len(stats.gantt_items) == 0

    def test_online_statistics_accuracy(self):
        """Test Welford's online algorithm for mean/std"""
        import numpy as np

        stats = SimulationStatistics()
        stats.configure_limits(max_service_samples=10)  # Limit storage

        values = [1.5, 2.3, 3.1, 4.7, 5.2, 6.8, 7.1, 8.9, 9.3, 10.1]
        for val in values:
            stats.record_service_time(val)

        # Even with limited storage, online stats are accurate
        assert stats.service_time_mean == pytest.approx(np.mean(values))
        assert stats.service_time_std == pytest.approx(np.std(values, ddof=1))
        assert stats.service_time_variance == pytest.approx(
            np.var(values, ddof=1)
        )

    def test_collection_info(self):
        """Test getting collection information"""
        stats = SimulationStatistics()
        stats.configure_limits(max_service_samples=5, max_gantt_items=3)

        for i in range(10):
            stats.record_service_time(float(i))
            item = GanttChartItem(
                channel=0, start=float(i), end=float(i + 1), duration=1.0
            )
            stats.record_gantt_item(item)

        info = stats.get_collection_info()

        assert info["service_times_collected"] == 5
        assert info["service_times_truncated"] is True
        assert info["service_times_limit"] == 5
        assert info["gantt_items_collected"] == 3
        assert info["gantt_items_truncated"] is True
        assert info["gantt_items_limit"] == 3
        assert info["total_service_time_count"] == 10


class TestMemoryEfficiency:
    def test_large_simulation_memory_limit(self):
        """Test that limits prevent excessive memory usage"""
        stats = SimulationStatistics()
        stats.configure_limits(max_service_samples=1000, max_gantt_items=1000)

        # Simulate 1 million events
        for i in range(1_000_000):
            stats.record_service_time(1.0)
            if i % 10 == 0:  # Every 10th event
                item = GanttChartItem(
                    channel=0, start=float(i), end=float(i + 1), duration=1.0
                )
                stats.record_gantt_item(item)

        # Memory limited
        assert len(stats.service_times) == 1000
        assert len(stats.gantt_items) == 1000

        # But statistics still accurate
        assert stats._count == 1_000_000
        assert stats.service_time_mean == pytest.approx(1.0)
