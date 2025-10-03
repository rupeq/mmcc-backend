import pytest
from pydantic import ValidationError

from src.simulations.core.schemas import (
    SimulationRequest,
    ExponentialParams,
)


class TestDataCollectionConfiguration:
    def test_default_collection_settings(self):
        """Test default data collection settings"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
        )

        assert request.collect_gantt_data is True
        assert request.collect_service_times is True
        assert request.max_gantt_items == 10000
        assert request.max_service_time_samples == 1000

    def test_disable_gantt_collection(self):
        """Test disabling Gantt chart collection"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            collect_gantt_data=False,
        )

        assert request.collect_gantt_data is False
        assert request.max_gantt_items == 0  # Auto-adjusted

    def test_disable_service_times_collection(self):
        """Test disabling service time collection"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            collect_service_times=False,
        )

        assert request.collect_service_times is False
        assert request.max_service_time_samples == 0  # Auto-adjusted

    def test_custom_limits(self):
        """Test custom collection limits"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            max_gantt_items=500,
            max_service_time_samples=2000,
        )

        assert request.max_gantt_items == 500
        assert request.max_service_time_samples == 2000

    def test_unlimited_collection(self):
        """Test unlimited data collection"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            max_gantt_items=None,
            max_service_time_samples=None,
        )

        assert request.max_gantt_items is None
        assert request.max_service_time_samples is None

    def test_validation_negative_limits(self):
        """Test that negative limits are rejected"""
        with pytest.raises(ValidationError):
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                max_gantt_items=-1,
            )

    def test_high_limit_warning(self, caplog):
        """Test warning for very high limits"""
        import logging

        caplog.set_level(logging.WARNING)

        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            max_gantt_items=200000,
        )

        assert "may cause memory issues" in caplog.text
