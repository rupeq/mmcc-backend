import pytest
from pydantic import ValidationError
from src.simulations.core.schemas import (
    SimulationRequest,
    ExponentialParams,
    ArrivalScheduleItem,
)


class TestArrivalScheduleValidation:
    """Test arrival schedule validation in SimulationRequest"""

    def test_valid_arrival_schedule(self):
        """Test that a valid arrival schedule is accepted"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=50.0, rate=1.0),
                ArrivalScheduleItem(duration=50.0, rate=2.0),
            ],
        )

        assert request.arrival_schedule is not None
        assert len(request.arrival_schedule) == 2
        assert request.arrival_schedule[0].duration == 50.0
        assert request.arrival_schedule[1].rate == 2.0

    def test_arrival_schedule_duration_less_than_sim_time(self):
        """Test that schedule duration < simulation_time raises error"""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                arrival_schedule=[
                    ArrivalScheduleItem(duration=30.0, rate=1.0),
                    ArrivalScheduleItem(duration=40.0, rate=2.0),
                    # Total: 70s < 100s simulation time
                ],
            )

        error_msg = str(exc_info.value)
        assert "schedule duration" in error_msg.lower()
        assert "less than" in error_msg.lower()
        assert "70.00" in error_msg  # Total duration
        assert "100.00" in error_msg  # Simulation time

    def test_arrival_schedule_exact_duration_match(self):
        """Test that schedule duration exactly matching simulation time is valid"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=60.0, rate=1.0),
                ArrivalScheduleItem(duration=40.0, rate=2.0),
                # Total: 100s = 100s simulation time
            ],
        )

        assert request.arrival_schedule is not None
        total_duration = sum(item.duration for item in request.arrival_schedule)
        assert total_duration == 100.0

    def test_arrival_schedule_duration_exceeds_sim_time(self):
        """Test that schedule duration > simulation_time is valid"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=80.0, rate=1.0),
                ArrivalScheduleItem(duration=50.0, rate=2.0),
                # Total: 130s > 100s simulation time (OK)
            ],
        )

        assert request.arrival_schedule is not None
        total_duration = sum(item.duration for item in request.arrival_schedule)
        assert total_duration > request.simulation_time

    def test_arrival_schedule_negative_duration(self):
        """Test that negative duration raises error"""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                arrival_schedule=[
                    ArrivalScheduleItem(duration=-10.0, rate=1.0),
                    ArrivalScheduleItem(duration=110.0, rate=2.0),
                ],
            )

        error_msg = str(exc_info.value).lower()
        # Pydantic field validation error
        assert "duration" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_arrival_schedule_zero_duration(self):
        """Test that zero duration raises error"""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                arrival_schedule=[
                    ArrivalScheduleItem(duration=50.0, rate=1.0),
                    ArrivalScheduleItem(duration=0.0, rate=2.0),
                    ArrivalScheduleItem(duration=50.0, rate=3.0),
                ],
            )

        error_msg = str(exc_info.value).lower()
        assert "duration" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_arrival_schedule_negative_rate(self):
        """Test that negative rate raises error"""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                arrival_schedule=[
                    ArrivalScheduleItem(duration=50.0, rate=1.0),
                    ArrivalScheduleItem(duration=50.0, rate=-2.0),
                ],
            )

        error_msg = str(exc_info.value).lower()
        assert "rate" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_arrival_schedule_zero_rate(self):
        """Test that zero rate raises error"""
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                arrival_schedule=[
                    ArrivalScheduleItem(duration=100.0, rate=0.0),
                ],
            )

        error_msg = str(exc_info.value).lower()
        assert "rate" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_arrival_schedule_single_interval(self):
        """Test arrival schedule with single interval"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=50.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=50.0, rate=5.0),
            ],
        )

        assert len(request.arrival_schedule) == 1
        assert request.arrival_schedule[0].rate == 5.0

    def test_arrival_schedule_many_intervals(self):
        """Test arrival schedule with many intervals"""
        intervals = [
            ArrivalScheduleItem(duration=10.0, rate=float(i + 1))
            for i in range(20)
        ]

        request = SimulationRequest(
            num_channels=2,
            simulation_time=200.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=intervals,
        )

        assert len(request.arrival_schedule) == 20
        total_duration = sum(item.duration for item in request.arrival_schedule)
        assert total_duration == 200.0

    def test_arrival_schedule_very_large_warns(self, caplog):
        """Test that very large schedule (>1000 intervals) triggers warning"""
        import logging
        caplog.set_level(logging.WARNING)

        # Create 1001 intervals
        intervals = [
            ArrivalScheduleItem(duration=1.0, rate=1.0) for _ in range(1001)
        ]

        request = SimulationRequest(
            num_channels=2,
            simulation_time=1001.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=intervals,
        )

        # Should still create successfully
        assert len(request.arrival_schedule) == 1001

        # But should log a warning
        assert any(
            "large arrival schedule" in record.message.lower()
            for record in caplog.records
        )
        assert any("1001" in record.message for record in caplog.records)

    def test_no_arrival_schedule_is_valid(self):
        """Test that absence of arrival schedule is valid (uses stationary arrival)"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=None,
        )

        assert request.arrival_schedule is None

    def test_arrival_schedule_empty_list(self):
        """Test that empty arrival schedule list is treated as None"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[],
        )

        # Empty list should be kept as is
        assert request.arrival_schedule == []

    def test_arrival_schedule_with_fractional_values(self):
        """Test arrival schedule with fractional durations and rates"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=33.333, rate=1.5),
                ArrivalScheduleItem(duration=33.333, rate=2.7),
                ArrivalScheduleItem(duration=33.334, rate=3.14159),
            ],
        )

        assert request.arrival_schedule is not None
        total_duration = sum(item.duration for item in request.arrival_schedule)
        assert abs(total_duration - 100.0) < 0.001

    def test_arrival_schedule_multiple_validation_errors(self):
        """Test that first validation error is reported"""
        # This schedule has negative duration in first item
        with pytest.raises(ValidationError) as exc_info:
            SimulationRequest(
                num_channels=2,
                simulation_time=100.0,
                arrival_process=ExponentialParams(rate=1.0),
                service_process=ExponentialParams(rate=2.0),
                arrival_schedule=[
                    ArrivalScheduleItem(duration=-10.0, rate=1.0),
                    ArrivalScheduleItem(duration=20.0, rate=-2.0),
                ],
            )

        # Should catch the first error (negative duration)
        error_msg = str(exc_info.value).lower()
        assert "duration" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_arrival_schedule_boundary_rates(self):
        """Test arrival schedule with very small and very large rates"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=100.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=50.0, rate=0.001),  # Very small
                ArrivalScheduleItem(duration=50.0, rate=1000.0),  # Very large
            ],
        )

        assert request.arrival_schedule[0].rate == 0.001
        assert request.arrival_schedule[1].rate == 1000.0

    def test_arrival_schedule_boundary_durations(self):
        """Test arrival schedule with very small durations"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=1.0,
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=0.001, rate=1.0),
                ArrivalScheduleItem(duration=0.999, rate=2.0),
            ],
        )

        total_duration = sum(item.duration for item in request.arrival_schedule)
        assert total_duration == 1.0

    def test_arrival_schedule_with_different_time_scales(self):
        """Test arrival schedule covering different time scales"""
        request = SimulationRequest(
            num_channels=2,
            simulation_time=10000.0,  # Long simulation
            arrival_process=ExponentialParams(rate=1.0),
            service_process=ExponentialParams(rate=2.0),
            arrival_schedule=[
                ArrivalScheduleItem(duration=1000.0, rate=1.0),
                ArrivalScheduleItem(duration=2000.0, rate=5.0),
                ArrivalScheduleItem(duration=3000.0, rate=10.0),
                ArrivalScheduleItem(duration=4000.0, rate=2.0),
            ],
        )

        assert len(request.arrival_schedule) == 4
        total_duration = sum(item.duration for item in request.arrival_schedule)
        assert total_duration == 10000.0


class TestArrivalScheduleItemValidation:
    """Test validation at the ArrivalScheduleItem level"""

    def test_valid_schedule_item(self):
        """Test creating valid schedule item"""
        item = ArrivalScheduleItem(duration=10.0, rate=2.0)
        assert item.duration == 10.0
        assert item.rate == 2.0

    def test_schedule_item_negative_duration_field_validation(self):
        """Test that negative duration fails at field level"""
        # This should fail at Pydantic's gt=0 validation
        with pytest.raises(ValidationError) as exc_info:
            ArrivalScheduleItem(duration=-5.0, rate=1.0)

        error_msg = str(exc_info.value).lower()
        assert "duration" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_schedule_item_negative_rate_field_validation(self):
        """Test that negative rate fails at field level"""
        # This should fail at Pydantic's gt=0 validation
        with pytest.raises(ValidationError) as exc_info:
            ArrivalScheduleItem(duration=5.0, rate=-1.0)

        error_msg = str(exc_info.value).lower()
        assert "rate" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_schedule_item_zero_duration_field_validation(self):
        """Test that zero duration fails at field level"""
        with pytest.raises(ValidationError) as exc_info:
            ArrivalScheduleItem(duration=0.0, rate=1.0)

        error_msg = str(exc_info.value).lower()
        assert "duration" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_schedule_item_zero_rate_field_validation(self):
        """Test that zero rate fails at field level"""
        with pytest.raises(ValidationError) as exc_info:
            ArrivalScheduleItem(duration=5.0, rate=0.0)

        error_msg = str(exc_info.value).lower()
        assert "rate" in error_msg
        assert ("greater than 0" in error_msg or "greater_than" in error_msg)

    def test_schedule_item_extreme_values(self):
        """Test schedule item with extreme but valid values"""
        # Very small values
        item1 = ArrivalScheduleItem(duration=0.0001, rate=0.0001)
        assert item1.duration == 0.0001
        assert item1.rate == 0.0001

        # Very large values
        item2 = ArrivalScheduleItem(duration=1e10, rate=1e10)
        assert item2.duration == 1e10
        assert item2.rate == 1e10


# Keep existing validation test
def test_mm_c_c_matches_erlang_b():
    """Validate simulator against theoretical results"""
    import math
    from src.simulations.core.engine import run_replications
    from src.simulations.core.schemas import ExponentialParams, SimulationRequest

    lambda_rate = 5.0
    mu_rate = 10.0
    c = 2
    rho = lambda_rate / mu_rate

    def erlang_b(rho: float, c: int) -> float:
        numerator = (rho**c) / math.factorial(c)
        denominator = sum((rho**k) / math.factorial(k) for k in range(c + 1))
        return numerator / denominator

    theoretical_blocking = erlang_b(rho, c)

    params = SimulationRequest(
        num_channels=c,
        simulation_time=10000,
        num_replications=30,
        arrival_process=ExponentialParams(rate=lambda_rate),
        service_process=ExponentialParams(rate=mu_rate),
        random_seed=42,
    )

    response = run_replications(params)
    simulated_blocking = response.aggregated_metrics.rejection_probability
    ci = response.aggregated_metrics.rejection_probability_ci

    assert ci.lower_bound <= theoretical_blocking <= ci.upper_bound, (
        f"Theoretical Erlang B ({theoretical_blocking:.4f}) outside "
        f"simulation CI [{ci.lower_bound:.4f}, {ci.upper_bound:.4f}]"
    )

    relative_error = (
            abs(simulated_blocking - theoretical_blocking) / theoretical_blocking
    )
    assert relative_error < 0.05, f"Relative error {relative_error:.2%} > 5%"
