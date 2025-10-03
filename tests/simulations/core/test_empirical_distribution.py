"""Tests for empirical distribution."""

import pytest
import numpy as np
from src.simulations.core.distributions import EmpiricalDistribution
from src.simulations.core.schemas import EmpiricalParams


class TestEmpiricalDistribution:
    """Test empirical distribution implementation."""

    def test_inverse_transform_basic(self):
        """Test basic inverse transform sampling."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        dist = EmpiricalDistribution(data, method="inverse_transform")

        # Generate samples
        samples = dist.generate_batch(1000)

        # Should be within observed range
        assert all(1.0 <= x <= 5.0 for x in samples)

        # Mean should be close to data mean
        assert np.mean(samples) == pytest.approx(3.0, abs=0.5)

    def test_kde_basic(self):
        """Test KDE sampling."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        dist = EmpiricalDistribution(data, method="kde")

        # Generate samples
        samples = dist.generate_batch(1000)

        # KDE can generate outside range, but should be close
        assert np.mean(samples) == pytest.approx(3.0, abs=1.0)

    def test_too_few_data_points(self):
        """Test that single data point raises error."""
        with pytest.raises(ValueError, match="at least 2 data points"):
            EmpiricalDistribution([1.0], method="inverse_transform")

    def test_invalid_method(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError, match="must be"):
            EmpiricalDistribution([1.0, 2.0], method="invalid")

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        data = [1.0, 1.5, 2.0, 2.5, 3.0]

        rng1 = np.random.default_rng(42)
        dist1 = EmpiricalDistribution(
            data, method="inverse_transform", rng=rng1
        )

        rng2 = np.random.default_rng(42)
        dist2 = EmpiricalDistribution(
            data, method="inverse_transform", rng=rng2
        )

        samples1 = [dist1.generate() for _ in range(10)]
        samples2 = [dist2.generate() for _ in range(10)]

        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_get_params(self):
        """Test parameter retrieval."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        dist = EmpiricalDistribution(data, method="inverse_transform")

        params = dist.get_params()

        assert params["method"] == "inverse_transform"
        assert params["data_size"] == 5
        assert params["data_min"] == 1.0
        assert params["data_max"] == 5.0
        assert params["data_mean"] == 3.0

    def test_repr(self):
        """Test string representation."""
        data = [1.0, 2.0, 3.0]
        dist = EmpiricalDistribution(data, method="kde")

        repr_str = repr(dist)
        assert "Empirical" in repr_str
        assert "n=3" in repr_str
        assert "kde" in repr_str


class TestEmpiricalParams:
    """Test EmpiricalParams schema validation."""

    def test_valid_params(self):
        """Test valid empirical parameters."""
        params = EmpiricalParams(
            data=[1.0, 2.0, 3.0, 4.0, 5.0],
            method="inverse_transform",
        )

        assert params.distribution == "empirical"
        assert len(params.data) == 5

    def test_minimum_data_validation(self):
        """Test that minimum 2 data points required."""
        with pytest.raises(ValueError):
            EmpiricalParams(data=[1.0], method="kde")

    def test_negative_values_warning(self, caplog):
        """Test warning for negative values."""
        import logging

        caplog.set_level(logging.WARNING)

        params = EmpiricalParams(
            data=[-1.0, 0.5, 1.0, 2.0],
            method="inverse_transform",
        )

        assert any(
            "negative values" in record.message.lower()
            for record in caplog.records
        )

    def test_small_sample_warning(self, caplog):
        """Test warning for small sample size."""
        import logging

        caplog.set_level(logging.WARNING)

        params = EmpiricalParams(
            data=[1.0, 2.0, 3.0],  # Only 3 points
            method="inverse_transform",
        )

        assert any(
            "small empirical sample" in record.message.lower()
            for record in caplog.records
        )
