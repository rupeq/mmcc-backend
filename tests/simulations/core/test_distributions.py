# tests/simulations/core/test_distributions.py
"""Tests for distribution generators.

Tests verify that all distributions:
- Generate positive values
- Are reproducible with seeds
- Have correct theoretical properties
- Handle edge cases properly
"""

import pytest
import numpy as np
from scipy import stats as scipy_stats

from src.simulations.core.distributions import (
    ExponentialDistribution,
    UniformDistribution,
    GammaDistribution,
    WeibullDistribution,
    TruncatedNormalDistribution,
    get_distribution,
)
from src.simulations.core.schemas import (
    ExponentialParams,
    UniformParams,
    GammaParams,
    WeibullParams,
    TruncatedNormalParams,
)


class TestDistributionBase:
    """Test abstract Distribution base class."""

    def test_generation_count_tracking(self):
        """Test that generation count is properly tracked."""
        dist = ExponentialDistribution(rate=1.0)

        assert dist.generation_count == 0

        dist.generate()
        assert dist.generation_count == 1

        dist.generate_batch(5)
        assert dist.generation_count == 6

        dist.reset_count()
        assert dist.generation_count == 0

    def test_batch_generation(self):
        """Test generating multiple variates at once."""
        dist = ExponentialDistribution(rate=2.0)
        batch = dist.generate_batch(100)

        assert len(batch) == 100
        assert all(x > 0 for x in batch)

    def test_batch_generation_invalid_size(self):
        """Test that invalid batch sizes raise errors."""
        dist = ExponentialDistribution(rate=1.0)

        with pytest.raises(ValueError, match="Size must be positive"):
            dist.generate_batch(0)

        with pytest.raises(ValueError, match="Size must be positive"):
            dist.generate_batch(-5)


class TestExponentialDistribution:
    """Test exponential distribution."""

    def test_initialization(self):
        """Test proper initialization."""
        dist = ExponentialDistribution(rate=2.0)

        assert dist.rate == 2.0
        assert dist.scale == 0.5
        assert dist.generation_count == 0

    def test_invalid_rate(self):
        """Test that invalid rates raise errors."""
        with pytest.raises(ValueError, match="Rate must be positive"):
            ExponentialDistribution(rate=0.0)

        with pytest.raises(ValueError, match="Rate must be positive"):
            ExponentialDistribution(rate=-1.0)

    def test_generate_returns_positive(self):
        """Test that generated values are positive."""
        dist = ExponentialDistribution(rate=1.0)

        for _ in range(100):
            value = dist.generate()
            assert value > 0

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same sequence."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        dist1 = ExponentialDistribution(rate=1.5, rng=rng1)
        dist2 = ExponentialDistribution(rate=1.5, rng=rng2)

        samples1 = [dist1.generate() for _ in range(10)]
        samples2 = [dist2.generate() for _ in range(10)]

        np.testing.assert_array_almost_equal(samples1, samples2)

    def test_mean_calculation(self):
        """Test theoretical mean calculation."""
        dist = ExponentialDistribution(rate=3.0)
        assert dist.get_mean() == pytest.approx(1.0 / 3.0)

    def test_empirical_mean_matches_theory(self):
        """Test that empirical mean matches theoretical value."""
        rate = 2.0
        rng = np.random.default_rng(123)
        dist = ExponentialDistribution(rate=rate, rng=rng)

        samples = dist.generate_batch(10000)
        empirical_mean = np.mean(samples)
        theoretical_mean = 1.0 / rate

        # Allow 5% error
        assert empirical_mean == pytest.approx(theoretical_mean, rel=0.05)

    def test_get_params(self):
        """Test parameter retrieval."""
        dist = ExponentialDistribution(rate=2.5)
        params = dist.get_params()

        assert params["rate"] == 2.5
        assert params["scale"] == 0.4

    def test_repr(self):
        """Test string representation."""
        dist = ExponentialDistribution(rate=1.5)
        repr_str = repr(dist)

        assert "Exponential" in repr_str
        assert "1.5" in repr_str


class TestUniformDistribution:
    """Test uniform distribution."""

    def test_initialization(self):
        """Test proper initialization."""
        dist = UniformDistribution(a=1.0, b=5.0)

        assert dist.a == 1.0
        assert dist.b == 5.0

    def test_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        with pytest.raises(
            ValueError, match="Lower bound must be less than upper bound"
        ):
            UniformDistribution(a=5.0, b=5.0)

        with pytest.raises(
            ValueError, match="Lower bound must be less than upper bound"
        ):
            UniformDistribution(a=10.0, b=5.0)

    def test_generate_in_range(self):
        """Test that generated values are within bounds."""
        dist = UniformDistribution(a=2.0, b=8.0)

        for _ in range(100):
            value = dist.generate()
            assert 2.0 <= value <= 8.0

    def test_mean_calculation(self):
        """Test theoretical mean calculation."""
        dist = UniformDistribution(a=1.0, b=5.0)
        assert dist.get_mean() == 3.0

    def test_empirical_mean_matches_theory(self):
        """Test that empirical mean matches theoretical value."""
        a, b = 1.0, 9.0
        rng = np.random.default_rng(456)
        dist = UniformDistribution(a=a, b=b, rng=rng)

        samples = dist.generate_batch(10000)
        empirical_mean = np.mean(samples)
        theoretical_mean = (a + b) / 2.0

        assert empirical_mean == pytest.approx(theoretical_mean, rel=0.05)

    def test_repr(self):
        """Test string representation."""
        dist = UniformDistribution(a=1.0, b=5.0)
        repr_str = repr(dist)

        assert "Uniform" in repr_str
        assert "1.0" in repr_str
        assert "5.0" in repr_str


class TestGammaDistribution:
    """Test gamma distribution."""

    def test_initialization(self):
        """Test proper initialization."""
        dist = GammaDistribution(k=2.0, theta=1.5)

        assert dist.k == 2.0
        assert dist.theta == 1.5

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(
            ValueError, match="Shape parameter k must be positive"
        ):
            GammaDistribution(k=0.0, theta=1.0)

        with pytest.raises(
            ValueError, match="Scale parameter theta must be positive"
        ):
            GammaDistribution(k=1.0, theta=-1.0)

    def test_generate_returns_positive(self):
        """Test that generated values are positive."""
        dist = GammaDistribution(k=2.0, theta=1.0)

        for _ in range(100):
            value = dist.generate()
            assert value > 0

    def test_mean_calculation(self):
        """Test theoretical mean calculation."""
        k, theta = 3.0, 2.0
        dist = GammaDistribution(k=k, theta=theta)

        assert dist.get_mean() == k * theta

    def test_empirical_mean_matches_theory(self):
        """Test that empirical mean matches theoretical value."""
        k, theta = 2.0, 3.0
        rng = np.random.default_rng(789)
        dist = GammaDistribution(k=k, theta=theta, rng=rng)

        samples = dist.generate_batch(10000)
        empirical_mean = np.mean(samples)
        theoretical_mean = k * theta

        assert empirical_mean == pytest.approx(theoretical_mean, rel=0.05)


class TestWeibullDistribution:
    """Test Weibull distribution."""

    def test_initialization(self):
        """Test proper initialization."""
        dist = WeibullDistribution(k=1.5, lambda_param=2.0)

        assert dist.k == 1.5
        assert dist.lambda_param == 2.0

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(
            ValueError, match="Shape parameter k must be positive"
        ):
            WeibullDistribution(k=0.0, lambda_param=1.0)

        with pytest.raises(
            ValueError, match="Scale parameter lambda must be positive"
        ):
            WeibullDistribution(k=1.0, lambda_param=-1.0)

    def test_generate_returns_positive(self):
        """Test that generated values are positive."""
        dist = WeibullDistribution(k=2.0, lambda_param=1.5)

        for _ in range(100):
            value = dist.generate()
            assert value > 0

    def test_mean_calculation(self):
        """Test theoretical mean uses gamma function."""
        dist = WeibullDistribution(k=2.0, lambda_param=1.0)
        mean = dist.get_mean()

        # For k=2, λ=1: E[X] = Γ(1.5) ≈ 0.8862
        assert mean == pytest.approx(0.8862, rel=0.01)


class TestTruncatedNormalDistribution:
    """Test truncated normal distribution."""

    def test_initialization(self):
        """Test proper initialization."""
        dist = TruncatedNormalDistribution(mu=5.0, sigma=2.0, a=0.0, b=10.0)

        assert dist.mu == 5.0
        assert dist.sigma == 2.0
        assert dist.a == 0.0
        assert dist.b == 10.0

    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(
            ValueError, match="Standard deviation must be positive"
        ):
            TruncatedNormalDistribution(mu=5.0, sigma=0.0, a=0.0, b=10.0)

        with pytest.raises(
            ValueError, match="Lower bound must be less than upper bound"
        ):
            TruncatedNormalDistribution(mu=5.0, sigma=1.0, a=10.0, b=5.0)

    def test_generate_in_range(self):
        """Test that generated values are within bounds."""
        dist = TruncatedNormalDistribution(mu=5.0, sigma=2.0, a=0.0, b=10.0)

        for _ in range(100):
            value = dist.generate()
            assert 0.0 <= value <= 10.0

    def test_empirical_mean_in_truncation_range(self):
        """Test that empirical mean is reasonable."""
        rng = np.random.default_rng(321)
        dist = TruncatedNormalDistribution(
            mu=5.0, sigma=2.0, a=0.0, b=10.0, rng=rng
        )

        samples = dist.generate_batch(10000)
        empirical_mean = np.mean(samples)

        # Mean should be close to mu since truncation is symmetric
        assert 4.0 <= empirical_mean <= 6.0


class TestGetDistributionFactory:
    """Test the factory function."""

    def test_exponential_creation(self):
        """Test creating exponential distribution."""
        params = ExponentialParams(rate=2.0)
        dist = get_distribution(params)

        assert isinstance(dist, ExponentialDistribution)
        assert dist.rate == 2.0

    def test_uniform_creation(self):
        """Test creating uniform distribution."""
        params = UniformParams(a=1.0, b=5.0)
        dist = get_distribution(params)

        assert isinstance(dist, UniformDistribution)
        assert dist.a == 1.0
        assert dist.b == 5.0

    def test_gamma_creation(self):
        """Test creating gamma distribution."""
        params = GammaParams(k=2.0, theta=1.5)
        dist = get_distribution(params)

        assert isinstance(dist, GammaDistribution)
        assert dist.k == 2.0
        assert dist.theta == 1.5

    def test_weibull_creation(self):
        """Test creating Weibull distribution."""
        params = WeibullParams(k=1.5, lambda_param=2.0)
        dist = get_distribution(params)

        assert isinstance(dist, WeibullDistribution)
        assert dist.k == 1.5
        assert dist.lambda_param == 2.0

    def test_truncated_normal_creation(self):
        """Test creating truncated normal distribution."""
        params = TruncatedNormalParams(mu=5.0, sigma=2.0, a=0.0, b=10.0)
        dist = get_distribution(params)

        assert isinstance(dist, TruncatedNormalDistribution)
        assert dist.mu == 5.0
        assert dist.sigma == 2.0

    def test_with_custom_rng(self):
        """Test factory with custom RNG."""
        params = ExponentialParams(rate=1.0)
        rng = np.random.default_rng(999)

        dist = get_distribution(params, rng=rng)
        assert dist.rng is rng

    def test_reproducibility_across_factory_calls(self):
        """Test that factory preserves reproducibility."""
        params = ExponentialParams(rate=1.5)

        rng1 = np.random.default_rng(42)
        dist1 = get_distribution(params, rng=rng1)

        rng2 = np.random.default_rng(42)
        dist2 = get_distribution(params, rng=rng2)

        samples1 = [dist1.generate() for _ in range(5)]
        samples2 = [dist2.generate() for _ in range(5)]

        np.testing.assert_array_almost_equal(samples1, samples2)


class TestStatisticalProperties:
    """Test statistical properties of distributions."""

    @pytest.mark.parametrize("rate", [0.5, 1.0, 2.0, 5.0])
    def test_exponential_follows_distribution(self, rate):
        """Test exponential follows theoretical distribution."""
        rng = np.random.default_rng(111)
        dist = ExponentialDistribution(rate=rate, rng=rng)

        samples = dist.generate_batch(1000)

        # Kolmogorov-Smirnov test
        _, p_value = scipy_stats.kstest(
            samples, lambda x: scipy_stats.expon.cdf(x, scale=1 / rate)
        )

        # Should not reject null hypothesis (p > 0.05)
        assert p_value > 0.05

    def test_uniform_follows_distribution(self):
        """Test uniform follows theoretical distribution."""
        rng = np.random.default_rng(222)
        dist = UniformDistribution(a=2.0, b=8.0, rng=rng)

        samples = dist.generate_batch(1000)

        _, p_value = scipy_stats.kstest(
            samples, lambda x: scipy_stats.uniform.cdf(x, loc=2.0, scale=6.0)
        )

        assert p_value > 0.05
