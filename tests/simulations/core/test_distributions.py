import pytest
from typing import Callable

from src.simulations.core.distributions import get_generator
from src.simulations.core.enums import DistributionType
from src.simulations.core.schemas import (
    ExponentialParams,
    UniformParams,
    GammaParams,
    WeibullParams,
    TruncatedNormalParams,
)


@pytest.mark.parametrize(
    "params",
    [
        ExponentialParams(rate=1.0),
        UniformParams(a=1.0, b=2.0),
        GammaParams(k=1.0, theta=1.0),
        WeibullParams(k=1.0, lambda_param=1.0),
        TruncatedNormalParams(mu=5, sigma=2, a=0, b=10),
    ],
)
def test_get_generator_returns_callable(params):
    """Verify that get_generator returns a callable function for supported types."""

    generator = get_generator(params)
    assert isinstance(generator, Callable)


@pytest.mark.parametrize(
    "params",
    [
        ExponentialParams(rate=1.0),
        UniformParams(a=1.0, b=2.0),
        GammaParams(k=1.0, theta=1.0),
        WeibullParams(k=1.0, lambda_param=1.0),
        TruncatedNormalParams(mu=5, sigma=2, a=0, b=10),
    ],
)
def test_generator_returns_positive_float(params):
    """Verify that a generated value is a non-negative float."""

    generator = get_generator(params)
    value = generator()
    assert isinstance(value, float)
    assert value >= 0.0


def test_get_generator_raises_not_implemented():
    """Verify that an unsupported distribution type raises an error."""

    class FakeParams:
        distribution = "fake_distribution"

    with pytest.raises(NotImplementedError):
        get_generator(FakeParams())
