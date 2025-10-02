"""Distribution generators for discrete event simulation.

This module provides factory functions to generate random variates from various
probability distributions for use in queueing simulations.
"""

import numpy as np
from typing import Callable
from scipy.stats import truncnorm

from src.simulations.core.enums import DistributionType
from src.simulations.core.schemas import DistributionParams


def get_generator(params: DistributionParams) -> Callable[[], float]:
    """Create a random variate generator based on distribution parameters.

    Args:
        params: Distribution parameters including type and shape parameters.

    Returns:
        A callable that generates random variates from the specified distribution.

    Raises:
        NotImplementedError: If the distribution type is not supported.

    Example:
        >>> from src.simulations.core.schemas import ExponentialParams
        >>> params = ExponentialParams(rate=2.0)
        >>> generator = get_generator(params)
        >>> sample = generator()  # Returns a random exponential variate
    """
    if params.distribution == DistributionType.EXPONENTIAL:
        # np.random.exponential expects scale = 1/lambda
        return lambda: np.random.exponential(1.0 / params.rate)

    if params.distribution == DistributionType.UNIFORM:
        return lambda: np.random.uniform(params.a, params.b)

    if params.distribution == DistributionType.GAMMA:
        # k -> shape, theta -> scale
        return lambda: np.random.gamma(params.k, params.theta)

    if params.distribution == DistributionType.WEIBULL:
        # np.random.weibull expects shape (k). Scale (lambda) multiplies.
        return lambda: params.lambda_param * np.random.weibull(params.k)

    if params.distribution == DistributionType.TRUNCATED_NORMAL:
        a_std = (params.a - params.mu) / params.sigma
        b_std = (params.b - params.mu) / params.sigma
        return lambda: truncnorm.rvs(
            a_std, b_std, loc=params.mu, scale=params.sigma
        )

    raise NotImplementedError(
        f"Distribution '{params.distribution}' is not supported."
    )
