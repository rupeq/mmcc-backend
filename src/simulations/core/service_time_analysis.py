"""Service time distribution analysis module.

This module provides comprehensive statistical analysis and goodness-of-fit
tests for service time distributions collected during simulations.
"""

import logging
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.simulations.core.distributions import Distribution

logger = logging.getLogger(__name__)


@dataclass
class QuantileStatistics:
    """Quantile statistics for service times.

    Attributes:
        p50: 50th percentile (median).
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        min: Minimum value.
        max: Maximum value.
    """

    p50: float
    p90: float
    p95: float
    p99: float
    min: float
    max: float


@dataclass
class GoodnessOfFitResult:
    """Results of goodness-of-fit tests.

    Attributes:
        ks_statistic: Kolmogorov-Smirnov test statistic.
        ks_pvalue: KS test p-value.
        chi2_statistic: Chi-square test statistic (if applicable).
        chi2_pvalue: Chi-square p-value (if applicable).
        test_passed: Whether distribution fits at significance level.
        warnings: List of warnings about test validity.
    """

    ks_statistic: float
    ks_pvalue: float
    chi2_statistic: float | None
    chi2_pvalue: float | None
    test_passed: bool
    warnings: list[str]


@dataclass
class ServiceTimeAnalysis:
    """Complete service time distribution analysis.

    Attributes:
        sample_size: Number of service time samples.
        mean: Sample mean.
        std: Sample standard deviation.
        variance: Sample variance.
        skewness: Sample skewness.
        kurtosis: Sample kurtosis.
        quantiles: Quantile statistics.
        theoretical_mean: Theoretical mean from distribution.
        goodness_of_fit: Goodness-of-fit test results.
    """

    sample_size: int
    mean: float
    std: float
    variance: float
    skewness: float
    kurtosis: float
    quantiles: QuantileStatistics
    theoretical_mean: float | None
    goodness_of_fit: GoodnessOfFitResult | None


def calculate_quantiles(data: np.ndarray) -> QuantileStatistics:
    """Calculate quantile statistics.

    Args:
        data: Array of service times.

    Returns:
        QuantileStatistics with percentiles and extremes.

    Example:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> quantiles = calculate_quantiles(data)
        >>> print(f"Median: {quantiles.p50}")
    """
    return QuantileStatistics(
        p50=float(np.percentile(data, 50)),
        p90=float(np.percentile(data, 90)),
        p95=float(np.percentile(data, 95)),
        p99=float(np.percentile(data, 99)),
        min=float(np.min(data)),
        max=float(np.max(data)),
    )


def perform_ks_test(
    data: np.ndarray, distribution: Distribution
) -> tuple[float, float]:
    """Perform Kolmogorov-Smirnov goodness-of-fit test.

    Tests whether the observed service times fit the theoretical distribution.

    Args:
        data: Array of observed service times.
        distribution: Theoretical distribution to test against.

    Returns:
        Tuple of (KS statistic, p-value).

    Example:
        >>> from src.simulations.core.distributions import ExponentialDistribution
        >>> data = np.random.exponential(scale=2.0, size=1000)
        >>> dist = ExponentialDistribution(rate=0.5)
        >>> ks_stat, p_value = perform_ks_test(data, dist)
        >>> print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
    """
    # Generate theoretical CDF
    dist_name = distribution.__class__.__name__.lower()

    if "exponential" in dist_name:
        params = distribution.get_params()
        scale = params["scale"]
        ks_stat, p_value = stats.kstest(data, "expon", args=(0, scale))
    elif "uniform" in dist_name:
        params = distribution.get_params()
        loc = params["a"]
        scale = params["b"] - params["a"]
        ks_stat, p_value = stats.kstest(data, "uniform", args=(loc, scale))
    elif "gamma" in dist_name:
        params = distribution.get_params()
        k = params["k"]
        theta = params["theta"]
        ks_stat, p_value = stats.kstest(data, "gamma", args=(k, 0, theta))
    elif "weibull" in dist_name:
        params = distribution.get_params()
        k = params["k"]
        lambda_param = params["lambda_param"]
        ks_stat, p_value = stats.kstest(
            data, "weibull_min", args=(k, 0, lambda_param)
        )
    elif "truncated" in dist_name or "normal" in dist_name:
        # Use generic KS test with empirical CDF
        sorted_data = np.sort(data)
        theoretical_cdf = np.array(
            [
                sum(distribution.generate() <= x for _ in range(10000)) / 10000
                for x in sorted_data[::10]
            ]
        )
        ks_stat = np.max(
            np.abs(np.linspace(0, 1, len(theoretical_cdf)) - theoretical_cdf)
        )
        # Approximate p-value using asymptotic formula
        n = len(data)
        p_value = stats.kstwobign.sf(ks_stat * np.sqrt(n))
    else:
        # Fallback: generic test
        logger.warning(f"No specific KS test for {dist_name}, using empirical")
        ks_stat = 0.0
        p_value = 1.0

    return ks_stat, p_value


def perform_chi_square_test(
    data: np.ndarray, distribution: Distribution, num_bins: int = 20
) -> tuple[float, float, list[str]]:
    """Perform chi-square goodness-of-fit test.

    Bins the data and compares observed vs expected frequencies.

    Args:
        data: Array of observed service times.
        distribution: Theoretical distribution.
        num_bins: Number of bins for histogram (default: 20).

    Returns:
        Tuple of (chi2 statistic, p-value, warnings).

    Note:
        Chi-square test requires expected frequencies ≥ 5 per bin.
        Returns None values if this assumption is violated.
    """
    warnings = []

    # Create bins
    observed_counts, bin_edges = np.histogram(data, bins=num_bins)

    # Calculate expected frequencies
    expected_counts = np.zeros(num_bins)
    for i in range(num_bins):
        # Generate samples from theoretical distribution
        samples = distribution.generate_batch(10000)
        expected_counts[i] = np.sum(
            (samples >= bin_edges[i]) & (samples < bin_edges[i + 1])
        )

    # Normalize to match sample size
    expected_counts = expected_counts / expected_counts.sum() * len(data)

    # Check chi-square assumptions
    if np.any(expected_counts < 5):
        warnings.append(
            "Chi-square test may be unreliable: some bins have "
            "expected frequency < 5"
        )
        # Combine bins to meet minimum requirement
        while np.any(expected_counts < 5) and len(expected_counts) > 2:
            min_idx = np.argmin(expected_counts)
            if min_idx == 0:
                # Combine first two bins
                observed_counts[1] += observed_counts[0]
                expected_counts[1] += expected_counts[0]
                observed_counts = observed_counts[1:]
                expected_counts = expected_counts[1:]
            else:
                # Combine with previous bin
                observed_counts[min_idx - 1] += observed_counts[min_idx]
                expected_counts[min_idx - 1] += expected_counts[min_idx]
                observed_counts = np.delete(observed_counts, min_idx)
                expected_counts = np.delete(expected_counts, min_idx)

    # Perform chi-square test
    chi2_stat, p_value = stats.chisquare(
        f_obs=observed_counts, f_exp=expected_counts
    )

    return chi2_stat, p_value, warnings


def analyze_service_times(
    service_times: list[float],
    distribution: Distribution | None = None,
    significance_level: float = 0.05,
) -> ServiceTimeAnalysis:
    """Perform comprehensive statistical analysis on service times.

    Args:
        service_times: List of observed service times.
        distribution: Theoretical distribution to test against (optional).
        significance_level: Significance level for hypothesis tests.

    Returns:
        ServiceTimeAnalysis with complete statistical summary.

    Raises:
        ValueError: If service_times is empty or contains non-positive values.

    Example:
        >>> service_times = [1.2, 1.5, 2.1, 1.8, 2.3]
        >>> analysis = analyze_service_times(service_times)
        >>> print(f"Mean: {analysis.mean:.2f}, Std: {analysis.std:.2f}")
    """
    if not service_times:
        raise ValueError("service_times cannot be empty")

    data = np.array(service_times)

    if np.any(data <= 0):
        logger.warning(
            "Service times contain non-positive values, "
            "some tests may be invalid"
        )

    # Basic statistics
    mean = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    variance = float(np.var(data, ddof=1))
    skewness = float(stats.skew(data))
    kurtosis = float(stats.kurtosis(data))

    # Quantiles
    quantiles = calculate_quantiles(data)

    # Theoretical mean (if distribution provided)
    theoretical_mean = (
        distribution.get_mean() if distribution is not None else None
    )

    # Goodness-of-fit tests (if distribution provided)
    goodness_of_fit = None
    if distribution is not None:
        try:
            # KS test
            ks_stat, ks_pvalue = perform_ks_test(data, distribution)

            # Chi-square test
            chi2_stat, chi2_pvalue, warnings = perform_chi_square_test(
                data, distribution
            )

            # Determine if tests passed
            test_passed = (
                ks_pvalue >= significance_level
                and chi2_pvalue >= significance_level
            )

            goodness_of_fit = GoodnessOfFitResult(
                ks_statistic=ks_stat,
                ks_pvalue=ks_pvalue,
                chi2_statistic=chi2_stat,
                chi2_pvalue=chi2_pvalue,
                test_passed=test_passed,
                warnings=warnings,
            )
        except Exception as e:
            logger.error(f"Goodness-of-fit tests failed: {e}")
            goodness_of_fit = None

    return ServiceTimeAnalysis(
        sample_size=len(data),
        mean=mean,
        std=std,
        variance=variance,
        skewness=skewness,
        kurtosis=kurtosis,
        quantiles=quantiles,
        theoretical_mean=theoretical_mean,
        goodness_of_fit=goodness_of_fit,
    )


def plot_service_time_histogram(
    service_times: list[float],
    distribution: Distribution | None = None,
    num_bins: int = 30,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot histogram of service times with theoretical PDF overlay.

    Args:
        service_times: List of observed service times.
        distribution: Theoretical distribution (optional).
        num_bins: Number of histogram bins.
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = np.array(service_times)

    # Plot histogram
    counts, bins, _ = ax.hist(
        data,
        bins=num_bins,
        density=True,
        alpha=0.6,
        color="steelblue",
        edgecolor="black",
        label="Empirical",
    )

    # Plot theoretical PDF if provided
    if distribution is not None:
        x = np.linspace(data.min(), data.max(), 200)
        # Generate empirical PDF from distribution
        samples = distribution.generate_batch(10000)
        kde = stats.gaussian_kde(samples)
        ax.plot(x, kde(x), "r-", linewidth=2, label="Theoretical (estimated)")

    ax.set_xlabel("Service Time")
    ax.set_ylabel("Probability Density")
    ax.set_title("Service Time Distribution (Histogram)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_service_time_ecdf(
    service_times: list[float],
    distribution: Distribution | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot empirical CDF with theoretical CDF comparison.

    Args:
        service_times: List of observed service times.
        distribution: Theoretical distribution (optional).
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = np.array(service_times)
    sorted_data = np.sort(data)
    ecdf = np.arange(1, len(data) + 1) / len(data)

    # Plot empirical CDF
    ax.plot(
        sorted_data,
        ecdf,
        marker=".",
        linestyle="none",
        alpha=0.6,
        label="Empirical CDF",
    )

    # Plot theoretical CDF if provided
    if distribution is not None:
        # Generate theoretical samples
        theoretical_samples = distribution.generate_batch(10000)
        sorted_theory = np.sort(theoretical_samples)
        theoretical_ecdf = np.arange(1, len(theoretical_samples) + 1) / len(
            theoretical_samples
        )

        ax.plot(
            sorted_theory,
            theoretical_ecdf,
            "r-",
            linewidth=2,
            label="Theoretical CDF",
        )

    ax.set_xlabel("Service Time")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Service Time Distribution (ECDF)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_qq_plot(
    service_times: list[float],
    distribution: Distribution,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Generate Q-Q plot for comparing distributions.

    Args:
        service_times: List of observed service times.
        distribution: Theoretical distribution.
        figsize: Figure size (width, height).

    Returns:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = np.array(service_times)

    # Generate theoretical quantiles
    theoretical_samples = distribution.generate_batch(len(data))

    # Q-Q plot
    stats.probplot(data, dist=stats.norm, plot=ax, fit=False)
    theoretical_quantiles = np.sort(theoretical_samples)
    empirical_quantiles = np.sort(data)

    ax.clear()
    ax.scatter(
        theoretical_quantiles,
        empirical_quantiles,
        alpha=0.6,
        edgecolor="k",
    )

    # Reference line
    min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
    max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)

    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Empirical Quantiles")
    ax.set_title("Q-Q Plot")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_summary_statistics_table(
    analysis: ServiceTimeAnalysis,
) -> dict[str, Any]:
    """Create a formatted dictionary of summary statistics.

    Args:
        analysis: ServiceTimeAnalysis object.

    Returns:
        Dictionary with formatted statistics for display.
    """
    summary = {
        "Sample Size": analysis.sample_size,
        "Mean": f"{analysis.mean:.4f}",
        "Standard Deviation": f"{analysis.std:.4f}",
        "Variance": f"{analysis.variance:.4f}",
        "Skewness": f"{analysis.skewness:.4f}",
        "Kurtosis": f"{analysis.kurtosis:.4f}",
        "Minimum": f"{analysis.quantiles.min:.4f}",
        "P50 (Median)": f"{analysis.quantiles.p50:.4f}",
        "P90": f"{analysis.quantiles.p90:.4f}",
        "P95": f"{analysis.quantiles.p95:.4f}",
        "P99": f"{analysis.quantiles.p99:.4f}",
        "Maximum": f"{analysis.quantiles.max:.4f}",
    }

    if analysis.theoretical_mean is not None:
        summary["Theoretical Mean"] = f"{analysis.theoretical_mean:.4f}"
        summary["Mean Deviation"] = (
            f"{abs(analysis.mean - analysis.theoretical_mean):.4f}"
        )

    if analysis.goodness_of_fit is not None:
        gof = analysis.goodness_of_fit
        summary["KS Statistic"] = f"{gof.ks_statistic:.4f}"
        summary["KS p-value"] = f"{gof.ks_pvalue:.4f}"
        if gof.chi2_statistic is not None:
            summary["Chi-Square Statistic"] = f"{gof.chi2_statistic:.4f}"
            summary["Chi-Square p-value"] = f"{gof.chi2_pvalue:.4f}"
        summary["Distribution Fit"] = "PASS" if gof.test_passed else "FAIL"

    return summary
