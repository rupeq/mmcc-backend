from typing import Union, Literal

from pydantic import Field, BaseModel, ConfigDict, model_validator

from src.simulations.core.enums import DistributionType


class ExponentialParams(BaseModel):
    """Represent parameters for an exponential distribution."""

    distribution: DistributionType = DistributionType.EXPONENTIAL
    rate: float = Field(..., gt=0, description="Rate (lambda)")


class UniformParams(BaseModel):
    """Represent parameters for a uniform distribution."""

    distribution: DistributionType = DistributionType.UNIFORM
    a: float = Field(..., description="Lower bound")
    b: float = Field(..., ge=0, description="Upper bound")


class TruncatedNormalParams(BaseModel):
    """Represent parameters for a truncated normal distribution."""

    distribution: DistributionType = DistributionType.TRUNCATED_NORMAL
    mu: float = Field(..., description="Mean (mu)")
    sigma: float = Field(..., gt=0, description="Standard deviation (sigma)")
    a: float = Field(0, description="Lower threshold")
    b: float = Field(float("inf"), description="Upper threshold")


class GammaParams(BaseModel):
    """Represent parameters for a gamma distribution."""

    distribution: DistributionType = DistributionType.GAMMA
    k: float = Field(..., gt=0, description="Form (k)")
    theta: float = Field(..., gt=0, description="Scale (theta)")


class WeibullParams(BaseModel):
    """Represent parameters for a Weibull distribution."""

    distribution: DistributionType = DistributionType.WEIBULL
    k: float = Field(..., gt=0, description="Form (k)")
    lambda_param: float = Field(..., gt=0, description="Scale (lambda)")


class EmpiricalParams(BaseModel):
    """Represent parameters for an empirical distribution.

    Use this to specify a custom distribution based on observed data.
    The simulator will use kernel density estimation (KDE) or
    inverse transform sampling to generate random variates.

    Attributes:
        distribution: Type identifier.
        data: List of observed values (must have at least 2 points).
        method: Sampling method ('kde' or 'inverse_transform').

    Examples:
        >>> # Service times from real observations
        >>> params = EmpiricalParams(
        ...     data=[1.2, 1.5, 2.1, 1.8, 2.3, 1.9],
        ...     method="kde"
        ... )
    """

    distribution: DistributionType = DistributionType.EMPIRICAL
    data: list[float] = Field(
        ...,
        min_length=2,
        description="Observed data points (minimum 2 values)",
    )
    method: Literal["kde", "inverse_transform"] = Field(
        default="inverse_transform",
        description="Sampling method: 'kde' (kernel density) or 'inverse_transform' (ECDF)",
    )
    bandwidth: float | None = Field(
        default=None,
        gt=0,
        description="KDE bandwidth (optional, auto-detected if None)",
    )

    @model_validator(mode="after")
    def validate_data_values(self):
        """Validate empirical data."""
        import logging

        if len(self.data) < 2:
            raise ValueError("Empirical data must contain at least 2 values")

        if any(x < 0 for x in self.data):
            logging.getLogger(__name__).warning(
                "Empirical data contains negative values. "
                "This may not be appropriate for inter-arrival or service times."
            )

        if len(set(self.data)) < len(self.data) * 0.5:
            logging.getLogger(__name__).warning(
                "Empirical data has many duplicate values (>50%%). "
                "Consider using a parametric distribution instead."
            )

        if len(self.data) < 30:
            logging.getLogger(__name__).warning(
                "Small empirical sample size (%d points). "
                "Results may be unreliable. Consider collecting more data.",
                len(self.data),
            )

        return self


DistributionParams = Union[
    ExponentialParams,
    UniformParams,
    TruncatedNormalParams,
    GammaParams,
    WeibullParams,
    EmpiricalParams,
]


class ArrivalScheduleItem(BaseModel):
    """Represent an item in an arrival schedule."""

    duration: float = Field(..., gt=0, description="Interval duration")
    rate: float = Field(
        ..., gt=0, description="Rate (lambda) on a given interval"
    )


class GanttChartItem(BaseModel):
    """Represent an item in a Gantt chart."""

    channel: int
    start: float
    end: float
    duration: float


class SimulationMetrics(BaseModel):
    """Represent the metrics collected from a single simulation run."""

    total_requests: int
    processed_requests: int
    rejected_requests: int
    rejection_probability: float
    avg_channel_utilization: float
    throughput: float


class ConfidenceInterval(BaseModel):
    """Represent a confidence interval with lower and upper bounds."""

    lower_bound: float
    upper_bound: float


class AggregatedMetrics(BaseModel):
    """Hold the aggregated metrics from multiple replications."""

    num_replications: int

    # Mean values
    total_requests: float
    processed_requests: float
    rejected_requests: float
    rejection_probability: float
    avg_channel_utilization: float
    throughput: float

    # Confidence intervals for key metrics (optional for n < 2)
    rejection_probability_ci: ConfidenceInterval | None = None
    avg_channel_utilization_ci: ConfidenceInterval | None = None
    throughput_ci: ConfidenceInterval | None = None


class SimulationRequest(BaseModel):
    num_channels: int = Field(..., gt=0, alias="numChannels")
    simulation_time: float = Field(..., gt=0, alias="simulationTime")
    num_replications: int = Field(1, ge=1, alias="numReplications")
    arrival_process: DistributionParams = Field(..., alias="arrivalProcess")
    service_process: DistributionParams = Field(..., alias="serviceProcess")
    arrival_schedule: list[ArrivalScheduleItem] | None = Field(
        None, alias="arrivalSchedule"
    )
    random_seed: int | None = Field(None, alias="randomSeed")

    collect_gantt_data: bool = Field(
        True,
        alias="collectGanttData",
        description="Collect Gantt chart data (memory intensive for long simulations)",
    )
    collect_service_times: bool = Field(
        True,
        alias="collectServiceTimes",
        description="Collect raw service time samples",
    )
    max_gantt_items: int | None = Field(
        10000,
        alias="maxGanttItems",
        ge=0,
        description="Maximum Gantt items to collect (None = unlimited, 0 = disabled)",
    )
    max_service_time_samples: int | None = Field(
        1000,
        alias="maxServiceTimeSamples",
        ge=0,
        description="Maximum service time samples to collect (None = unlimited, 0 = disabled)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @model_validator(mode="after")
    def validate_arrival_schedule(self):
        if not self.arrival_schedule:
            return self

        total_duration = sum(item.duration for item in self.arrival_schedule)
        if total_duration < self.simulation_time:
            raise ValueError(
                f"Arrival schedule duration ({total_duration:.2f}s) is less than "
                f"simulation time ({self.simulation_time:.2f}s)"
            )

        for i, item in enumerate(self.arrival_schedule):
            if item.duration <= 0:
                raise ValueError(f"Schedule item {i} has non-positive duration")
            if item.rate <= 0:
                raise ValueError(f"Schedule item {i} has non-positive rate")

        if len(self.arrival_schedule) > 1000:
            import logging

            logging.getLogger(__name__).warning(
                "Very large arrival schedule (%d intervals) may impact performance",
                len(self.arrival_schedule),
            )

        return self

    @model_validator(mode="after")
    def validate_data_collection_limits(self):
        """Validate data collection configuration"""
        if not self.collect_gantt_data and self.max_gantt_items != 0:
            self.max_gantt_items = 0

        if (
            not self.collect_service_times
            and self.max_service_time_samples != 0
        ):
            self.max_service_time_samples = 0

        if self.max_gantt_items and self.max_gantt_items > 100000:
            import logging

            logging.getLogger(__name__).warning(
                "Very high max_gantt_items (%d) may cause memory issues",
                self.max_gantt_items,
            )

        return self


class SimulationResult(BaseModel):
    """Represent the results of a single simulation run."""

    metrics: SimulationMetrics
    gantt_chart: list[GanttChartItem]
    raw_service_times: list[float] | None = None


class SimulationResponse(BaseModel):
    """Represent the response containing aggregated and individual simulation results."""

    aggregated_metrics: AggregatedMetrics
    replications: list[SimulationResult]


class SweepParameter(BaseModel):
    """Define the parameter to be varied in a sweep experiment."""

    name: str = Field(
        ...,
        description="Name of the parameter to sweep, using dot notation for nested fields.",
        examples=["num_channels", "arrival_process.rate"],
    )
    values: list[Union[int, float]] = Field(
        ...,
        min_length=1,
        description="A list of values to use for the specified parameter.",
    )


class SweepRequest(BaseModel):
    """Represent a request to perform a parameter sweep simulation."""

    base_request: SimulationRequest
    sweep_parameter: SweepParameter = Field(..., alias="sweepParameter")

    model_config = ConfigDict(
        populate_by_name=True,
    )


class SweepResultItem(BaseModel):
    """Represent the simulation results for a single parameter value in a sweep."""

    parameter_value: Union[int, float, str]
    result: SimulationResponse


class SweepResponse(BaseModel):
    """Represent the complete response for a parameter sweep experiment."""

    results: list[SweepResultItem]


class OptimizationRequest(BaseModel):
    """Request for channel optimization"""

    base_request: SimulationRequest
    optimization_type: Literal[
        "binary_search",
        "cost_minimization",
        "gradient_descent",
        "multi_objective",
    ] = Field(..., alias="optimizationType")

    # Binary search parameters
    target_rejection_prob: float | None = Field(
        None, alias="targetRejectionProb", ge=0, le=1
    )
    max_channels: int | None = Field(None, alias="maxChannels", ge=1, le=1000)
    tolerance: float | None = Field(None, ge=0, le=1)

    # Cost parameters
    channel_cost: float | None = Field(None, alias="channelCost", ge=0)
    rejection_penalty: float | None = Field(
        None, alias="rejectionPenalty", ge=0
    )

    # Multi-objective weights
    rejection_weight: float | None = Field(
        None, alias="rejectionWeight", ge=0, le=1
    )
    utilization_weight: float | None = Field(
        None, alias="utilizationWeight", ge=0, le=1
    )
    cost_weight: float | None = Field(None, alias="costWeight", ge=0, le=1)

    model_config = ConfigDict(populate_by_name=True)


class OptimizationResultResponse(BaseModel):
    """Response with optimization results"""

    optimal_channels: int
    achieved_rejection_prob: float
    achieved_utilization: float
    throughput: float
    total_cost: float | None = None
    iterations: int
    convergence_history: list[dict] | None = None

    model_config = ConfigDict(from_attributes=True)

    def plot_convergence(self) -> str:
        """Generate base64-encoded convergence plot for frontend"""
        if not self.convergence_history:
            return None

        import matplotlib.pyplot as plt
        import io
        import base64

        iterations = [h["iteration"] for h in self.convergence_history]
        values = [h["objective_value"] for h in self.convergence_history]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(iterations, values, marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective Function Value")
        ax.set_title("Optimization Convergence")
        ax.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode()
        plt.close()

        return f"data:image/png;base64,{plot_base64}"
