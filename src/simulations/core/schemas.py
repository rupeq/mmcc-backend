from typing import Union

from pydantic import Field, BaseModel, ConfigDict

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


DistributionParams = Union[
    ExponentialParams,
    UniformParams,
    TruncatedNormalParams,
    GammaParams,
    WeibullParams,
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
    """Represent a request to run a simulation."""

    num_channels: int = Field(..., gt=0, alias="numChannels")
    simulation_time: float = Field(..., gt=0, alias="simulationTime")
    num_replications: int = Field(1, ge=1, alias="numReplications")

    arrival_process: DistributionParams = Field(..., alias="arrivalProcess")
    service_process: DistributionParams = Field(..., alias="serviceProcess")

    # optional for nonstationary flow
    arrival_schedule: list[ArrivalScheduleItem] | None = Field(
        None, alias="arrivalSchedule"
    )

    random_seed: int | None = Field(None, alias="randomSeed")

    model_config = ConfigDict(
        populate_by_name=True,
    )


class SimulationResult(BaseModel):
    """Represent the results of a single simulation run."""

    metrics: SimulationMetrics
    gantt_chart: list[GanttChartItem]


class SimulationResponse(BaseModel):
    """Represent the response containing aggregated and individual simulation results."""

    aggregated_metrics: AggregatedMetrics
    replications: list[SimulationResult]
