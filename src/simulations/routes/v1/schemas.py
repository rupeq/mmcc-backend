import datetime
import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.simulations.core.schemas import (
    SimulationRequest,
    SimulationResponse,
    SweepResponse,
    TemporalProfile,
)
from src.simulations.models.enums import ReportStatus


class SimulationReport(BaseModel):
    id: uuid.UUID
    status: ReportStatus
    results: SimulationResponse | SweepResponse | None
    error_message: str | None
    configuration_id: uuid.UUID
    created_at: datetime.datetime
    completed_at: datetime.datetime | None
    is_active: bool

    model_config = ConfigDict(
        from_attributes=True,
    )


class SimulationConfigurationInfo(BaseModel):
    """Represent the information of a simulation configuration.

    Attributes:
        id: The unique identifier of the simulation configuration.
        name: The name of the simulation configuration.
        description: The description of the simulation configuration.
        created_at: The timestamp when the simulation configuration was created.
        updated_at: The timestamp when the simulation configuration was last updated.
    """

    id: uuid.UUID
    name: str | None = None
    description: str | None = None
    created_at: datetime.datetime | None = None
    updated_at: datetime.datetime | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )


class GetSimulationsResponse(BaseModel):
    """Represent the response for getting a list of simulations.

    Attributes:
        items: A list of simulation configurations.
        total_items: The total number of simulation configurations.
        total_pages: The total number of pages available.
        page: The current page number.
        limit: The number of items per page.
    """

    items: list[SimulationConfigurationInfo]
    total_items: int
    total_pages: int | None = None
    page: int | None = None
    limit: int | None = None


class CreateSimulationRequest(BaseModel):
    """Represent the request body for creating a new simulation.

    Attributes:
        name: A unique and descriptive name for the simulation configuration.
              Must be between 1 and 255 characters long.
        description: An optional detailed description of the
                     simulation's purpose or characteristics.
        simulation_parameters: The detailed parameters
                               required by the simulation
                               engine. This field is aliased
                               as "simulationParameters"
                               for API consumption.
    """

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    simulation_parameters: SimulationRequest = Field(
        ..., alias="simulationParameters"
    )

    model_config = ConfigDict(
        populate_by_name=True,
    )


class CreateSimulationResponse(BaseModel):
    """Represent the response body after successfully creating a simulation.
    Attributes:
        simulation_configuration_id: The unique identifier for the
                                     newly created simulation
                                     configuration.
        simulation_report_id: The unique identifier for the
                              initial simulation report generated
                              for this configuration.
        task_id: The background task ID for tracking execution status.
    """

    simulation_configuration_id: uuid.UUID
    simulation_report_id: uuid.UUID
    task_id: uuid.UUID


class GetSimulationConfigurationResponse(SimulationConfigurationInfo):
    """Represent the response for getting a single of simulation configuration."""

    pass


class GetSimulationConfigurationReportsResponse(BaseModel):
    """Represent the response for getting a list of reports."""

    reports: list[SimulationReport]


class GetSimulationConfigurationReportResponse(SimulationReport):
    """Represent the response for getting a single report."""

    pass


class CreateAnimationResponse(BaseModel):
    """Response after requesting an animation generation task."""

    task_id: uuid.UUID


class QuantileStatisticsResponse(BaseModel):
    """Response schema for quantile statistics.

    Attributes:
        p50: 50th percentile (median).
        p90: 90th percentile.
        p95: 95th percentile.
        p99: 99th percentile.
        min: Minimum value.
        max: Maximum value.
    """

    p50: float = Field(..., description="50th percentile (median)")
    p90: float = Field(..., description="90th percentile")
    p95: float = Field(..., description="95th percentile")
    p99: float = Field(..., description="99th percentile")
    min: float = Field(..., description="Minimum value")
    max: float = Field(..., description="Maximum value")

    model_config = ConfigDict(from_attributes=True)


class GoodnessOfFitResponse(BaseModel):
    """Response schema for goodness-of-fit test results.

    Attributes:
        ks_statistic: Kolmogorov-Smirnov test statistic.
        ks_pvalue: KS test p-value.
        chi2_statistic: Chi-square test statistic.
        chi2_pvalue: Chi-square p-value.
        test_passed: Whether the distribution fits at significance level.
        warnings: List of warnings about test validity.
    """

    ks_statistic: float = Field(..., description="KS test statistic")
    ks_pvalue: float = Field(..., description="KS test p-value")
    chi2_statistic: float | None = Field(
        None, description="Chi-square test statistic"
    )
    chi2_pvalue: float | None = Field(None, description="Chi-square p-value")
    test_passed: bool = Field(..., description="Whether distribution fits data")
    warnings: list[str] = Field(
        default_factory=list, description="Test validity warnings"
    )

    model_config = ConfigDict(from_attributes=True)


class ServiceTimeAnalysisResponse(BaseModel):
    """Response schema for complete service time analysis.

    Attributes:
        sample_size: Number of service time samples analyzed.
        mean: Sample mean.
        std: Sample standard deviation.
        variance: Sample variance.
        skewness: Sample skewness.
        kurtosis: Sample kurtosis.
        quantiles: Quantile statistics.
        theoretical_mean: Theoretical mean from distribution.
        goodness_of_fit: Goodness-of-fit test results.
        summary_table: Formatted summary statistics.
    """

    sample_size: int = Field(..., description="Number of samples")
    mean: float = Field(..., description="Sample mean")
    std: float = Field(..., description="Standard deviation")
    variance: float = Field(..., description="Sample variance")
    skewness: float = Field(..., description="Sample skewness")
    kurtosis: float = Field(..., description="Sample kurtosis")
    quantiles: QuantileStatisticsResponse = Field(
        ..., description="Quantile statistics"
    )
    theoretical_mean: float | None = Field(
        None, description="Theoretical mean from distribution"
    )
    goodness_of_fit: GoodnessOfFitResponse | None = Field(
        None, description="Goodness-of-fit test results"
    )
    summary_table: dict[str, Any] = Field(
        ..., description="Formatted summary statistics"
    )

    model_config = ConfigDict(from_attributes=True)


class PlotResponse(BaseModel):
    """Response schema for plot images.

    Attributes:
        plot_type: Type of plot (histogram, ecdf, qq_plot).
        image_base64: Base64-encoded PNG image.
        content_type: MIME type (image/png).
    """

    plot_type: str = Field(..., description="Type of plot")
    image_base64: str = Field(..., description="Base64-encoded image")
    content_type: str = Field(default="image/png", description="MIME type")


class ServiceTimeVisualizationsResponse(BaseModel):
    """Response schema for all service time visualizations.

    Attributes:
        histogram: Histogram plot with theoretical PDF overlay.
        ecdf: Empirical CDF plot with theoretical comparison.
        qq_plot: Q-Q plot for distribution comparison.
    """

    histogram: PlotResponse = Field(
        ..., description="Histogram with PDF overlay"
    )
    ecdf: PlotResponse = Field(..., description="ECDF comparison plot")
    qq_plot: PlotResponse | None = Field(
        None, description="Q-Q plot (requires theoretical distribution)"
    )


class GetTemporalAnalysis(TemporalProfile):
    """Represent Temporal analysis"""

    pass
