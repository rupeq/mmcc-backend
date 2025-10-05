import logging
import math
import base64
import io

from matplotlib import pyplot as plt
from sqlalchemy import Sequence, RowMapping, Row
from matplotlib.figure import Figure

from src.simulations.core.schemas import (
    ExponentialParams,
    UniformParams,
    GammaParams,
    WeibullParams,
    TruncatedNormalParams,
    EmpiricalParams,
)
from src.simulations.core.distributions import get_distribution, Distribution
from src.simulations.core.optimization import (
    multi_objective_optimization,
    CostFunction,
    gradient_descent_channels,
    minimize_cost,
    binary_search_channels,
)
from src.simulations.core.schemas import OptimizationRequest
from src.simulations.models.enums import ReportStatus
from src.simulations.models.simulations import SimulationConfiguration
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)
from src.simulations.routes.v1.schemas import (
    GetSimulationsResponse,
    SimulationConfigurationInfo,
    SimulationReport,
)


logger = logging.getLogger(__name__)


def parse_search_query(search: str | None) -> dict[str, str]:
    """Parse a search query string into a dictionary of filters.

    Args:
        search: The search query string, e.g., "key1:value1,key2:value2".

    Returns:
        A dictionary of filters.

    Raises:
        BadFilterFormat: If the search query format is invalid.
    """
    if not search:
        return {}

    filters = {}

    try:
        for item in search.split(","):
            key, value = item.split(":", 1)
            filters[key.strip()] = value.strip()
    except ValueError:
        raise BadFilterFormat()

    return filters


def verify_report_status_value(filters: dict[str, str]) -> None:
    """Verify if the report_status filter has a valid value.

    Args:
        filters: A dictionary of filters.

    Raises:
        InvalidReportStatus: If the report_status value is invalid.
    """
    if "report_status" not in filters:
        return None

    status_value = filters["report_status"].lower()
    valid_statuses = {s.value for s in ReportStatus}
    if status_value not in valid_statuses:
        raise InvalidReportStatus()

    filters["report_status"] = status_value
    return None


def validate_simulation_columns(columns: list[str] | None) -> list[str] | None:
    """Validate the requested simulation columns.

    Ensures that all requested columns are valid and adds 'id' if not present.

    Args:
        columns: A list of column names to validate.

    Returns:
        A list of valid column names, including 'id', or None if no columns were specified.

    Raises:
        InvalidColumn: If any of the requested columns are invalid.
    """
    if not columns:
        return None

    valid_columns = {c.name for c in SimulationConfiguration.__table__.columns}

    all_cols = []
    for col_group in columns:
        all_cols.extend(c.strip() for c in col_group.split(","))

    if "id" not in all_cols:
        all_cols.insert(0, "id")

    for col in all_cols:
        if col not in valid_columns:
            raise InvalidColumn()

    return all_cols


def get_simulations_response(
    configs: Sequence[Row | RowMapping],
    total_items: int,
    *,
    limit: int | None,
    page: int | None,
    columns: list[str] | None = None,
) -> GetSimulationsResponse:
    """Construct the GetSimulationsResponse object.

    Args:
        configs: A sequence of simulation configuration rows.
        total_items: The total number of items.
        limit: The number of items per page.
        page: The current page number.
        columns: A list of columns to include in the response.

    Returns:
        A GetSimulationsResponse object.
    """
    items_data = []
    fields_to_extract = (
        columns if columns else SimulationConfigurationInfo.model_fields.keys()
    )
    for config in configs:
        items_data.append(
            {
                field: getattr(config, field)
                for field in fields_to_extract
                if hasattr(config, field)
            }
        )

    return GetSimulationsResponse(
        items=items_data,
        total_items=total_items,
        total_pages=math.ceil(total_items / limit)
        if limit is not None
        else None,
        page=page,
        limit=limit,
    )


def _optimize_binary_search(request: OptimizationRequest):
    """Execute binary search optimization."""
    if request.target_rejection_prob is None:
        raise ValueError(
            "target_rejection_prob is required for binary_search optimization"
        )

    return binary_search_channels(
        base_request=request.base_request,
        target_rejection_prob=request.target_rejection_prob,
        max_channels=request.max_channels or 100,
        tolerance=request.tolerance or 0.01,
    )


def _optimize_cost_minimization(request: OptimizationRequest):
    """Execute cost minimization optimization."""
    if request.channel_cost is None:
        raise ValueError("channel_cost is required for cost_minimization")
    if request.rejection_penalty is None:
        raise ValueError("rejection_penalty is required for cost_minimization")

    cost_function = CostFunction(
        channel_cost=request.channel_cost,
        rejection_penalty=request.rejection_penalty,
    )

    return minimize_cost(
        base_request=request.base_request,
        cost_function=cost_function,
        max_channels=request.max_channels or 50,
    )


def _optimize_gradient_descent(request: OptimizationRequest):
    """Execute gradient descent optimization."""
    if (
        request.channel_cost is not None
        and request.rejection_penalty is not None
    ):
        objective = "cost"
        cost_function = CostFunction(
            channel_cost=request.channel_cost,
            rejection_penalty=request.rejection_penalty,
        )
    else:
        objective = "rejection"
        cost_function = None

    return gradient_descent_channels(
        base_request=request.base_request,
        objective=objective,
        cost_function=cost_function,
        initial_channels=request.max_channels,
        max_iterations=20,
    )


def _optimize_multi_objective(request: OptimizationRequest):
    """Execute multi-objective optimization."""
    if request.rejection_weight is None:
        raise ValueError("rejection_weight is required for multi_objective")
    if request.utilization_weight is None:
        raise ValueError("utilization_weight is required for multi_objective")
    if request.cost_weight is None:
        raise ValueError("cost_weight is required for multi_objective")
    if request.channel_cost is None:
        raise ValueError("channel_cost is required for multi_objective")
    if request.rejection_penalty is None:
        raise ValueError("rejection_penalty is required for multi_objective")

    return multi_objective_optimization(
        base_request=request.base_request,
        rejection_weight=request.rejection_weight,
        utilization_weight=request.utilization_weight,
        cost_weight=request.cost_weight,
        channel_cost=request.channel_cost,
        rejection_penalty=request.rejection_penalty,
        max_channels=request.max_channels or 50,
    )


def figure_to_base64(fig: Figure) -> str:
    """Convert matplotlib figure to base64-encoded PNG.

    Args:
        fig: Matplotlib figure object.

    Returns:
        Base64-encoded string of PNG image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def extract_service_times_from_report(
    report: SimulationReport,
) -> list[float]:
    """Extract service times from simulation report.

    Args:
        report: Simulation report with results.

    Returns:
        List of service times from all replications.

    Raises:
        ValueError: If report has no service times data.
    """
    if not report.results or "replications" not in report.results:
        raise ValueError("Report has no replication data")

    service_times = []
    for replication in report.results["replications"]:
        if "raw_service_times" in replication:
            raw_times = replication["raw_service_times"]
            if raw_times:
                service_times.extend(raw_times)

    if not service_times:
        raise ValueError(
            "No service times collected. Ensure simulation was run with "
            "collect_service_times=true"
        )

    return service_times


async def get_service_distribution_from_report(
    report: SimulationReport,
) -> Distribution | None:
    """Extract service distribution from simulation configuration.

    Args:
        report: Simulation report with configuration.

    Returns:
        Service distribution object or None if cannot be reconstructed.
    """
    try:
        config = await report.awaitable_attrs.configuration
        sim_params = config.simulation_parameters

        if "serviceProcess" not in sim_params:
            return None

        service_params = sim_params["serviceProcess"]

        dist_type = service_params.get("distribution")

        if dist_type == "exponential":
            params = ExponentialParams(**service_params)
        elif dist_type == "uniform":
            params = UniformParams(**service_params)
        elif dist_type == "gamma":
            params = GammaParams(**service_params)
        elif dist_type == "weibull":
            params = WeibullParams(**service_params)
        elif dist_type == "truncated_normal":
            params = TruncatedNormalParams(**service_params)
        elif dist_type == "empirical":
            params = EmpiricalParams(**service_params)
        else:
            return None

        return get_distribution(params)

    except Exception as e:
        logger.error(f"Failed to reconstruct service distribution: {e}")
        return None
