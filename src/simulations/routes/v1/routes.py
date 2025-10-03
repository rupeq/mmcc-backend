"""API routes for simulation configurations and reports.

This module provides RESTful endpoints for creating, retrieving, updating,
and deleting simulation configurations and their associated reports.
"""

import datetime
import logging
import uuid

from another_fastapi_jwt_auth import AuthJWT
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status

from src.simulations.core.engine import run_replications
from src.simulations.core.schemas import (
    OptimizationResultResponse,
    OptimizationRequest,
)
from src.simulations.routes.v1.rate_limiter import check_rate_limit
from src.simulations.tasks.simulations import run_simulation_task
from src.core.db_session import get_session
from src.simulations.db_utils.exceptions import (
    IdColumnRequiredException,
    SimulationNotFound,
    SimulationReportsNotFound,
    SimulationReportNotFound,
)
from src.simulations.models.enums import ReportStatus
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)
from src.simulations.routes.v1.schemas import (
    GetSimulationsResponse,
    CreateSimulationResponse,
    CreateSimulationRequest,
    GetSimulationConfigurationResponse,
    GetSimulationConfigurationReportsResponse,
    GetSimulationConfigurationReportResponse,
)
from src.simulations.db_utils.simulation_configurations import (
    get_simulation_configurations,
    get_simulation_configuration as get_simulation_configuration_from_db,
    create_simulation_configuration,
    delete_simulation_configuration as delete_simulation_configuration_from_db,
)
from src.simulations.db_utils.simulation_reports import (
    get_simulation_configuration_report as get_simulation_configuration_report_from_db,
    get_simulation_configuration_reports as get_simulation_configuration_reports_from_db,
    delete_simulation_configuration_report as delete_simulation_configuration_report_from_db,
    update_simulation_report_status,
)
from src.simulations.routes.v1.utils import (
    parse_search_query,
    validate_simulation_columns,
    verify_report_status_value,
    get_simulations_response,
    _optimize_binary_search,
    _optimize_cost_minimization,
    _optimize_gradient_descent,
    _optimize_multi_objective,
)

router = APIRouter(tags=["v1", "simulations"], prefix="/v1/simulations")
logger = logging.getLogger(__name__)


@router.get(
    "",
    response_model=GetSimulationsResponse,
    status_code=status.HTTP_200_OK,
    response_model_exclude_none=True,
)
async def get_simulations(
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    columns: list[str] | None = Query(
        None, description="Comma-separated list of columns to return."
    ),
    filters: str | None = Query(
        None,
        description="Search filters in key:value format, comma-separated.",
    ),
    page: int | None = Query(None, ge=1, description="Page number"),
    limit: int | None = Query(None, ge=1, le=100, description="Items per page"),
):
    """Retrieve simulation configurations for the authenticated user.

    Fetch a paginated list of simulation configurations belonging to the
    authenticated user, with optional filtering and column selection.

    Args:
        authorize: JWT authentication dependency.
        session: Asynchronous database session.
        columns: Optional list of column names to include in response.
        filters: Optional search filters in "key:value,key:value" format.
        page: Optional page number for pagination (1-indexed).
        limit: Optional maximum number of items per page (1-100).

    Returns:
        GetSimulationsResponse containing:
            - items: List of simulation configurations
            - total_items: Total count of matching configurations
            - total_pages: Total number of pages (if paginated)
            - page: Current page number
            - limit: Items per page

    Raises:
        HTTPException 400: If filters or columns are invalid.
        HTTPException 401: If user is not authenticated.
        HTTPException 500: If an internal server error occurs.

    Example:
        GET /api/v1/simulations?page=1&limit=10&filters=name:Test
    """
    authorize.jwt_required()

    try:
        filters = parse_search_query(filters)
        verify_report_status_value(filters)
    except BadFilterFormat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid search format. Use comma-separated key:value pairs.",
        )
    except InvalidReportStatus:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid report_status.",
        )

    try:
        columns = validate_simulation_columns(columns)
    except InvalidColumn:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid columns. Use comma-separated valid column names.",
        )

    try:
        configs, total_items = await get_simulation_configurations(
            session,
            user_id=authorize.get_jwt_subject(),
            columns=columns,
            filters=filters,
            page=page,
            limit=limit,
        )
    except IdColumnRequiredException:
        logger.exception(
            msg="Unexpected error: id must be in the columns list."
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    return get_simulations_response(
        configs, total_items, page=page, limit=limit, columns=columns
    )


@router.post(
    "",
    response_model=CreateSimulationResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_simulation(
    request: CreateSimulationRequest,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Create a new simulation configuration and start execution.

    This endpoint creates a simulation configuration and report in the database,
    then dispatches an asynchronous task to execute the simulation. The task
    is processed by a Celery worker and results are stored when complete.

    Args:
        request: Simulation configuration and parameters.
        authorize: JWT authentication dependency.
        session: Database session dependency.

    Returns:
        Response containing configuration ID and report ID for tracking.

    Raises:
        HTTPException 401: If authentication fails.
        HTTPException 500: If task dispatch fails.
    """
    authorize.jwt_required()

    configuration, report = await create_simulation_configuration(
        session,
        user_id=authorize.get_jwt_subject(),
        name=request.name,
        description=request.description,
        simulation_parameters=request.simulation_parameters.model_dump(
            by_alias=True
        ),
    )

    try:
        task = run_simulation_task.delay(
            report_id=str(report.id),
            simulation_params=request.simulation_parameters.model_dump(
                by_alias=True
            ),
        )

        logger.info(
            "Dispatched simulation task: report_id=%s, task_id=%s",
            report.id,
            task.id,
        )

    except Exception as e:
        logger.error(
            "Failed to dispatch simulation task for report %s: %s",
            report.id,
            e,
            exc_info=True,
        )

        try:
            await update_simulation_report_status(
                session,
                report_id=report.id,
                status=ReportStatus.FAILED,
                error_message=f"Task dispatch failed: {str(e)[:500]}",
                completed_at=datetime.datetime.now(datetime.timezone.utc),
            )
        except Exception as db_error:
            logger.error(
                "Failed to update report after dispatch failure: %s", db_error
            )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Simulation service temporarily unavailable",
        )

    return CreateSimulationResponse(
        simulation_configuration_id=configuration.id,
        simulation_report_id=report.id,
    )


@router.get(
    "/{simulation_configuration_id}",
    response_model=GetSimulationConfigurationResponse,
    status_code=status.HTTP_200_OK,
)
async def get_simulation_configuration(
    simulation_configuration_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve a specific simulation configuration by ID.

    Fetch detailed information about a simulation configuration belonging
    to the authenticated user.

    Args:
        simulation_configuration_id: UUID of the configuration to retrieve.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        GetSimulationConfigurationResponse containing configuration details.

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If configuration not found or doesn't belong to user.

    Example:
        GET /api/v1/simulations/{uuid}
    """
    authorize.jwt_required()
    try:
        return await get_simulation_configuration_from_db(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation configuration not found",
        )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}",
    response_model=GetSimulationConfigurationReportResponse,
    status_code=status.HTTP_200_OK,
)
async def get_simulation_configuration_report(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve a specific simulation report by ID.

    Fetch detailed information about a simulation report, including its
    status, results, and any error messages.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the report to retrieve.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        GetSimulationConfigurationReportResponse containing:
            - id: Report UUID
            - status: Report status (PENDING, COMPLETED, FAILED)
            - results: Simulation results (if completed)
            - error_message: Error details (if failed)
            - created_at: Creation timestamp
            - completed_at: Completion timestamp (if applicable)

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If report not found or doesn't belong to user.

    Example:
        GET /api/v1/simulations/{config_uuid}/reports/{report_uuid}
    """
    authorize.jwt_required()
    try:
        return await get_simulation_configuration_report_from_db(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            report_id=report_id,
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation report not found",
        )


@router.get(
    "/{simulation_configuration_id}/reports",
    response_model=GetSimulationConfigurationReportsResponse,
    status_code=status.HTTP_200_OK,
)
async def get_simulation_configuration_reports(
    simulation_configuration_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Retrieve all reports for a specific simulation configuration.

    Fetch all simulation reports associated with a given configuration,
    including historical runs and their results.

    Args:
        simulation_configuration_id: UUID of the configuration.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        GetSimulationConfigurationReportsResponse containing:
            - reports: List of all reports for this configuration

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If configuration not found or has no reports.

    Example:
        GET /api/v1/simulations/{config_uuid}/reports
    """
    authorize.jwt_required()
    try:
        return {
            "reports": await get_simulation_configuration_reports_from_db(
                session,
                user_id=uuid.UUID(authorize.get_jwt_subject()),
                simulation_configuration_id=simulation_configuration_id,
            )
        }
    except SimulationReportsNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation reports not found",
        )


@router.delete(
    "/{simulation_configuration_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_simulation_configuration(
    simulation_configuration_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Delete a simulation configuration and all associated reports.

    Soft delete (mark as inactive) a simulation configuration and cascade
    the deletion to all associated reports. This operation is irreversible.

    Args:
        simulation_configuration_id: UUID of the configuration to delete.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        None (204 No Content on success).

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If configuration not found or doesn't belong to user.

    Example:
        DELETE /api/v1/simulations/{uuid}
    """
    authorize.jwt_required()
    try:
        await delete_simulation_configuration_from_db(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation configuration not found",
        )
    return None


@router.delete(
    "/{simulation_configuration_id}/reports/{report_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_simulation_configuration_report(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """Delete a specific simulation report.

    Soft delete (mark as inactive) a simulation report. The parent
    configuration remains active. This operation is irreversible.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the report to delete.
        authorize: JWT authentication dependency.
        session: Asynchronous database session.

    Returns:
        None (204 No Content on success).

    Raises:
        HTTPException 401: If user is not authenticated.
        HTTPException 404: If report not found or doesn't belong to user.

    Example:
        DELETE /api/v1/simulations/{config_uuid}/reports/{report_uuid}
    """
    authorize.jwt_required()
    try:
        await delete_simulation_configuration_report_from_db(
            session,
            report_id=report_id,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            simulation_configuration_id=simulation_configuration_id,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation report not found",
        )
    return None


@router.post(
    "/optimize",
    response_model=OptimizationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Optimize channel configuration",
    description="""
    Find optimal number of channels based on different objectives:
    - **binary_search**: Find minimum channels for target rejection probability
    - **cost_minimization**: Minimize total cost (channel cost + rejection penalty)
    - **gradient_descent**: Use gradient descent for flexible optimization
    - **multi_objective**: Balance rejection, utilization, and cost
    """,
)
async def optimize_channels(
    request: OptimizationRequest,
    authorize: AuthJWT = Depends(),
):
    """Optimize channel configuration using various algorithms.

    This endpoint runs optimization algorithms to find the optimal number
    of channels based on the specified objective. Each optimization type
    requires different parameters.

    **Binary Search Parameters:**
    - target_rejection_prob (required): Target maximum rejection probability
    - max_channels (optional): Maximum channels to consider (default: 100)
    - tolerance (optional): Acceptable tolerance (default: 0.01)

    **Cost Minimization Parameters:**
    - channel_cost (required): Cost per channel
    - rejection_penalty (required): Penalty per rejected request
    - max_channels (optional): Maximum channels to evaluate

    **Gradient Descent Parameters:**
    - channel_cost (optional): For cost objective
    - rejection_penalty (optional): For cost objective
    - max_channels (optional): Starting point hint

    **Multi-Objective Parameters:**
    - rejection_weight (required): Weight for rejection (0-1)
    - utilization_weight (required): Weight for utilization (0-1)
    - cost_weight (required): Weight for cost (0-1)
    - channel_cost (required): Cost per channel
    - rejection_penalty (required): Penalty per rejected request

    Args:
        request: Optimization configuration and parameters.
        authorize: JWT authentication dependency.

    Returns:
        OptimizationResultResponse containing:
            - optimal_channels: Recommended number of channels
            - achieved_rejection_prob: Rejection probability with optimal channels
            - achieved_utilization: Channel utilization with optimal channels
            - throughput: System throughput
            - total_cost: Total cost (if applicable)
            - iterations: Number of iterations performed
            - convergence_history: Optimization trajectory

    Raises:
        HTTPException 400: If required parameters are missing.
        HTTPException 401: If authentication fails.
        HTTPException 408: If optimization times out.
        HTTPException 422: If parameters are invalid.

    Example:
        ```json
        {
          "base_request": {
            "numChannels": 1,
            "simulationTime": 100,
            "numReplications": 10,
            "arrivalProcess": {"distribution": "exponential", "rate": 5.0},
            "serviceProcess": {"distribution": "exponential", "rate": 10.0}
          },
          "optimizationType": "binary_search",
          "targetRejectionProb": 0.05,
          "maxChannels": 20
        }
        ```
    """
    authorize.jwt_required()
    check_rate_limit(
        user_id=authorize.get_jwt_subject(),
        max_per_hour=5,
        window_seconds=3600,
        fail_open=True,
    )

    logger.info(
        "Optimization request: type=%s, user=%s",
        request.optimization_type,
        authorize.get_jwt_subject(),
    )

    try:
        if request.optimization_type == "binary_search":
            result = _optimize_binary_search(request)
        elif request.optimization_type == "cost_minimization":
            result = _optimize_cost_minimization(request)
        elif request.optimization_type == "gradient_descent":
            result = _optimize_gradient_descent(request)
        elif request.optimization_type == "multi_objective":
            result = _optimize_multi_objective(request)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown optimization type: {request.optimization_type}",
            )

        logger.info(
            "Optimization complete: type=%s, optimal_channels=%d, iterations=%d",
            request.optimization_type,
            result.optimal_channels,
            result.iterations,
        )

        return OptimizationResultResponse.model_validate(
            result, from_attributes=True
        )

    except ValueError as e:
        logger.error("Optimization validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid optimization parameters: {str(e)}",
        )
    except TimeoutError:
        logger.error("Optimization timeout")
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Optimization took too long. Try reducing max_channels or simulation_time.",
        )
    except Exception as e:
        logger.exception("Optimization failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Optimization failed. Please check your parameters and try again.",
        )


@router.post(
    "/compare-theory",
    status_code=status.HTTP_200_OK,
    summary="Compare simulation with theoretical predictions",
    description="""
    Run simulation and compare results with theoretical predictions (when available).

    Currently supports:
    - **M/M/c/c systems**: Exponential arrivals and service (Erlang B formula)

    Returns simulation results alongside theoretical predictions with error analysis.
    """,
)
async def compare_with_theory(
    request: CreateSimulationRequest,
    authorize: AuthJWT = Depends(),
):
    """Compare simulation results with theoretical predictions.

    This endpoint:
    1. Runs the simulation (in-process for immediate results)
    2. Calculates theoretical metrics (if system is M/M/c/c)
    3. Compares simulation vs theory
    4. Returns detailed comparison with error analysis

    **Supported Systems:**
    - M/M/c/c: Exponential inter-arrival and service times

    **Unsupported Systems:**
    - Non-exponential distributions
    - Non-stationary arrivals
    - G/G/c/c systems (no closed-form solution)

    Args:
        request: Simulation configuration.
        authorize: JWT authentication dependency.

    Returns:
        Dictionary containing:
            - simulation_results: Aggregated simulation metrics
            - theoretical_results: Theoretical predictions
            - comparison: Detailed error analysis

    Raises:
        HTTPException 401: If authentication fails.
        HTTPException 429: If rate limit exceeded.
        HTTPException 422: If parameters are invalid.

    Example:
        ```json
        {
          "name": "M/M/3/3 Validation",
          "simulationParameters": {
            "numChannels": 3,
            "simulationTime": 1000,
            "numReplications": 30,
            "arrivalProcess": {"distribution": "exponential", "rate": 5.0},
            "serviceProcess": {"distribution": "exponential", "rate": 10.0},
            "randomSeed": 42
          }
        }
        ```
    """
    authorize.jwt_required()

    check_rate_limit(
        user_id=authorize.get_jwt_subject(),
        max_per_hour=20,
        window_seconds=3600,
        fail_open=True,
    )

    logger.info(
        "Theoretical comparison request: user=%s, channels=%d",
        authorize.get_jwt_subject(),
        request.simulation_parameters.num_channels,
    )

    try:
        from src.simulations.core.theoretical import (
            calculate_theoretical_metrics,
            compare_with_theory as compare_metrics,
        )

        theoretical_metrics = calculate_theoretical_metrics(
            request.simulation_parameters
        )

        simulation_response = run_replications(request.simulation_parameters)

        agg = simulation_response.aggregated_metrics
        sim_metrics_dict = {
            "rejection_probability": agg.rejection_probability,
            "avg_channel_utilization": agg.avg_channel_utilization,
            "throughput": agg.throughput,
            "total_requests": agg.total_requests,
            "processed_requests": agg.processed_requests,
            "rejected_requests": agg.rejected_requests,
        }

        if agg.rejection_probability_ci:
            sim_metrics_dict["rejection_probability_ci"] = {
                "lower_bound": agg.rejection_probability_ci.lower_bound,
                "upper_bound": agg.rejection_probability_ci.upper_bound,
            }

        if agg.avg_channel_utilization_ci:
            sim_metrics_dict["avg_channel_utilization_ci"] = {
                "lower_bound": agg.avg_channel_utilization_ci.lower_bound,
                "upper_bound": agg.avg_channel_utilization_ci.upper_bound,
            }

        if agg.throughput_ci:
            sim_metrics_dict["throughput_ci"] = {
                "lower_bound": agg.throughput_ci.lower_bound,
                "upper_bound": agg.throughput_ci.upper_bound,
            }

        comparison = compare_metrics(sim_metrics_dict, theoretical_metrics)

        logger.info(
            "Comparison complete: system=%s, applicable=%s",
            theoretical_metrics.system_type,
            theoretical_metrics.is_applicable,
        )

        return {
            "simulation_results": simulation_response,
            "theoretical_results": {
                "applicable": theoretical_metrics.is_applicable,
                "system_type": theoretical_metrics.system_type,
                "traffic_intensity": theoretical_metrics.traffic_intensity,
                "blocking_probability": theoretical_metrics.blocking_probability,
                "utilization": theoretical_metrics.utilization,
                "throughput": theoretical_metrics.throughput,
                "mean_service_time": theoretical_metrics.mean_service_time,
                "mean_interarrival_time": theoretical_metrics.mean_interarrival_time,
                "assumptions": theoretical_metrics.assumptions,
                "warnings": theoretical_metrics.warnings,
            },
            "comparison": comparison,
        }

    except Exception as e:
        logger.exception("Theoretical comparison failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}",
        )
