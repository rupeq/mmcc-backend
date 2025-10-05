import io
import logging
import uuid
from pathlib import Path

from another_fastapi_jwt_auth import AuthJWT
from celery.result import AsyncResult
from fastapi import APIRouter, Depends, Query, HTTPException, BackgroundTasks
from matplotlib import pyplot as plt
from sqlalchemy.ext.asyncio import AsyncSession
from starlette import status
from starlette.responses import Response, FileResponse

from src.background_tasks.db_utils.background_tasks import (
    get_background_task,
    create_background_task,
)
from src.background_tasks.models.enums import TaskType
from src.core.db_session import get_session
from src.simulations.core.schemas import GanttChartItem
from src.simulations.core.service_time_analysis import (
    analyze_service_times,
    create_summary_statistics_table,
    plot_service_time_histogram,
    plot_service_time_ecdf,
    plot_qq_plot,
)
from src.simulations.core.visualization import SimulationVisualizer
from src.simulations.db_utils.exceptions import (
    SimulationReportNotFound,
    BackgroundTaskNotFound,
)
from src.simulations.db_utils.simulation_reports import (
    get_simulation_configuration_report,
)
from src.simulations.models.enums import ReportStatus
from src.simulations.routes.v1.schemas import (
    ServiceTimeAnalysisResponse,
    ServiceTimeVisualizationsResponse,
    PlotResponse,
    QuantileStatisticsResponse,
    GoodnessOfFitResponse,
    CreateAnimationResponse,
)
from src.simulations.routes.v1.utils import (
    extract_service_times_from_report,
    get_service_distribution_from_report,
    figure_to_base64,
)
from src.simulations.tasks.animation import generate_animation_task
from src.simulations.worker import celery_app


router = APIRouter(
    tags=["v1", "simulations", "analysis"], prefix="/v1/simulations"
)
logger = logging.getLogger(__name__)


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/service-time-analysis",
    response_model=ServiceTimeAnalysisResponse,
    summary="Analyze service time distribution",
    description="Perform comprehensive statistical analysis on service times",
)
async def get_service_time_analysis(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    significance_level: float = Query(
        0.05, ge=0.01, le=0.1, description="Significance level for tests"
    ),
) -> ServiceTimeAnalysisResponse:
    """Perform statistical analysis on service time distribution.

    This endpoint analyzes the service times collected during simulation
    and provides comprehensive statistics including:

    - **Descriptive Statistics**: Mean, std, variance, skewness, kurtosis
    - **Quantiles**: P50, P90, P95, P99, min, max
    - **Goodness-of-Fit Tests**: KS test and Chi-square test
    - **Comparison**: Observed vs theoretical distribution

    **Requirements:**
    - Simulation must be completed (status = COMPLETED)
    - Simulation must have been run with `collect_service_times=true`

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the simulation report.
        authorize: JWT authentication dependency.
        session: Database session dependency.
        significance_level: Significance level for hypothesis tests (default: 0.05).

    Returns:
        ServiceTimeAnalysisResponse with complete statistical analysis.

    Raises:
        HTTPException 400: If service times were not collected.
        HTTPException 404: If report not found or not completed.

    Example:
        ```
        GET /api/v1/simulations/{config_id}/reports/{report_id}/service-time-analysis

        Response:
        {
            "sample_size": 1000,
            "mean": 2.05,
            "std": 0.98,
            "variance": 0.96,
            "skewness": 0.12,
            "kurtosis": -0.34,
            "quantiles": {
                "p50": 2.01,
                "p90": 3.45,
                "p95": 3.89,
                "p99": 4.67,
                "min": 0.23,
                "max": 5.12
            },
            "theoretical_mean": 2.0,
            "goodness_of_fit": {
                "ks_statistic": 0.023,
                "ks_pvalue": 0.342,
                "chi2_statistic": 15.67,
                "chi2_pvalue": 0.478,
                "test_passed": true,
                "warnings": []
            }
        }
        ```
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        report = await get_simulation_configuration_report(
            session=session,
            simulation_configuration_id=simulation_configuration_id,
            report_id=report_id,
            user_id=user_id,
            status=ReportStatus.COMPLETED,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation report {report_id} not found or not completed yet.",
        )

    try:
        service_times = extract_service_times_from_report(report)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    distribution = await get_service_distribution_from_report(report)
    analysis = analyze_service_times(
        service_times=service_times,
        distribution=distribution,
        significance_level=significance_level,
    )
    summary_table = create_summary_statistics_table(analysis)

    return ServiceTimeAnalysisResponse(
        sample_size=analysis.sample_size,
        mean=analysis.mean,
        std=analysis.std,
        variance=analysis.variance,
        skewness=analysis.skewness,
        kurtosis=analysis.kurtosis,
        quantiles=QuantileStatisticsResponse(**analysis.quantiles.__dict__),
        theoretical_mean=analysis.theoretical_mean,
        goodness_of_fit=(
            GoodnessOfFitResponse(**analysis.goodness_of_fit.__dict__)
            if analysis.goodness_of_fit
            else None
        ),
        summary_table=summary_table,
    )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/service-time-visualizations",
    response_model=ServiceTimeVisualizationsResponse,
    summary="Get service time visualizations",
    description="Generate all service time distribution plots",
)
async def get_service_time_visualizations(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    num_bins: int = Query(
        30, ge=10, le=100, description="Number of bins for histogram"
    ),
) -> ServiceTimeVisualizationsResponse:
    """Generate all service time distribution visualizations.

    This endpoint generates three types of plots:

    1. **Histogram**: Empirical distribution with theoretical PDF overlay
    2. **ECDF Plot**: Empirical CDF compared to theoretical CDF
    3. **Q-Q Plot**: Quantile-quantile plot for distribution comparison

    All plots are returned as base64-encoded PNG images suitable for
    direct embedding in HTML or displaying in frontend applications.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the simulation report.
        authorize: JWT authentication dependency.
        session: Database session dependency.
        num_bins: Number of bins for histogram (default: 30).

    Returns:
        ServiceTimeVisualizationsResponse with all plots as base64 strings.

    Raises:
        HTTPException 400: If service times were not collected.
        HTTPException 404: If report not found or not completed yet.

    Example:
        ```
        GET /api/v1/simulations/{config_id}/reports/{report_id}/service-time-visualizations

        Response:
        {
            "histogram": {
                "plot_type": "histogram",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "content_type": "image/png"
            },
            "ecdf": {
                "plot_type": "ecdf",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "content_type": "image/png"
            },
            "qq_plot": {
                "plot_type": "qq_plot",
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "content_type": "image/png"
            }
        }
        ```
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        report = await get_simulation_configuration_report(
            session=session,
            simulation_configuration_id=simulation_configuration_id,
            report_id=report_id,
            user_id=user_id,
            status=ReportStatus.COMPLETED,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation report {report_id} not found or not completed yet.",
        )

    try:
        service_times = extract_service_times_from_report(report)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    distribution = await get_service_distribution_from_report(report)
    histogram_fig = plot_service_time_histogram(
        service_times=service_times,
        distribution=distribution,
        num_bins=num_bins,
    )
    histogram_base64 = figure_to_base64(histogram_fig)

    ecdf_fig = plot_service_time_ecdf(
        service_times=service_times,
        distribution=distribution,
    )
    ecdf_base64 = figure_to_base64(ecdf_fig)

    qq_plot_base64 = None
    if distribution is not None:
        qq_fig = plot_qq_plot(
            service_times=service_times,
            distribution=distribution,
        )
        qq_plot_base64 = figure_to_base64(qq_fig)

    return ServiceTimeVisualizationsResponse(
        histogram=PlotResponse(
            plot_type="histogram",
            image_base64=histogram_base64,
        ),
        ecdf=PlotResponse(
            plot_type="ecdf",
            image_base64=ecdf_base64,
        ),
        qq_plot=(
            PlotResponse(
                plot_type="qq_plot",
                image_base64=qq_plot_base64,
            )
            if qq_plot_base64
            else None
        ),
    )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/service-time-histogram",
    response_class=Response,
    summary="Get service time histogram",
    description="Generate histogram plot as PNG image",
)
async def get_service_time_histogram_image(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    num_bins: int = Query(30, ge=10, le=100),
) -> Response:
    """Get service time histogram as PNG image.

    Returns the histogram plot as a binary PNG image, suitable for
    direct display in browsers or download.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the simulation report.
        authorize: JWT authentication dependency.
        session: Database session dependency.
        num_bins: Number of bins for histogram.

    Returns:
        PNG image (binary response).
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        report = await get_simulation_configuration_report(
            session=session,
            simulation_configuration_id=simulation_configuration_id,
            report_id=report_id,
            user_id=user_id,
            status=ReportStatus.COMPLETED,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation report {report_id} not found or not completed yet.",
        )

    try:
        service_times = extract_service_times_from_report(report)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    distribution = await get_service_distribution_from_report(report)

    fig = plot_service_time_histogram(
        service_times=service_times,
        distribution=distribution,
        num_bins=num_bins,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": f"inline; filename=histogram_{report_id}.png"
        },
    )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/service-time-ecdf",
    response_class=Response,
    summary="Get service time ECDF plot",
    description="Generate ECDF comparison plot as PNG image",
)
async def get_service_time_ecdf_image(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Get service time ECDF plot as PNG image.

    Returns the empirical CDF plot as a binary PNG image.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the simulation report.
        authorize: JWT authentication dependency.
        session: Database session dependency.

    Returns:
        PNG image (binary response).
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        report = await get_simulation_configuration_report(
            session=session,
            simulation_configuration_id=simulation_configuration_id,
            report_id=report_id,
            user_id=user_id,
            status=ReportStatus.COMPLETED,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation report {report_id} not found or not completed yet.",
        )

    try:
        service_times = extract_service_times_from_report(report)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    distribution = await get_service_distribution_from_report(report)

    fig = plot_service_time_ecdf(
        service_times=service_times,
        distribution=distribution,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": f"inline; filename=ecdf_{report_id}.png"
        },
    )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/service-time-qq-plot",
    response_class=Response,
    summary="Get service time Q-Q plot",
    description="Generate Q-Q plot as PNG image",
)
async def get_service_time_qq_plot_image(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """Get service time Q-Q plot as PNG image.

    Returns the quantile-quantile plot as a binary PNG image.
    Requires that a theoretical distribution is available.

    Args:
        simulation_configuration_id: UUID of the parent configuration.
        report_id: UUID of the simulation report.
        authorize: JWT authentication dependency.
        session: Database session dependency.

    Returns:
        PNG image (binary response).

    Raises:
        HTTPException 400: If theoretical distribution not available.
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        report = await get_simulation_configuration_report(
            session=session,
            simulation_configuration_id=simulation_configuration_id,
            report_id=report_id,
            user_id=user_id,
            status=ReportStatus.COMPLETED,
        )
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=404,
            detail=f"Simulation report {report_id} not found or not completed yet.",
        )

    try:
        service_times = extract_service_times_from_report(report)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    distribution = await get_service_distribution_from_report(report)

    if distribution is None:
        raise HTTPException(
            status_code=400,
            detail="Q-Q plot requires a theoretical distribution",
        )

    fig = plot_qq_plot(
        service_times=service_times,
        distribution=distribution,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "Content-Disposition": f"inline; filename=qq_plot_{report_id}.png"
        },
    )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/gantt",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    summary="Generate Gantt chart visualization",
)
async def get_gantt_chart(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
    width: int = Query(12, ge=4, le=20, description="Figure width in inches"),
    height: int = Query(8, ge=4, le=16, description="Figure height in inches"),
):
    """Generate Gantt chart from simulation report.

    Returns an image showing channel occupancy over time. Requires that
    the simulation was run with collect_gantt_data=true.
    """
    """
    Generate and return a Gantt chart for a completed simulation report.
    """
    authorize.jwt_required()
    try:
        report = await get_simulation_configuration_report(
            session,
            user_id=uuid.UUID(authorize.get_jwt_subject()),
            report_id=report_id,
            simulation_configuration_id=simulation_configuration_id,
        )
        configuration = await report.awaitable_attrs.configuration
    except SimulationReportNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Simulation report not found.",
        )

    if report.status != ReportStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Report is not complete. Current status: {report.status}",
        )

    if not report.results or not report.results.get("replications"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No results found in report.",
        )

    first_replication = report.results["replications"][0]
    gantt_data = first_replication.get("gantt_chart")

    if not gantt_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No Gantt chart data found. Ensure simulation was run with 'collectGanttData=true'.",
        )

    try:
        gantt_items = [GanttChartItem(**item) for item in gantt_data]
        num_channels = configuration.simulation_parameters.get("numChannels", 1)

        image_buffer = SimulationVisualizer.plot_gantt_chart(
            gantt_items=gantt_items,
            num_channels=num_channels,
            width=width,
            height=height,
        )
    except Exception as e:
        logger.exception(
            "Failed to generate Gantt chart for report %s", report_id
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate visualization: {e}",
        )

    return Response(content=image_buffer.getvalue(), media_type="image/png")


@router.post(
    "/{simulation_configuration_id}/reports/{report_id}/animation",
    response_model=CreateAnimationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Request animation generation",
)
async def request_animation_generation(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    authorize: AuthJWT = Depends(),
    fps: int = Query(20, ge=5, le=60),
    duration: int = Query(15, ge=5, le=60),
    session: AsyncSession = Depends(get_session),
):
    """
    Dispatch a background task to generate an MP4 animation of the simulation.
    This endpoint returns a task ID immediately. Use the task status endpoint
    to check for completion, then use the GET endpoint to download the video.
    """
    authorize.jwt_required()
    user_id = authorize.get_jwt_subject()

    try:
        task = generate_animation_task.delay(
            report_id=str(report_id),
            configuration_id=str(simulation_configuration_id),
            user_id=user_id,
            fps=fps,
            duration=duration,
        )
        background_task = await create_background_task(
            session,
            task_id=task.id,
            task_type=TaskType.ANIMATION,
            user_id=uuid.UUID(user_id),
            subject_id=report_id,
        )
        return CreateAnimationResponse(task_id=background_task.id)
    except Exception as e:
        logger.exception(
            "Failed to dispatch animation task for report %s", report_id
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to dispatch animation task: {e}",
        )


@router.get(
    "/{simulation_configuration_id}/reports/{report_id}/animations/{background_task_id}",
    response_class=FileResponse,
    summary="Download generated animation",
    responses={
        200: {"content": {"video/mp4": {}}},
        404: {"description": "Task not found or animation not ready"},
    },
)
async def get_animation(
    simulation_configuration_id: uuid.UUID,
    report_id: uuid.UUID,
    background_task_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    authorize: AuthJWT = Depends(),
    session: AsyncSession = Depends(get_session),
):
    """
    Download a generated animation video file.
    Poll the task status endpoint first. Once the state is 'SUCCESS',
    use this endpoint with the task ID to retrieve the MP4 file.
    """
    authorize.jwt_required()
    user_id = uuid.UUID(authorize.get_jwt_subject())

    try:
        background_task = await get_background_task(
            session, background_task_id=background_task_id, user_id=user_id
        )
    except BackgroundTaskNotFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Background task not found.",
        )

    result = AsyncResult(background_task.task_id, app=celery_app)

    if not result.ready():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animation is still being generated or task ID is invalid.",
        )
    if result.state != "SUCCESS":
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task failed and could not generate animation. Reason: {result.info}",
        )

    filepath = result.result.get("filepath")
    if not filepath or not Path(filepath).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Animation file not found. It may have expired or failed to save.",
        )

    background_tasks.add_task(
        lambda p: Path(p).unlink(missing_ok=True), filepath
    )

    return FileResponse(
        path=filepath,
        media_type="video/mp4",
        filename=f"simulation_{background_task_id}.mp4",
    )
