"""Animation generation task with standardized state management."""

import logging
import uuid
from pathlib import Path

from celery import Task

from src.simulations.worker import celery_app, worker_settings
from src.simulations.tasks.db_session import get_worker_session
from src.simulations.tasks.state_manager import create_task_manager
from src.simulations.db_utils.simulation_reports import (
    get_simulation_configuration_report,
)
from src.simulations.core.schemas import GanttChartItem
from src.simulations.core.visualization import SimulationVisualizer


logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/simulation_cache/animations")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@celery_app.task(
    name="simulations.generate_animation",
    bind=True,
    max_retries=worker_settings.task_max_retries,
    default_retry_delay=worker_settings.task_default_retry_delay,
    acks_late=True,
    reject_on_worker_lost=True,
    track_started=True,
    soft_time_limit=None,
    time_limit=None,
)
def generate_animation_task(
    self: Task,
    report_id: str,
    configuration_id: str,
    user_id: str,
    **kwargs,
):
    """Generate and save a simulation animation.

    Args:
        self: Task instance.
        report_id: Simulation report ID.
        configuration_id: Parent configuration ID.
        user_id: User ID requesting the animation.
        **kwargs: Additional parameters like fps and duration.

    Returns:
        Dictionary containing the path to the saved video file.
    """
    state_manager = create_task_manager(self, "animation")
    state_manager.report_started("Fetching simulation data...")

    logger.info("Starting animation task for report %s", report_id)

    async def _run():
        state_manager.report_progress(
            current=1,
            total=3,
            message="Loading simulation results...",
        )

        async with get_worker_session() as session:
            report = await get_simulation_configuration_report(
                session,
                report_id=uuid.UUID(report_id),
                simulation_configuration_id=uuid.UUID(configuration_id),
                user_id=uuid.UUID(user_id),
            )

            gantt_data = report.results.get("replications", [{}])[0].get(
                "gantt_chart"
            )

            if not gantt_data:
                raise ValueError("Report contains no Gantt chart data.")

            gantt_items = [GanttChartItem(**item) for item in gantt_data]
            configuration = await report.awaitable_attrs.configuration
            sim_params = configuration.simulation_parameters
            num_channels = sim_params.get("numChannels", 1)
            total_time = sim_params.get("simulationTime", 100)

        state_manager.report_progress(
            current=2,
            total=3,
            message="Generating video frames...",
        )

        output_path = CACHE_DIR / f"{self.request.id}.mp4"

        SimulationVisualizer.animate_gantt_chart(
            gantt_items=gantt_items,
            num_channels=num_channels,
            total_time=total_time,
            filepath=str(output_path),
            fps=kwargs.get("fps", 20),
            duration_sec=kwargs.get("duration", 15),
        )

        logger.info("Animation saved to %s", output_path)

        state_manager.report_progress(
            current=3,
            total=3,
            message="Animation complete",
        )

        result = {"filepath": str(output_path)}
        return state_manager.report_success(
            result=result,
            summary=f"Animation generated: {output_path.name}",
        )

    try:
        import asyncio

        return asyncio.run(_run())
    except Exception as e:
        logger.exception("Animation task failed for report %s", report_id)
        state_manager.report_failure(e)
        raise
