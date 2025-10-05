import logging
import uuid
from pathlib import Path

from celery import Task

from src.simulations.worker import celery_app, worker_settings
from src.simulations.tasks.db_session import get_worker_session
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
    on_failure=lambda self, exc, task_id, args, kwargs, _: (
        celery_app.send_task(
            "simulations.dlq",
            args=[task_id, args, kwargs, str(exc)],
            queue="simulations_dlq",
        )
    ),
    soft_time_limit=None,
    time_limit=None,
)
def generate_animation_task(
    self: Task, report_id: str, configuration_id: str, user_id: str, **kwargs
):
    """
    Celery task to generate and save a simulation animation.

    Args:
        self: Task.
        report_id: The ID of the simulation report.
        configuration_id: The ID of the parent configuration.
        user_id: The ID of the user requesting the animation.
        **kwargs: Additional parameters like fps and duration.

    Returns:
        A dictionary containing the path to the saved video file.
    """
    logger.info("Starting animation task for report %s", report_id)
    self.update_state(state="STARTED", meta={"status": "Fetching data..."})

    async def _run():
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

            output_path = CACHE_DIR / f"{self.request.id}.mp4"

            self.update_state(
                state="PROGRESS", meta={"status": "Generating video..."}
            )
            SimulationVisualizer.animate_gantt_chart(
                gantt_items=gantt_items,
                num_channels=num_channels,
                total_time=total_time,
                filepath=str(output_path),
                fps=kwargs.get("fps", 20),
                duration_sec=kwargs.get("duration", 15),
            )

            logger.info("Animation saved to %s", output_path)
            self.update_state(state="SUCCESS")
            return {"filepath": str(output_path)}

    try:
        import asyncio

        return asyncio.run(_run())
    except Exception:
        logger.exception("Animation task failed for report %s", report_id)
        self.update_state(state="FAILURE")
        raise
