import io
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

from src.simulations.core.schemas import GanttChartItem


logger = logging.getLogger(__name__)


class SimulationVisualizer:
    """Handle generation of simulation visualizations."""

    @staticmethod
    def plot_gantt_chart(
        gantt_items: List[GanttChartItem],
        num_channels: int,
        width: int = 12,
        height: int = 8,
    ) -> io.BytesIO:
        """
        Generate a Gantt chart plot and return it as an in-memory byte buffer.

        Args:
            gantt_items: List of Gantt chart data points.
            num_channels: Total number of channels to display on the y-axis.
            width: Figure width in inches.
            height: Figure height in inches.

        Returns:
            An io.BytesIO buffer containing the PNG image data.
        """
        logger.debug(
            "Generating Gantt chart with %d items for %d channels.",
            len(gantt_items),
            num_channels,
        )
        fig, ax = plt.subplots(figsize=(width, height))

        colors = plt.cm.viridis([i / num_channels for i in range(num_channels)])

        for item in gantt_items:
            if item.channel >= num_channels:
                continue
            ax.barh(
                y=item.channel,
                width=item.duration,
                left=item.start,
                height=0.6,
                color=colors[item.channel % len(colors)],
                edgecolor="black",
                linewidth=0.5,
                alpha=0.75,
            )

        ax.set_xlabel("Simulation Time")
        ax.set_ylabel("Channel ID")
        ax.set_yticks(range(num_channels))
        ax.set_yticklabels([f"Channel {i}" for i in range(num_channels)])
        ax.set_title("Channel Occupancy Gantt Chart")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)

        ax.invert_yaxis()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        logger.info("Gantt chart generated successfully.")
        return buf

    @staticmethod
    def animate_gantt_chart(
        gantt_items: List[GanttChartItem],
        num_channels: int,
        total_time: float,
        filepath: str,
        fps: int = 20,
        duration_sec: int = 15,
    ) -> None:
        """
        Generate an animated Gantt chart and saves it to a file.
        """
        logger.info(
            "Starting Gantt chart animation generation: duration=%ds, fps=%d",
            duration_sec,
            fps,
        )
        fig, ax = plt.subplots(figsize=(10, max(4, num_channels * 0.4)))
        colors = plt.cm.viridis([i / num_channels for i in range(num_channels)])
        sorted_items = sorted(gantt_items, key=lambda item: item.start)

        def update(frame_time: float):
            ax.clear()
            for item in sorted_items:
                if item.start > frame_time:
                    break
                current_width = min(item.duration, frame_time - item.start)
                ax.barh(
                    y=item.channel,
                    width=current_width,
                    left=item.start,
                    height=0.6,
                    color=colors[item.channel % len(colors)],
                    edgecolor="black",
                    linewidth=0.5,
                    alpha=0.8,
                )
            ax.set_xlim(0, total_time)
            ax.set_ylim(-0.5, num_channels - 0.5)
            ax.set_xlabel("Simulation Time")
            ax.set_ylabel("Channel ID")
            ax.set_title(f"Simulation Progress (Time: {frame_time:.2f}s)")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            ax.invert_yaxis()

        num_frames = duration_sec * fps
        frames = np.linspace(0, total_time, num_frames)
        anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)

        writer = FFMpegWriter(fps=fps, metadata=dict(artist="MMCC Simulator"))

        anim.save(filepath, writer=writer)

        plt.close(fig)
        logger.info("Animation generation complete. Saved to %s", filepath)
