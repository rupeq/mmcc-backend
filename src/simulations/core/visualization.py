import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from src.simulations.core.schemas import GanttChartItem


class SimulationVisualizer:
    @staticmethod
    def plot_gantt_chart(
        gantt_items: list[GanttChartItem],
        num_channels: int,
        filename: str = None,
    ):
        fig, ax = plt.subplots(figsize=(12, max(4, num_channels * 0.5)))

        colors = plt.cm.Set3(range(num_channels))
        for item in gantt_items:
            ax.barh(
                y=item.channel,
                width=item.duration,
                left=item.start,
                height=0.8,
                color=colors[item.channel % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )

        ax.set_xlabel("Time")
        ax.set_ylabel("Channel")
        ax.set_yticks(range(num_channels))
        ax.set_title("Channel Occupancy Gantt Chart")
        ax.grid(True, alpha=0.3)

        if filename:
            plt.savefig(filename, dpi=300, bbox_inches="tight")
        return fig

    @staticmethod
    def animate_simulation(
        gantt_items: list[GanttChartItem],
        num_channels: int,
        total_time: float,
        interval: int = 50,
    ) -> FuncAnimation:
        fig, ax = plt.subplots(figsize=(12, max(4, num_channels * 0.5)))

        def update(frame_time):
            ax.clear()
            visible = [item for item in gantt_items if item.start <= frame_time]

            for item in visible:
                width = min(item.duration, frame_time - item.start)
                ax.barh(
                    y=item.channel,
                    width=width,
                    left=item.start,
                    height=0.8,
                    color="skyblue",
                    edgecolor="black",
                )

            ax.set_xlim(0, total_time)
            ax.set_ylim(-0.5, num_channels - 0.5)
            ax.set_xlabel("Time")
            ax.set_ylabel("Channel")
            ax.set_title(f"Simulation Progress (t={frame_time:.2f})")
            ax.grid(True, alpha=0.3)

        frames = np.linspace(0, total_time, int(total_time * 50 / interval))
        anim = FuncAnimation(fig, update, frames=frames, interval=interval)
        return anim
