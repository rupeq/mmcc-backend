"""Celery worker monitoring script.

This script provides real-time monitoring of Celery workers, displaying
active workers, task queues, and execution statistics. It's useful for
debugging and operational monitoring of the simulation task processing system.

Usage:
    python scripts/monitor_worker.py
"""

from celery import Celery

from src.simulations.config import get_worker_settings


settings = get_worker_settings()
app = Celery(broker=settings.celery_broker_url)


def print_section(title: str) -> None:
    """Print a formatted section header.

    Args:
        title: Section title to display.
    """
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    """Display comprehensive Celery worker status and statistics."""
    inspect = app.control.inspect()

    print_section("Active Workers")
    stats = inspect.stats()
    if stats:
        for worker, info in stats.items():
            print(f"\n📊 {worker}")
            print(
                f"   Pool: {info.get('pool', {}).get('implementation', 'unknown')}"
            )
            print(
                f"   Concurrency: {info.get('pool', {}).get('max-concurrency', 'unknown')}"
            )
            print(f"   Total Tasks: {info.get('total', {})}")
    else:
        print("❌ No active workers found!")
        print(
            "   Start workers with: celery -A src.simulations.worker.celery_app worker"
        )

    print_section("Active Tasks")
    active = inspect.active()
    if active:
        total_active = sum(len(tasks) for tasks in active.values())
        print(f"📈 Total active tasks: {total_active}\n")

        for worker, tasks in active.items():
            print(f"\n🔄 {worker}: {len(tasks)} active")
            for task in tasks[:5]:  # Show first 5
                print(f"   • {task['name']}")
                print(f"     ID: {task['id'][:12]}...")
                print(f"     Started: {task.get('time_start', 'N/A')}")
    else:
        print("✅ No active tasks")

    print_section("Reserved Tasks")
    reserved = inspect.reserved()
    if reserved:
        total_reserved = sum(len(tasks) for tasks in reserved.values())
        print(f"📥 Total reserved tasks: {total_reserved}")
        for worker, tasks in reserved.items():
            print(f"   {worker}: {len(tasks)} reserved")
    else:
        print("✅ No reserved tasks")

    print_section("Scheduled Tasks")
    scheduled = inspect.scheduled()
    if scheduled:
        total_scheduled = sum(len(tasks) for tasks in scheduled.values())
        print(f"⏰ Total scheduled tasks: {total_scheduled}")
        for worker, tasks in scheduled.items():
            print(f"   {worker}: {len(tasks)} scheduled")
    else:
        print("✅ No scheduled tasks")

    print_section("Registered Tasks")
    registered = inspect.registered()
    if registered:
        for worker, tasks in registered.items():
            print(f"\n📋 {worker}:")
            for task in sorted(tasks):
                if "simulations" in task:
                    print(f"   • {task}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
