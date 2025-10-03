"""Configuration module for Celery worker settings.

This module provides comprehensive configuration for Celery workers,
including broker settings, task timeouts, retry behavior, and production
optimizations. All settings are validated and documented.
"""

import os
import logging
from functools import lru_cache
from typing import Literal
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class WorkerSettings(BaseSettings):
    """Configure Celery worker behavior and resource limits.

    This class defines all Celery worker configuration using Pydantic
    for validation and type safety. Settings can be overridden via
    environment variables or a .worker.env file.

    Environment-Specific Defaults:
        Development:
            - Lower concurrency (2-4)
            - Shorter timeouts
            - Detailed logging
            - No compression

        Production:
            - Higher concurrency (8-16)
            - Longer timeouts
            - JSON logging
            - Gzip compression
            - Memory limits enabled

    Attributes:
        redis_url: Redis connection URL for general purposes.
        celery_broker_url: Redis URL for Celery message broker.
        celery_result_backend: Redis URL for storing task results.
        worker_concurrency: Number of concurrent worker processes (1-32).
        worker_prefetch_multiplier: Tasks to prefetch per worker (1-10).
        worker_max_tasks_per_child: Max tasks before worker restart (1-1000).
        task_acks_late: Acknowledge tasks after completion vs on receipt.
        task_reject_on_worker_lost: Requeue tasks if worker dies.
        task_time_limit: Hard timeout for task execution in seconds (60-7200).
        task_soft_time_limit: Soft timeout that raises exception (60-7200).
        task_max_retries: Maximum retry attempts (0-10).
        task_default_retry_delay: Seconds between retries (10-3600).
        result_expires: Seconds before results expire (3600-604800).
        worker_pool_type: Worker pool implementation type.
        task_track_started: Enable task start tracking for monitoring.
        worker_send_task_events: Enable event messages for monitoring.
        task_send_sent_event: Send event when task dispatched to broker.
        broker_connection_retry: Enable automatic broker reconnection.
        broker_connection_retry_on_startup: Retry connection on startup.
        broker_connection_max_retries: Max broker connection retries (0=infinite).
        worker_max_memory_per_child: Max memory before worker restart (KB).
        task_compression: Compression algorithm for task messages.
        result_compression: Compression algorithm for results.
        worker_enable_remote_control: Enable remote control commands.
        environment: Runtime environment (development/production).

    Example:
        ```python
        # Development
        settings = WorkerSettings(
            worker_concurrency=2,
            environment="development"
        )

        # Production
        settings = WorkerSettings(
            worker_concurrency=8,
            task_compression="gzip",
            environment="production"
        )
        ```
    """

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), ".worker.env"),
        case_sensitive=False,
        validate_default=True,
    )

    # Redis Configuration
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL for general use",
    )
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for Celery message broker",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL for task result storage",
    )

    # Worker Process Configuration
    worker_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of concurrent worker processes",
    )
    worker_prefetch_multiplier: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Number of tasks to prefetch per worker",
    )
    worker_max_tasks_per_child: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Max tasks before worker process restart (prevents memory leaks)",
    )

    # Task Acknowledgment Settings
    task_acks_late: bool = Field(
        default=True,
        description="Acknowledge task after completion (not on receipt)",
    )
    task_reject_on_worker_lost: bool = Field(
        default=True,
        description="Requeue task if worker dies unexpectedly",
    )

    # Task Timeout Configuration
    task_time_limit: int = Field(
        default=60 * 60,  # 1 hour
        ge=60,
        le=7200,
        description="Hard time limit for task execution (seconds)",
    )
    task_soft_time_limit: int = Field(
        default=60 * 55,  # 55 minutes
        ge=60,
        le=7200,
        description="Soft time limit that raises SoftTimeLimitExceeded",
    )

    # Retry Configuration
    task_max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed tasks",
    )
    task_default_retry_delay: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Delay between retry attempts (seconds)",
    )

    # Result Storage Configuration
    result_expires: int = Field(
        default=60 * 60 * 24,  # 24 hours
        ge=3600,
        le=604800,
        description="Time before task results expire (seconds)",
    )

    # Worker Pool Configuration
    worker_pool_type: Literal["prefork", "eventlet", "gevent", "solo"] = Field(
        default="prefork",
        description="Worker pool implementation",
    )

    # Monitoring Configuration
    task_track_started: bool = Field(
        default=True,
        description="Track when tasks are started (for monitoring)",
    )
    worker_send_task_events: bool = Field(
        default=True,
        description="Enable task event messages for monitoring",
    )
    task_send_sent_event: bool = Field(
        default=True,
        description="Send event when task is dispatched to broker",
    )

    # Connection Resilience
    broker_connection_retry: bool = Field(
        default=True,
        description="Enable automatic broker reconnection",
    )
    broker_connection_retry_on_startup: bool = Field(
        default=True,
        description="Retry broker connection on startup",
    )
    broker_connection_max_retries: int = Field(
        default=10,
        ge=0,
        le=100,
        description="Maximum broker connection retry attempts (0 = infinite)",
    )

    # Resource Limits (Optional)
    worker_max_memory_per_child: int | None = Field(
        default=None,
        ge=100000,  # 100MB minimum
        description="Max memory before worker restart (KB, None = no limit)",
    )

    # Compression Configuration (Production)
    task_compression: Literal["gzip", "bzip2", "zlib", None] = Field(
        default=None,
        description="Compression algorithm for task messages",
    )
    result_compression: Literal["gzip", "bzip2", "zlib", None] = Field(
        default=None,
        description="Compression algorithm for results",
    )

    # Security & Control
    worker_enable_remote_control: bool = Field(
        default=True,
        description="Enable remote control commands",
    )

    # Environment
    environment: Literal["development", "production"] = Field(
        default="development",
        description="Runtime environment",
    )

    @field_validator("worker_concurrency")
    @classmethod
    def validate_concurrency(cls, v: int) -> int:
        """Validate worker concurrency is within acceptable range.

        Args:
            v: Proposed concurrency value.

        Returns:
            Validated concurrency value.

        Raises:
            ValueError: If concurrency is outside 1-32 range.

        Warns:
            If concurrency > 16, as this may cause resource contention.
        """
        if v < 1 or v > 32:
            raise ValueError("worker_concurrency must be between 1 and 32")
        if v > 16:
            logger.warning(
                "High worker concurrency (%d) may cause resource contention", v
            )
        return v

    @field_validator("task_time_limit")
    @classmethod
    def validate_time_limit(cls, v: int) -> int:
        """Validate task time limit is reasonable.

        Args:
            v: Proposed time limit in seconds.

        Returns:
            Validated time limit.

        Raises:
            ValueError: If time limit is outside 60-7200 seconds (1min-2hr).
        """
        if v < 60 or v > 7200:
            raise ValueError(
                "task_time_limit must be between 60 and 7200 seconds"
            )
        return v

    @field_validator("task_soft_time_limit")
    @classmethod
    def validate_soft_time_limit(cls, v: int, info) -> int:
        """Validate soft time limit is reasonable and less than hard limit.

        Args:
            v: Proposed soft time limit in seconds.
            info: Validation context containing other field values.

        Returns:
            Validated soft time limit.

        Raises:
            ValueError: If soft limit is outside 60-7200 seconds.
        """
        if v < 60 or v > 7200:
            raise ValueError(
                "task_soft_time_limit must be between 60 and 7200 seconds"
            )

        if "task_time_limit" in info.data:
            hard_limit = info.data["task_time_limit"]
            if v >= hard_limit:
                raise ValueError(
                    f"task_soft_time_limit ({v}s) must be less than "
                    f"task_time_limit ({hard_limit}s)"
                )

        return v

    def get_celery_config(self) -> dict:
        """Generate Celery configuration dictionary.

        Creates a comprehensive Celery configuration dict from settings,
        applying production optimizations when environment is production.

        Returns:
            Dictionary containing all Celery configuration options.

        Example:
            ```python
            settings = get_worker_settings()
            celery_app.conf.update(settings.get_celery_config())
            ```
        """
        config = {
            # Broker & Backend
            "broker_url": self.celery_broker_url,
            "result_backend": self.celery_result_backend,
            # Serialization
            "task_serializer": "json",
            "accept_content": ["json"],
            "result_serializer": "json",
            # Timezone
            "timezone": "UTC",
            "enable_utc": True,
            # Task Execution
            "task_acks_late": self.task_acks_late,
            "task_reject_on_worker_lost": self.task_reject_on_worker_lost,
            "task_time_limit": self.task_time_limit,
            "task_soft_time_limit": self.task_soft_time_limit,
            "task_track_started": self.task_track_started,
            # Worker Configuration
            "worker_prefetch_multiplier": self.worker_prefetch_multiplier,
            "worker_max_tasks_per_child": self.worker_max_tasks_per_child,
            "worker_send_task_events": self.worker_send_task_events,
            # Results
            "result_expires": self.result_expires,
            "result_extended": True,
            # Connection Resilience
            "broker_connection_retry": self.broker_connection_retry,
            "broker_connection_retry_on_startup": self.broker_connection_retry_on_startup,
            "broker_connection_max_retries": self.broker_connection_max_retries,
        }

        # Optional: Memory limit
        if self.worker_max_memory_per_child:
            config["worker_max_memory_per_child"] = (
                self.worker_max_memory_per_child
            )

        # Optional: Compression
        if self.task_compression:
            config["task_compression"] = self.task_compression
        if self.result_compression:
            config["result_compression"] = self.result_compression

        # Production optimizations
        if self.environment == "production":
            config.update(
                {
                    "task_compression": self.task_compression or "gzip",
                    "result_compression": self.result_compression or "gzip",
                    "worker_disable_rate_limits": False,
                    "task_default_rate_limit": "100/m",
                }
            )

        return config


@lru_cache()
def get_worker_settings() -> WorkerSettings:
    """Get cached worker settings instance.

    Returns:
        Singleton instance of WorkerSettings.

    Example:
        ```python
        settings = get_worker_settings()
        print(f"Concurrency: {settings.worker_concurrency}")
        ```
    """
    return WorkerSettings()
