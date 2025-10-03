"""Rate limiting for simulation task submissions.

This module provides Redis-based rate limiting to prevent abuse and
ensure fair resource allocation across users.

Architecture:
    - Per-user counters stored in Redis
    - Sliding window algorithm
    - Automatic expiration
    - Configurable limits per environment
"""

from fastapi import HTTPException, status
from redis import Redis, RedisError
import logging

from src.simulations.config import get_worker_settings


logger = logging.getLogger(__name__)
redis_client = Redis.from_url(
    get_worker_settings().redis_url,
    decode_responses=True,
    socket_connect_timeout=5,
    socket_keepalive=True,
    health_check_interval=30,
)


def check_rate_limit(
    user_id: str,
    max_per_hour: int = 10,
    window_seconds: int = 3600,
    fail_open: bool = False,
) -> None:
    """Check if user has exceeded simulation submission rate limit.

    Use Redis-based sliding window rate limiting to prevent abuse.
    If the limit is exceeded, raise an HTTP 429 exception.

    Args:
        user_id: User identifier.
        max_per_hour: Maximum submissions allowed per hour.
        window_seconds: Time window in seconds (default: 3600 = 1 hour).
        fail_open: If True, allow requests when Redis is unavailable.

    Raises:
        HTTPException: If rate limit exceeded (429 status code).
        HTTPException: If Redis unavailable and fail_open=False (503 status code).

    Note:
        In production:
        - Set fail_open=True to prevent Redis outages from blocking all traffic
        - Monitor failed rate limit checks in metrics
        - Consider setting up Redis Sentinel for HA

    Example:
        >>> try:
        ...     check_rate_limit(user_id="user123", max_per_hour=10, fail_open=True)
        ... except HTTPException as e:
        ...     print(f"Rate limited: {e.detail}")
    """
    key = f"rate_limit:simulations:{user_id}"

    try:
        current = redis_client.incr(key)

        if current == 1:
            redis_client.expire(key, window_seconds)
            logger.debug(
                "Rate limit initialized for user %s: 1/%d",
                user_id,
                max_per_hour,
            )

        if current > max_per_hour:
            remaining_ttl = redis_client.ttl(key)
            logger.warning(
                "Rate limit exceeded for user %s: %d/%d (resets in %ds)",
                user_id,
                current,
                max_per_hour,
                remaining_ttl,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {max_per_hour} simulations per hour. "
                f"Resets in {remaining_ttl} seconds.",
                headers={"Retry-After": str(remaining_ttl)},
            )

        logger.debug(
            "Rate limit check passed for user %s: %d/%d",
            user_id,
            current,
            max_per_hour,
        )

    except RedisError as e:
        logger.error(
            "Redis error during rate limit check for user %s: %s",
            user_id,
            e,
            exc_info=True,
        )

        if fail_open:
            logger.warning(
                "Rate limiting unavailable, allowing request (fail-open mode): user=%s",
                user_id,
            )
            return
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Rate limiting temporarily unavailable",
            )
