# tests/simulations/worker/test_rate_limiter.py
"""Tests for rate limiting functionality."""
import pytest
from unittest.mock import patch
from fastapi import HTTPException
from redis.exceptions import RedisError

from src.simulations.routes.v1.rate_limiter import check_rate_limit


class TestRateLimitEnforcement:
    """Test rate limit enforcement logic."""

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_allows_requests_under_limit(self, mock_redis):
        """Test that requests under limit are allowed."""
        mock_redis.incr.return_value = 5  # Under limit
        mock_redis.ttl.return_value = 3600

        # Should not raise
        check_rate_limit(user_id="user123", max_per_hour=10)

        mock_redis.incr.assert_called_once_with("rate_limit:simulations:user123")

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_blocks_requests_over_limit(self, mock_redis):
        """Test that requests over limit are blocked."""
        mock_redis.incr.return_value = 11  # Over limit
        mock_redis.ttl.return_value = 1800

        with pytest.raises(HTTPException) as exc_info:
            check_rate_limit(user_id="user123", max_per_hour=10)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail
        assert exc_info.value.headers["Retry-After"] == "1800"

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_initializes_counter_on_first_request(self, mock_redis):
        """Test counter initialization on first request."""
        mock_redis.incr.return_value = 1  # First request
        mock_redis.expire.return_value = True

        check_rate_limit(user_id="user123", max_per_hour=10, window_seconds=3600)

        mock_redis.expire.assert_called_once_with(
            "rate_limit:simulations:user123", 3600
        )


class TestRateLimitKeyFormat:
    """Test rate limit key formatting."""

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_uses_correct_key_format(self, mock_redis):
        """Test that rate limit keys follow correct format."""
        mock_redis.incr.return_value = 1

        check_rate_limit(user_id="test_user_456")

        expected_key = "rate_limit:simulations:test_user_456"
        mock_redis.incr.assert_called_with(expected_key)


class TestRateLimitWindow:
    """Test sliding window behavior."""

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_respects_custom_window(self, mock_redis):
        """Test that custom time windows are respected."""
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True

        custom_window = 1800  # 30 minutes
        check_rate_limit(
            user_id="user123",
            max_per_hour=10,
            window_seconds=custom_window,
        )

        mock_redis.expire.assert_called_with(
            "rate_limit:simulations:user123", custom_window
        )


class TestRedisFailureHandling:
    """Test handling of Redis failures."""

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_fail_open_allows_requests_on_redis_error(self, mock_redis):
        """Test fail-open mode allows requests when Redis is down."""
        mock_redis.incr.side_effect = RedisError("Connection lost")

        # Should not raise with fail_open=True
        check_rate_limit(user_id="user123", fail_open=True)

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_fail_closed_blocks_on_redis_error(self, mock_redis):
        """Test fail-closed mode blocks requests when Redis is down."""
        mock_redis.incr.side_effect = RedisError("Connection lost")

        with pytest.raises(HTTPException) as exc_info:
            check_rate_limit(user_id="user123", fail_open=False)

        assert exc_info.value.status_code == 503
        assert "temporarily unavailable" in exc_info.value.detail


class TestRateLimitLogging:
    """Test rate limit logging."""

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_logs_rate_limit_exceeded(self, mock_redis, caplog):
        """Test that rate limit exceeded events are logged."""
        import logging
        caplog.set_level(logging.WARNING)

        mock_redis.incr.return_value = 11
        mock_redis.ttl.return_value = 1800

        try:
            check_rate_limit(user_id="user123", max_per_hour=10)
        except HTTPException:
            pass

        assert any(
            "Rate limit exceeded" in record.message
            for record in caplog.records
        )

    @patch("src.simulations.routes.v1.rate_limiter.redis_client")
    def test_logs_redis_errors(self, mock_redis, caplog):
        """Test that Redis errors are logged."""
        import logging
        caplog.set_level(logging.ERROR)

        mock_redis.incr.side_effect = RedisError("Connection timeout")

        check_rate_limit(user_id="user123", fail_open=True)

        assert any(
            "Redis error" in record.message for record in caplog.records
        )