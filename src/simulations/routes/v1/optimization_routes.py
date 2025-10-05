import logging

from another_fastapi_jwt_auth import AuthJWT
from fastapi import APIRouter, Depends, HTTPException
from starlette import status

from src.simulations.core.schemas import (
    OptimizationResultResponse,
    OptimizationRequest,
)
from src.simulations.routes.v1.rate_limiter import check_rate_limit
from src.simulations.routes.v1.utils import (
    _optimize_binary_search,
    _optimize_cost_minimization,
    _optimize_gradient_descent,
    _optimize_multi_objective,
)

router = APIRouter(
    tags=["v1", "simulations", "optimization"], prefix="/v1/simulations"
)
logger = logging.getLogger(__name__)


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
