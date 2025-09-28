from another_fastapi_jwt_auth.exceptions import AuthJWTException
from starlette.requests import Request
from starlette.responses import JSONResponse


def authjwt_exception_handler(_: Request, exc: AuthJWTException):
    """
    Handle AuthJWTExceptions by returning a JSON response with the error detail.

    Args:
        _ (Request): The incoming request (unused).
        exc (AuthJWTException): The AuthJWTException instance.

    Returns:
        JSONResponse: A JSON response containing the status code and error message from the exception.
    """
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.message}
    )
