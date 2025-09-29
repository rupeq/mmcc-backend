from unittest.mock import MagicMock
from starlette.responses import JSONResponse

from src.core.exception_handlers import authjwt_exception_handler


def test_authjwt_exception_handler():
    """
    Test that the handler converts an AuthJWTException-like object
    into a proper JSONResponse.
    """
    mock_request = MagicMock()
    test_status_code = 401
    test_message = "Test Unauthorized"

    mock_exception = MagicMock()
    mock_exception.status_code = test_status_code
    mock_exception.message = test_message

    response = authjwt_exception_handler(mock_request, mock_exception)

    assert isinstance(response, JSONResponse)
    assert response.status_code == test_status_code
    assert response.body == b'{"detail":"Test Unauthorized"}'
