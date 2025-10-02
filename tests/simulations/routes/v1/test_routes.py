import uuid
import pytest
from another_fastapi_jwt_auth.exceptions import AuthJWTException, JWTDecodeError
from fastapi import status
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

from another_fastapi_jwt_auth import AuthJWT
from src.main import app
from src.core.db_session import get_session

# Импортируем кастомные исключения
from src.simulations.routes.v1.exceptions import (
    BadFilterFormat,
    InvalidColumn,
    InvalidReportStatus,
)

TEST_USER_ID = uuid.uuid4()
BASE_URL = "/api/v1/simulations"


# --- ФИКСУРЫ ДЛЯ ТЕСТОВ ---


@pytest.fixture
def client():
    """Предоставляет TestClient для приложения."""
    return TestClient(app)


@pytest.fixture(autouse=True)
def setup_mocks(mocker):
    """
    Централизованная фикстура для мокирования зависимостей.
    Применяется автоматически ко всем тестам.
    """
    # Мокаем зависимость AuthJWT
    mock_authorize = MagicMock()
    mock_authorize.get_jwt_subject.return_value = str(TEST_USER_ID)
    mocker.patch.object(AuthJWT, "jwt_required", return_value=None)
    app.dependency_overrides[AuthJWT] = lambda: mock_authorize

    # Мокаем сессию БД
    mock_session = AsyncMock()
    app.dependency_overrides[get_session] = lambda: mock_session

    yield {"authorize": mock_authorize, "session": mock_session}

    # Очистка после каждого теста
    app.dependency_overrides = {}


@pytest.fixture
def mock_simulation_configs():
    """Предоставляет список мок-объектов конфигураций симуляций."""

    class MockSimConfig:
        def __init__(self, id, name, description):
            self.id = id
            self.name = name
            self.description = description
            self.created_at = "2025-10-01T12:00:00Z"
            self.updated_at = "2025-10-01T12:00:00Z"

    return [
        MockSimConfig(uuid.uuid4(), "Test Sim 1", "Description for sim 1"),
        MockSimConfig(uuid.uuid4(), "Another Sim 2", "Details about sim 2"),
    ]


# --- ТЕСТОВЫЕ СЦЕНАРИИ ---


def test_get_simulations_unauthorized(client):
    """Тест: эндпоинт должен требовать аутентификацию."""
    app.dependency_overrides = {}  # Убираем моки для этого теста
    with patch.object(
        AuthJWT,
        "jwt_required",
        side_effect=JWTDecodeError(
            status.HTTP_401_UNAUTHORIZED, "Missing cookie"
        ),
    ):
        response = client.get(BASE_URL)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert "Missing cookie" in response.json()["detail"]


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_success_basic(
    mock_get_configs, client, mock_simulation_configs
):
    """Тест: базовый успешный запрос с параметрами по умолчанию."""
    mock_get_configs.return_value = (
        mock_simulation_configs,
        len(mock_simulation_configs),
    )
    response = client.get(BASE_URL)

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    mock_get_configs.assert_called_once()
    assert data["total_items"] == 2
    assert len(data["items"]) == 2
    assert data["items"][0]["name"] == "Test Sim 1"


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_with_pagination(
    mock_get_configs, client, mock_simulation_configs
):
    """Тест: эндпоинт с параметрами пагинации."""
    mock_get_configs.return_value = ([mock_simulation_configs[0]], 10)
    response = client.get(f"{BASE_URL}?page=2&limit=5")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    call_args = mock_get_configs.call_args[1]
    assert call_args["page"] == 2
    assert call_args["limit"] == 5
    assert data["total_pages"] == 2


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_with_column_selection(
    mock_get_configs, client, mock_simulation_configs
):
    """Тест: выборка определенного набора колонок."""
    mock_get_configs.return_value = (mock_simulation_configs, 2)
    response = client.get(f"{BASE_URL}?columns=name,description")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    item = data["items"][0]
    assert "name" in item
    assert "description" in item
    assert "created_at" not in item


@patch(
    "src.simulations.routes.v1.routes.validate_simulation_columns",
    side_effect=InvalidColumn,
)
def test_get_simulations_with_invalid_column(mock_validator, client):
    """Тест: запрос с несуществующей колонкой."""
    response = client.get(f"{BASE_URL}?columns=non_existent_field")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Invalid columns" in response.json()["detail"]


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_with_filters(
    mock_get_configs, client, mock_simulation_configs
):
    """Тест: фильтрация по имени и статусу отчета."""
    mock_get_configs.return_value = ([mock_simulation_configs[0]], 1)
    response = client.get(
        f"{BASE_URL}?filters=name:Test,report_status:completed"
    )
    assert response.status_code == status.HTTP_200_OK
    call_args = mock_get_configs.call_args[1]
    assert call_args["filters"] == {
        "name": "Test",
        "report_status": "completed",
    }


@pytest.mark.parametrize(
    "invalid_filter, mock_path, exception, detail",
    [
        (
            "name-Test",
            "src.simulations.routes.v1.routes.parse_search_query",
            BadFilterFormat,
            "Invalid search format.",
        ),
        (
            "report_status:wrong",
            "src.simulations.routes.v1.routes.verify_report_status_value",
            InvalidReportStatus,
            "Invalid report_status.",
        ),
        (
            "name:Test:Value",
            "src.simulations.routes.v1.routes.parse_search_query",
            BadFilterFormat,
            "Invalid search format.",
        ),
    ],
)
def test_get_simulations_with_invalid_filters(
    client, mocker, invalid_filter, mock_path, exception, detail
):
    """Тест: различные невалидные форматы фильтров."""
    mocker.patch(mock_path, side_effect=exception)
    response = client.get(f"{BASE_URL}?filters={invalid_filter}")
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert detail in response.json()["detail"]


@patch("src.simulations.routes.v1.routes.get_simulation_configurations")
def test_get_simulations_no_results(mock_get_configs, client):
    """Тест: ответ, когда ни одна конфигурация не подходит под фильтры."""
    mock_get_configs.return_value = ([], 0)
    response = client.get(f"{BASE_URL}?filters=name:DoesNotExist&limit=10")

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_items"] == 0
    assert data["items"] == []
    # Исправленное утверждение
    assert data["total_pages"] == 0
