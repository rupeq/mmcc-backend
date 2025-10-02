import datetime
import uuid

from pydantic import BaseModel, ConfigDict


class SimulationConfigurationInfo(BaseModel):
    id: uuid.UUID
    name: str | None = None
    description: str | None = None
    created_at: datetime.datetime | None = None
    updated_at: datetime.datetime | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )


class GetSimulationsResponse(BaseModel):
    items: list[SimulationConfigurationInfo]
    total_items: int
    total_pages: int | None = None
    page: int | None = None
    limit: int | None = None
