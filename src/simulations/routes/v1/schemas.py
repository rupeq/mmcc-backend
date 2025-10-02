import datetime
import uuid

from pydantic import BaseModel, ConfigDict


class SimulationConfigurationInfo(BaseModel):
    """Represent the information of a simulation configuration.

    Attributes:
        id: The unique identifier of the simulation configuration.
        name: The name of the simulation configuration.
        description: The description of the simulation configuration.
        created_at: The timestamp when the simulation configuration was created.
        updated_at: The timestamp when the simulation configuration was last updated.
    """

    id: uuid.UUID
    name: str | None = None
    description: str | None = None
    created_at: datetime.datetime | None = None
    updated_at: datetime.datetime | None = None

    model_config = ConfigDict(
        from_attributes=True,
    )


class GetSimulationsResponse(BaseModel):
    """Represent the response for getting a list of simulations.

    Attributes:
        items: A list of simulation configurations.
        total_items: The total number of simulation configurations.
        total_pages: The total number of pages available.
        page: The current page number.
        limit: The number of items per page.
    """

    items: list[SimulationConfigurationInfo]
    total_items: int
    total_pages: int | None = None
    page: int | None = None
    limit: int | None = None
