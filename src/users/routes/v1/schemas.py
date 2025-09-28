import datetime
import uuid

from pydantic import BaseModel


class GetMeResponse(BaseModel):
    """
    Schema for retrieving current user details.

    Attributes:
        id (uuid.UUID): The unique identifier of the user.
        email (str): The email address of the user.
        is_active (bool): A flag indicating if the user account is active.
        created_at (datetime.datetime): The timestamp when the user account was created.
        updated_at (datetime.datetime): The timestamp when the user account was last updated.
    """

    id: uuid.UUID
    email: str
    is_active: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime
