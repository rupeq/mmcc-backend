from pydantic import BaseModel, EmailStr


class SignInRequestSchema(BaseModel):
    """
    Schema for user sign-in requests.

    Attributes:
        email (EmailStr): The user's email address.
        password (str): The user's password.
    """

    email: EmailStr
    password: str


class SignUpRequestSchema(BaseModel):
    """
    Schema for user sign-up requests.

    Attributes:
        email (EmailStr): The user's email address.
        password (str): The user's password.
    """

    email: EmailStr
    password: str


class SignUpResponseSchema(BaseModel):
    """
    Schema for user sign-up responses.

    Attributes:
        email (EmailStr): The email of the newly signed-up user.
    """

    email: EmailStr
