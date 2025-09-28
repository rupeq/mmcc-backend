from pydantic import BaseModel, EmailStr


class SignInRequestSchema(BaseModel):
    email: EmailStr
    password: str


class SignUpRequestSchema(BaseModel):
    email: EmailStr
    password: str


class SignUpResponseSchema(BaseModel):
    email: EmailStr
