from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from another_fastapi_jwt_auth.exceptions import AuthJWTException

from src.config import get_settings
from src.core.exception_handlers import authjwt_exception_handler
from src.authorization.routes.v1.routes import router as authorization_router
from src.core.lifespan import lifespan
from src.users.routes.v1.routes import router as users_router

app = FastAPI(
    debug=get_settings().service.debug,
    title=get_settings().service.app_name,
    docs_url="/docs" if get_settings().service.debug else None,
    redoc_url="/redoc" if get_settings().service.debug else None,
    openapi_url="/openapi" if get_settings().service.debug else None,
    lifespan=lifespan,
)


if cors_origins := get_settings().service.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(cors_origin) for cors_origin in cors_origins],
        allow_credentials=get_settings().service.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.exception_handler(AuthJWTException)(authjwt_exception_handler)
app.include_router(
    authorization_router, prefix=get_settings().service.api_prefix
)
app.include_router(users_router, prefix=get_settings().service.api_prefix)
