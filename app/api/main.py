from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from starlette.datastructures import Headers, URL
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import RedirectResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from app.api.dependencies import require_access
from app.api.routes_auth import router as auth_router
from app.api.routes_documents import router as documents_router
from app.api.routes_metrics import router as metrics_router
from app.api.routes_query import router as query_router
from app.core.config import get_settings
from app.db.init_db import initialize_database
from app.models.responses import HealthResponse
from app.services.document_ingestion_service import DocumentIngestionService


class LoopbackAwareHTTPSRedirectMiddleware:
    def __init__(self, app: ASGIApp, *, loopback_hosts: list[str]) -> None:
        self.app = app
        self.loopback_hosts = {host.lower() for host in loopback_hosts}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        host = headers.get('host', '').split(':', 1)[0].lower()
        if scope.get('scheme') == 'http' and host not in self.loopback_hosts:
            response = RedirectResponse(str(URL(scope=scope).replace(scheme='https')), status_code=307)
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    initialize_database()
    DocumentIngestionService().reconcile_storage()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    settings.validate_public_ui_settings()
    settings.validate_production_settings()

    app = FastAPI(
        title=settings.app_name,
        version='0.1.0',
        description=(
            'Elevated AI document workflow analyst with a controlled multi-step '
            'workflow, validation, and observability hooks.'
        ),
        lifespan=lifespan,
        docs_url='/docs' if settings.docs_are_enabled() else None,
        redoc_url='/redoc' if settings.docs_are_enabled() else None,
        openapi_url='/openapi.json' if settings.docs_are_enabled() else None,
    )

    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_credentials=False,
            allow_methods=['GET', 'POST', 'OPTIONS'],
            allow_headers=['Authorization', 'Content-Type', 'X-API-Key'],
        )
    allowed_hosts = settings.resolved_trusted_hosts()
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
    if settings.force_https_enabled():
        app.add_middleware(
            LoopbackAwareHTTPSRedirectMiddleware,
            loopback_hosts=settings.internal_api_loopback_hosts(),
        )
    if settings.trust_proxy_headers:
        app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=settings.proxy_trusted_ips)

    @app.middleware('http')
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
        response.headers['X-Robots-Tag'] = 'noindex, nofollow'
        response.headers['Cache-Control'] = 'no-store'
        if settings.force_https_enabled():
            response.headers['Strict-Transport-Security'] = 'max-age=63072000; includeSubDomains'
        return response

    app.include_router(auth_router)
    app.include_router(documents_router)
    app.include_router(query_router)
    app.include_router(metrics_router)

    @app.get('/livez', tags=['health'])
    def livez() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get(
        '/health',
        response_model=HealthResponse,
        tags=['health'],
        dependencies=[Depends(require_access('reader', 'analyst', 'reviewer', 'admin', required_scopes=('health.read',)))],
    )
    def health_check() -> HealthResponse:
        return HealthResponse(status='ok', app_name=settings.app_name, version='0.1.0')

    return app


app = create_app()
