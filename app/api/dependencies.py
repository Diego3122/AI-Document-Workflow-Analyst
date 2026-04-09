from __future__ import annotations

from collections.abc import AsyncGenerator

from fastapi import Request, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.services.auth_context import pop_current_principal, push_current_principal
from app.services.security_service import AuthPrincipal, SecurityService


api_key_header = APIKeyHeader(name='X-API-Key', auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


def require_access(*allowed_roles: str, required_scopes: tuple[str, ...] = ()):  # noqa: B008
    async def dependency(
        request: Request,
        api_key: str | None = Security(api_key_header),
        bearer_credentials: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
    ) -> AsyncGenerator[AuthPrincipal | None, None]:
        security_service = SecurityService()
        security_service.enforce_ip_rate_limit(request)
        bearer_token = bearer_credentials.credentials if bearer_credentials is not None else None
        principal = security_service.authenticate(api_key=api_key, bearer_token=bearer_token)
        security_service.enforce_user_rate_limit(request, principal)
        security_service.authorize(principal, allowed_roles, required_scopes=required_scopes)
        request.state.auth_principal = principal
        previous = push_current_principal(principal)
        try:
            yield principal
        finally:
            pop_current_principal(previous)

    return dependency
