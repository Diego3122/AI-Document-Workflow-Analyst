from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import require_access
from app.models.responses import AuthStatusResponse
from app.services.security_service import AuthPrincipal


router = APIRouter(prefix='/auth', tags=['auth'])


@router.get('/me', response_model=AuthStatusResponse)
def get_current_identity(
    principal: AuthPrincipal | None = Depends(require_access()),
) -> AuthStatusResponse:
    if principal is None:
        raise RuntimeError('Authenticated principal was not resolved.')
    return AuthStatusResponse(
        subject_id=principal.subject_id,
        tenant_id=principal.tenant_id,
        auth_type=principal.auth_type,
        role=principal.role,
        roles=list(principal.roles),
        scopes=list(principal.scopes),
        display_name=principal.display_name,
        can_upload_and_query=principal.can_upload_and_query(),
        can_review=principal.can_review(),
        can_view_metrics=principal.can_view_metrics(),
    )
