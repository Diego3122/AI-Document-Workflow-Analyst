from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.security_service import AuthPrincipal


_CURRENT_AUTH_PRINCIPAL: ContextVar['AuthPrincipal | None'] = ContextVar('current_auth_principal', default=None)


def get_current_principal() -> 'AuthPrincipal | None':
    return _CURRENT_AUTH_PRINCIPAL.get()


def push_current_principal(principal: 'AuthPrincipal | None') -> 'AuthPrincipal | None':
    previous = _CURRENT_AUTH_PRINCIPAL.get()
    _CURRENT_AUTH_PRINCIPAL.set(principal)
    return previous


def pop_current_principal(previous: 'AuthPrincipal | None') -> None:
    _CURRENT_AUTH_PRINCIPAL.set(previous)
