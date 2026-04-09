from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import HTTPException, Request, status

from app.core.config import get_settings
from app.core.utils import generate_id, iso_timestamp
from app.db.sqlite import get_connection
from app.services.jwt_validator import (
    JwtConfigurationError,
    JwtMetadataError,
    JwtValidationError,
    JwtValidator,
)


ROLE_READER = 'reader'
ROLE_ANALYST = 'analyst'
ROLE_REVIEWER = 'reviewer'
ROLE_ADMIN = 'admin'
ROLE_AUTHENTICATED = 'authenticated'
ALL_ROLES = {ROLE_READER, ROLE_ANALYST, ROLE_REVIEWER, ROLE_ADMIN}
ROLE_PRIORITY = [ROLE_ADMIN, ROLE_REVIEWER, ROLE_ANALYST, ROLE_READER]


@dataclass(frozen=True)
class AuthPrincipal:
    auth_type: str
    subject_id: str
    role: str
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    tenant_id: str | None = None
    issuer: str | None = None
    token_id: str | None = None
    display_name: str | None = None

    @property
    def is_admin(self) -> bool:
        return self.has_role(ROLE_ADMIN) or self.has_scope('admin')

    def has_role(self, role: str) -> bool:
        normalized = role.strip().lower()
        return normalized == self.role or normalized in self.roles

    def has_scope(self, scope: str) -> bool:
        return scope.strip().lower() in self.scopes

    def has_all_scopes(self, scopes: tuple[str, ...]) -> bool:
        normalized = {scope.strip().lower() for scope in scopes if scope.strip()}
        return not normalized or normalized.issubset(set(self.scopes))

    def subject_rate_limit_key(self) -> str:
        components = [self.auth_type, self.issuer or '', self.tenant_id or '', self.subject_id]
        return hashlib.sha256('|'.join(components).encode('utf-8')).hexdigest()[:24]

    def can_upload_and_query(self) -> bool:
        return self.is_admin or self.has_role(ROLE_ANALYST) or self.has_all_scopes(('documents.read', 'documents.write', 'query.run'))

    def can_review(self) -> bool:
        return self.is_admin or self.has_role(ROLE_REVIEWER) or self.has_scope('query.review')

    def can_view_metrics(self) -> bool:
        return self.is_admin or self.has_scope('metrics.read') or self.has_scope('metrics.write')


class SecurityService:
    """Authentication, authorization, and request throttling for API access."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.jwt_validator = JwtValidator(self.settings)

    def authenticate(self, api_key: str | None, bearer_token: str | None) -> AuthPrincipal | None:
        auth_mode = self.settings.resolved_auth_mode()
        if auth_mode == 'disabled':
            return None

        if auth_mode in {'jwt', 'hybrid'} and bearer_token:
            return self._authenticate_jwt(bearer_token)

        if auth_mode == 'jwt':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='A bearer token is required for this endpoint.',
            )

        if auth_mode in {'api_key', 'hybrid'} and api_key:
            return self._authenticate_api_key(api_key)

        if auth_mode == 'hybrid':
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='A bearer token or API key is required for this endpoint.',
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='An API key is required for this endpoint.',
        )

    def authorize(
        self,
        principal: AuthPrincipal | None,
        allowed_roles: tuple[str, ...],
        required_scopes: tuple[str, ...] = (),
    ) -> None:
        if not self.settings.auth_is_enabled():
            return

        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='Authentication is required for this endpoint.',
            )

        if principal.is_admin:
            return

        if not allowed_roles and not required_scopes:
            return

        if any(principal.has_role(role) for role in allowed_roles):
            return

        if required_scopes and principal.has_all_scopes(required_scopes):
            return

        detail = 'Your identity is not authorized for this endpoint.'
        if principal.auth_type == 'api_key':
            detail = 'Your API key is not authorized for this endpoint.'
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)

    def enforce_ip_rate_limit(self, request: Request) -> None:
        ip_limit = self.settings.api_rate_limit_requests_per_ip
        if ip_limit is None:
            return

        created_at, window_start, ip_address = self._rate_limit_context(request)
        with get_connection() as connection:
            self._prune_expired_events(connection, window_start)
            self._raise_if_limit_hit(
                connection=connection,
                subject_kind='ip',
                subject_id=ip_address,
                window_start=window_start,
                limit=ip_limit,
                message='Too many requests from this IP address. Please try again later.',
            )
            self._record_event(
                connection,
                subject_kind='ip',
                subject_id=ip_address,
                path=request.url.path,
                created_at=created_at,
            )

    def enforce_user_rate_limit(self, request: Request, principal: AuthPrincipal | None) -> None:
        user_limit = self.settings.api_rate_limit_requests_per_user
        if principal is None or user_limit is None:
            return

        created_at, window_start, _ = self._rate_limit_context(request)
        subject_label = 'authenticated user' if principal.auth_type == 'jwt' else 'API key'
        with get_connection() as connection:
            self._prune_expired_events(connection, window_start)
            self._raise_if_limit_hit(
                connection=connection,
                subject_kind='user',
                subject_id=principal.subject_rate_limit_key(),
                window_start=window_start,
                limit=user_limit,
                message=f'Too many requests for this {subject_label}. Please try again later.',
            )
            self._record_event(
                connection,
                subject_kind='user',
                subject_id=principal.subject_rate_limit_key(),
                path=request.url.path,
                created_at=created_at,
            )

    def _authenticate_api_key(self, api_key: str) -> AuthPrincipal:
        cleaned_key = api_key.strip()
        if not cleaned_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='An API key is required for this endpoint.',
            )

        role = self.settings.api_key_role_map().get(cleaned_key)
        if role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='The provided API key is invalid.',
            )

        return AuthPrincipal(
            auth_type='api_key',
            role=role,
            roles=(role,),
            scopes=(),
            subject_id=self._fingerprint(cleaned_key),
        )

    def _authenticate_jwt(self, bearer_token: str) -> AuthPrincipal:
        token = bearer_token.strip()
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail='A bearer token is required for this endpoint.',
            )

        try:
            validated = self.jwt_validator.validate(token)
        except JwtValidationError as exc:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
        except JwtConfigurationError as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
        except JwtMetadataError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='The identity provider metadata could not be validated right now.',
            ) from exc

        claims = validated.claims
        roles = self._extract_roles(claims)
        scopes = self._extract_scopes(claims)
        display_name = self._extract_display_name(claims)
        tenant_id = self._extract_optional_string(claims.get(self.settings.jwt_tenant_claim))
        token_id = self._extract_optional_string(claims.get('jti'))
        issuer = self._extract_optional_string(claims.get('iss'))
        subject = self._extract_optional_string(claims.get(self.settings.jwt_subject_claim))
        if subject is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f'The bearer token is missing the {self.settings.jwt_subject_claim} claim.',
            )

        primary_role = self._resolve_jwt_role(claims=claims, roles=roles, subject=subject)
        effective_roles = self._effective_roles(primary_role, roles)

        return AuthPrincipal(
            auth_type='jwt',
            subject_id=subject,
            role=primary_role,
            roles=effective_roles,
            scopes=tuple(scopes),
            tenant_id=tenant_id,
            issuer=issuer,
            token_id=token_id,
            display_name=display_name,
        )

    def _extract_roles(self, claims: dict[str, Any]) -> list[str]:
        claim_value = claims.get(self.settings.jwt_role_claim)
        values: list[str] = []
        if isinstance(claim_value, str):
            values = [claim_value]
        elif isinstance(claim_value, list):
            values = [str(item) for item in claim_value]

        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            candidate = value.strip().lower()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            normalized.append(candidate)
        return normalized

    def _extract_scopes(self, claims: dict[str, Any]) -> list[str]:
        claim_value = claims.get(self.settings.jwt_scope_claim)
        values: list[str] = []
        if isinstance(claim_value, str):
            values = [segment.strip().lower() for segment in claim_value.split(' ') if segment.strip()]
        elif isinstance(claim_value, list):
            values = [str(item).strip().lower() for item in claim_value if str(item).strip()]

        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            normalized.append(value)
        return normalized

    def _resolve_jwt_role(self, claims: dict[str, Any], roles: list[str], subject: str) -> str:
        if self._is_allowlisted_admin(claims=claims, subject=subject):
            return ROLE_ADMIN
        for role in ROLE_PRIORITY:
            if role in roles:
                return role
        return self.settings.jwt_default_role

    def _effective_roles(self, primary_role: str, roles: list[str]) -> tuple[str, ...]:
        ordered_roles: list[str] = []
        for role in [primary_role, *roles]:
            if role == ROLE_AUTHENTICATED or role in ordered_roles:
                continue
            ordered_roles.append(role)
        return tuple(ordered_roles)

    def _is_allowlisted_admin(self, claims: dict[str, Any], subject: str) -> bool:
        normalized_subjects = {value.strip() for value in self.settings.jwt_admin_subjects if value.strip()}
        if subject in normalized_subjects:
            return True

        normalized_emails = {value.strip().lower() for value in self.settings.jwt_admin_emails if value.strip()}
        if not normalized_emails:
            return False

        return bool(self._extract_identity_emails(claims).intersection(normalized_emails))

    def _extract_identity_emails(self, claims: dict[str, Any]) -> set[str]:
        emails: set[str] = set()
        for key in ('preferred_username', 'email', 'upn'):
            candidate = self._extract_optional_string(claims.get(key))
            if candidate:
                emails.add(candidate.lower())
        return emails

    def _extract_display_name(self, claims: dict[str, Any]) -> str | None:
        for key in (self.settings.jwt_name_claim, 'name', 'preferred_username', 'email', 'upn'):
            candidate = self._extract_optional_string(claims.get(key))
            if candidate:
                return candidate
        return None

    def _extract_optional_string(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        cleaned = value.strip()
        return cleaned or None

    def _rate_limit_context(self, request: Request) -> tuple[str, str, str]:
        created_at = iso_timestamp()
        window_start = (datetime.now(timezone.utc) - timedelta(seconds=self.settings.api_rate_limit_window_seconds)).isoformat()
        return created_at, window_start, self._client_ip(request)

    def _prune_expired_events(self, connection, window_start: str) -> None:
        connection.execute('DELETE FROM api_request_events WHERE created_at < ?', (window_start,))

    def _raise_if_limit_hit(
        self,
        connection,
        subject_kind: str,
        subject_id: str,
        window_start: str,
        limit: int,
        message: str,
    ) -> None:
        row = connection.execute(
            """
            SELECT COUNT(*) AS request_count
            FROM api_request_events
            WHERE subject_kind = ? AND subject_id = ? AND created_at >= ?
            """,
            (subject_kind, subject_id, window_start),
        ).fetchone()
        if int(row['request_count'] or 0) >= limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=message)

    def _record_event(self, connection, subject_kind: str, subject_id: str, path: str, created_at: str) -> None:
        connection.execute(
            """
            INSERT INTO api_request_events (
                event_id,
                subject_kind,
                subject_id,
                path,
                created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (generate_id('api'), subject_kind, subject_id, path, created_at),
        )

    def _client_ip(self, request: Request) -> str:
        if request.client is not None and request.client.host:
            return request.client.host
        return 'unknown'

    def _fingerprint(self, api_key: str) -> str:
        return hashlib.sha256(api_key.encode('utf-8')).hexdigest()[:16]
