from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.core.config import get_settings
from app.core.utils import generate_id, iso_timestamp
from app.db.sqlite import get_connection
from app.services.auth_context import get_current_principal


class ProviderUsageLimitExceeded(RuntimeError):
    """Raised when configured provider safety limits block a paid call."""


@dataclass
class ProviderUsageEstimate:
    input_chars: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class ProviderUsageSubject:
    subject_key: str
    subject_id: str
    tenant_id: str | None
    auth_type: str


class ProviderGuardService:
    """Enforces simple app-level request and spend caps for paid providers."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def check_and_record(self, provider: str, operation: str, texts: list[str]) -> None:
        if provider == 'local':
            return

        usage = self._estimate_usage(operation=operation, texts=texts)
        usage_date = self._usage_date()
        subject = self._current_subject()
        with get_connection() as connection:
            # Serialize quota checks and inserts so concurrent callers cannot both observe the same remaining budget.
            connection.execute('BEGIN IMMEDIATE')
            window_request_count = self._window_request_count(connection=connection, provider=provider)
            daily_usage = self._daily_usage(connection=connection, provider=provider, usage_date=usage_date)
            self._enforce_global_limits(
                provider=provider,
                window_request_count=window_request_count,
                daily_request_count=int(daily_usage['request_count'] or 0),
                daily_input_chars=int(daily_usage['input_chars'] or 0),
                daily_estimated_cost_usd=float(daily_usage['estimated_cost_usd'] or 0.0),
                pending=usage,
            )

            if subject is not None:
                subject_window_request_count = self._window_request_count(
                    connection=connection,
                    provider=provider,
                    subject=subject,
                )
                subject_daily_usage = self._daily_usage(
                    connection=connection,
                    provider=provider,
                    usage_date=usage_date,
                    subject=subject,
                )
                self._enforce_subject_limits(
                    provider=provider,
                    subject=subject,
                    window_request_count=subject_window_request_count,
                    daily_request_count=int(subject_daily_usage['request_count'] or 0),
                    daily_input_chars=int(subject_daily_usage['input_chars'] or 0),
                    daily_estimated_cost_usd=float(subject_daily_usage['estimated_cost_usd'] or 0.0),
                    pending=usage,
                )

            connection.execute(
                """
                INSERT INTO provider_usage_events (
                    event_id,
                    provider,
                    operation,
                    input_chars,
                    estimated_cost_usd,
                    created_at,
                    usage_date,
                    subject_key,
                    subject_id,
                    tenant_id,
                    auth_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    generate_id('usage'),
                    provider,
                    operation,
                    usage.input_chars,
                    usage.estimated_cost_usd,
                    iso_timestamp(),
                    usage_date,
                    subject.subject_key if subject is not None else None,
                    subject.subject_id if subject is not None else None,
                    subject.tenant_id if subject is not None else None,
                    subject.auth_type if subject is not None else None,
                ),
            )

    def _estimate_usage(self, operation: str, texts: list[str]) -> ProviderUsageEstimate:
        input_chars = sum(len(text) for text in texts)
        unit_cost = (
            self.settings.llm_estimated_cost_per_1k_chars_usd
            if operation == 'llm'
            else self.settings.embedding_estimated_cost_per_1k_chars_usd
        )
        estimated_cost_usd = round((input_chars / 1000) * unit_cost, 6)
        return ProviderUsageEstimate(input_chars=input_chars, estimated_cost_usd=estimated_cost_usd)

    def _current_subject(self) -> ProviderUsageSubject | None:
        principal = get_current_principal()
        if principal is None:
            return None
        return ProviderUsageSubject(
            subject_key=principal.subject_rate_limit_key(),
            subject_id=principal.subject_id,
            tenant_id=principal.tenant_id,
            auth_type=principal.auth_type,
        )

    def _usage_date(self) -> str:
        return datetime.now(timezone.utc).date().isoformat()

    def _window_request_count(self, connection, provider: str, subject: ProviderUsageSubject | None = None) -> int:
        window_start = (
            datetime.now(timezone.utc) - timedelta(seconds=self.settings.provider_rate_limit_window_seconds)
        ).isoformat()
        if subject is None:
            row = connection.execute(
                """
                SELECT COUNT(*) AS request_count
                FROM provider_usage_events
                WHERE provider = ? AND created_at >= ?
                """,
                (provider, window_start),
            ).fetchone()
        else:
            row = connection.execute(
                """
                SELECT COUNT(*) AS request_count
                FROM provider_usage_events
                WHERE provider = ? AND created_at >= ? AND subject_key = ?
                """,
                (provider, window_start, subject.subject_key),
            ).fetchone()
        return int(row['request_count'] or 0)

    def _daily_usage(self, connection, provider: str, usage_date: str, subject: ProviderUsageSubject | None = None):
        if subject is None:
            return connection.execute(
                """
                SELECT COUNT(*) AS request_count,
                       COALESCE(SUM(input_chars), 0) AS input_chars,
                       COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
                FROM provider_usage_events
                WHERE provider = ? AND usage_date = ?
                """,
                (provider, usage_date),
            ).fetchone()

        return connection.execute(
            """
            SELECT COUNT(*) AS request_count,
                   COALESCE(SUM(input_chars), 0) AS input_chars,
                   COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd
            FROM provider_usage_events
            WHERE provider = ? AND usage_date = ? AND subject_key = ?
            """,
            (provider, usage_date, subject.subject_key),
        ).fetchone()

    def _enforce_global_limits(
        self,
        provider: str,
        window_request_count: int,
        daily_request_count: int,
        daily_input_chars: int,
        daily_estimated_cost_usd: float,
        pending: ProviderUsageEstimate,
    ) -> None:
        if (
            self.settings.provider_max_requests_per_window is not None
            and window_request_count >= self.settings.provider_max_requests_per_window
        ):
            raise ProviderUsageLimitExceeded(
                f'{provider} request limit reached for the current {self.settings.provider_rate_limit_window_seconds}-second window.'
            )

        if (
            self.settings.provider_daily_request_limit is not None
            and daily_request_count >= self.settings.provider_daily_request_limit
        ):
            raise ProviderUsageLimitExceeded(f'{provider} daily request limit has been reached.')

        if (
            self.settings.provider_daily_input_char_limit is not None
            and daily_input_chars + pending.input_chars > self.settings.provider_daily_input_char_limit
        ):
            raise ProviderUsageLimitExceeded(f'{provider} daily input character limit would be exceeded.')

        if (
            self.settings.provider_daily_estimated_cost_limit_usd is not None
            and daily_estimated_cost_usd + pending.estimated_cost_usd > self.settings.provider_daily_estimated_cost_limit_usd
        ):
            raise ProviderUsageLimitExceeded(f'{provider} daily estimated cost limit would be exceeded.')

    def _enforce_subject_limits(
        self,
        provider: str,
        subject: ProviderUsageSubject,
        window_request_count: int,
        daily_request_count: int,
        daily_input_chars: int,
        daily_estimated_cost_usd: float,
        pending: ProviderUsageEstimate,
    ) -> None:
        subject_label = self._subject_limit_label(subject)

        if (
            self.settings.provider_per_user_max_requests_per_window is not None
            and window_request_count >= self.settings.provider_per_user_max_requests_per_window
        ):
            raise ProviderUsageLimitExceeded(
                f'{provider} request limit reached for {subject_label} in the current {self.settings.provider_rate_limit_window_seconds}-second window.'
            )

        if (
            self.settings.provider_per_user_daily_request_limit is not None
            and daily_request_count >= self.settings.provider_per_user_daily_request_limit
        ):
            raise ProviderUsageLimitExceeded(f'{provider} daily request limit has been reached for {subject_label}.')

        if (
            self.settings.provider_per_user_daily_input_char_limit is not None
            and daily_input_chars + pending.input_chars > self.settings.provider_per_user_daily_input_char_limit
        ):
            raise ProviderUsageLimitExceeded(
                f'{provider} daily input character limit would be exceeded for {subject_label}.'
            )

        if (
            self.settings.provider_per_user_daily_estimated_cost_limit_usd is not None
            and daily_estimated_cost_usd + pending.estimated_cost_usd > self.settings.provider_per_user_daily_estimated_cost_limit_usd
        ):
            raise ProviderUsageLimitExceeded(
                f'{provider} daily estimated cost limit would be exceeded for {subject_label}.'
            )

    def _subject_limit_label(self, subject: ProviderUsageSubject) -> str:
        if subject.auth_type == 'jwt':
            return 'this authenticated user'
        if subject.auth_type == 'api_key':
            return 'this API key'
        return 'this caller'
