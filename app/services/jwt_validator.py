from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from app.core.config import Settings, get_settings


SUPPORTED_ALGORITHMS = {'HS256', 'RS256'}
JWKS_CACHE_TTL_SECONDS = 300
DISCOVERY_CACHE_TTL_SECONDS = 300
_JWKS_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_DISCOVERY_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


class JwtValidationError(ValueError):
    """Raised when a JWT is malformed or fails validation."""


class JwtConfigurationError(ValueError):
    """Raised when JWT settings are incomplete or inconsistent."""


class JwtMetadataError(RuntimeError):
    """Raised when issuer metadata or JWKS cannot be retrieved."""


@dataclass(frozen=True)
class ValidatedJwt:
    header: dict[str, Any]
    claims: dict[str, Any]


class JwtValidator:
    """Validates bearer tokens without trusting any frontend-provided identity fields."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def validate(self, token: str) -> ValidatedJwt:
        header, claims, signing_input, signature = self._parse_token(token)
        algorithm = str(header.get('alg') or '')
        if algorithm not in self.settings.jwt_algorithms:
            raise JwtValidationError('The bearer token uses an unsupported signing algorithm.')
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise JwtConfigurationError(
                f'JWT algorithm {algorithm} is configured but not supported by the application.'
            )

        self._verify_signature(
            header=header,
            algorithm=algorithm,
            signing_input=signing_input,
            signature=signature,
        )
        self._validate_claims(claims)
        return ValidatedJwt(header=header, claims=claims)

    def _parse_token(self, token: str) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
        segments = token.split('.')
        if len(segments) != 3:
            raise JwtValidationError('The bearer token is malformed.')

        header_segment, payload_segment, signature_segment = segments
        signing_input = f'{header_segment}.{payload_segment}'.encode('ascii')
        try:
            header = json.loads(self._b64decode(header_segment))
            claims = json.loads(self._b64decode(payload_segment))
            signature = self._b64decode_bytes(signature_segment)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise JwtValidationError('The bearer token payload could not be decoded.') from exc

        if not isinstance(header, dict) or not isinstance(claims, dict):
            raise JwtValidationError('The bearer token payload is invalid.')
        return header, claims, signing_input, signature

    def _verify_signature(
        self,
        header: dict[str, Any],
        algorithm: str,
        signing_input: bytes,
        signature: bytes,
    ) -> None:
        if algorithm == 'HS256':
            if not self.settings.jwt_shared_secret:
                raise JwtConfigurationError('JWT_SHARED_SECRET is required for HS256 token validation.')
            expected = hmac.new(
                self.settings.jwt_shared_secret.encode('utf-8'),
                signing_input,
                hashlib.sha256,
            ).digest()
            if not hmac.compare_digest(expected, signature):
                raise JwtValidationError('The bearer token signature is invalid.')
            return

        if algorithm == 'RS256':
            public_key = self._resolve_rsa_public_key(header=header)
            try:
                public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
            except InvalidSignature as exc:
                raise JwtValidationError('The bearer token signature is invalid.') from exc
            return

        raise JwtConfigurationError(f'JWT algorithm {algorithm} is not supported.')

    def _validate_claims(self, claims: dict[str, Any]) -> None:
        now = time.time()
        clock_skew = float(self.settings.jwt_clock_skew_seconds)

        subject_claim = self.settings.jwt_subject_claim
        subject = claims.get(subject_claim)
        if not isinstance(subject, str) or not subject.strip():
            raise JwtValidationError(f'The bearer token is missing the {subject_claim} claim.')

        issuer = claims.get('iss')
        if self.settings.jwt_issuer and issuer != self.settings.jwt_issuer:
            raise JwtValidationError('The bearer token issuer is not allowed.')

        if self.settings.jwt_audiences:
            claim_audiences = self._normalize_audience_claim(claims.get('aud'))
            if not claim_audiences.intersection(self.settings.jwt_audiences):
                raise JwtValidationError('The bearer token audience is not allowed.')

        token_type = str(claims.get('typ') or claims.get('token_use') or '').lower()
        if token_type and token_type not in {'access', 'at+jwt', 'jwt'}:
            raise JwtValidationError('The bearer token type is not allowed for API access.')

        expiration = self._read_numeric_claim(claims, 'exp', required=True)
        if expiration < (now - clock_skew):
            raise JwtValidationError('The bearer token has expired.')

        not_before = self._read_numeric_claim(claims, 'nbf', required=False)
        if not_before is not None and not_before > (now + clock_skew):
            raise JwtValidationError('The bearer token is not valid yet.')

        issued_at = self._read_numeric_claim(claims, 'iat', required=False)
        if issued_at is not None and issued_at > (now + clock_skew):
            raise JwtValidationError('The bearer token has an invalid issued-at timestamp.')

    def _read_numeric_claim(self, claims: dict[str, Any], claim_name: str, *, required: bool) -> float | None:
        value = claims.get(claim_name)
        if value is None:
            if required:
                raise JwtValidationError(f'The bearer token is missing the {claim_name} claim.')
            return None
        if not isinstance(value, (int, float)):
            raise JwtValidationError(f'The bearer token contains an invalid {claim_name} claim.')
        return float(value)

    def _resolve_rsa_public_key(self, header: dict[str, Any]) -> rsa.RSAPublicKey:
        kid = header.get('kid')
        if not isinstance(kid, str) or not kid.strip():
            raise JwtValidationError('The bearer token is missing a key identifier.')

        jwks = self._load_jwks()
        for key in jwks.get('keys', []):
            if key.get('kid') != kid:
                continue
            if key.get('kty') != 'RSA':
                raise JwtValidationError('The bearer token uses an unsupported key type.')
            return self._rsa_public_key_from_jwk(key)

        self._clear_jwks_cache()
        jwks = self._load_jwks()
        for key in jwks.get('keys', []):
            if key.get('kid') == kid and key.get('kty') == 'RSA':
                return self._rsa_public_key_from_jwk(key)
        raise JwtValidationError('The bearer token references an unknown signing key.')

    def _load_jwks(self) -> dict[str, Any]:
        url = self._resolve_jwks_url()
        cached = _JWKS_CACHE.get(url)
        now = time.time()
        if cached and cached[0] > now:
            return cached[1]

        if self.settings.jwt_require_https_metadata and not url.startswith('https://'):
            raise JwtConfigurationError('JWT metadata URLs must use HTTPS.')

        try:
            response = httpx.get(url, timeout=5.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover
            raise JwtMetadataError('Unable to retrieve the JWT signing keys.') from exc

        if not isinstance(payload, dict) or not isinstance(payload.get('keys'), list):
            raise JwtMetadataError('The JWT signing key payload is invalid.')

        _JWKS_CACHE[url] = (now + JWKS_CACHE_TTL_SECONDS, payload)
        return payload

    def _resolve_jwks_url(self) -> str:
        if self.settings.jwt_jwks_url:
            return self.settings.jwt_jwks_url
        if not self.settings.jwt_issuer:
            raise JwtConfigurationError('JWT_ISSUER or JWT_JWKS_URL must be configured for RS256 validation.')

        issuer = self.settings.jwt_issuer.rstrip('/')
        discovery_url = f'{issuer}/.well-known/openid-configuration'
        cached = _DISCOVERY_CACHE.get(discovery_url)
        now = time.time()
        if cached and cached[0] > now:
            jwks_uri = cached[1].get('jwks_uri')
            if isinstance(jwks_uri, str) and jwks_uri.strip():
                return jwks_uri

        if self.settings.jwt_require_https_metadata and not discovery_url.startswith('https://'):
            raise JwtConfigurationError('JWT metadata URLs must use HTTPS.')

        try:
            response = httpx.get(discovery_url, timeout=5.0)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover
            raise JwtMetadataError('Unable to retrieve the issuer metadata for JWT validation.') from exc

        jwks_uri = payload.get('jwks_uri')
        if not isinstance(jwks_uri, str) or not jwks_uri.strip():
            raise JwtMetadataError('The issuer metadata does not include a JWKS URI.')
        _DISCOVERY_CACHE[discovery_url] = (now + DISCOVERY_CACHE_TTL_SECONDS, payload)
        return jwks_uri

    def _clear_jwks_cache(self) -> None:
        if self.settings.jwt_jwks_url:
            _JWKS_CACHE.pop(self.settings.jwt_jwks_url, None)
            return
        if self.settings.jwt_issuer:
            issuer = self.settings.jwt_issuer.rstrip('/')
            discovery_url = f'{issuer}/.well-known/openid-configuration'
            cached = _DISCOVERY_CACHE.pop(discovery_url, None)
            if cached:
                jwks_uri = cached[1].get('jwks_uri')
                if isinstance(jwks_uri, str):
                    _JWKS_CACHE.pop(jwks_uri, None)

    def _rsa_public_key_from_jwk(self, jwk: dict[str, Any]) -> rsa.RSAPublicKey:
        modulus = jwk.get('n')
        exponent = jwk.get('e')
        if not isinstance(modulus, str) or not isinstance(exponent, str):
            raise JwtValidationError('The signing key payload is missing RSA key parameters.')
        public_numbers = rsa.RSAPublicNumbers(
            e=int.from_bytes(self._b64decode_bytes(exponent), 'big'),
            n=int.from_bytes(self._b64decode_bytes(modulus), 'big'),
        )
        return public_numbers.public_key()

    def _normalize_audience_claim(self, value: Any) -> set[str]:
        if isinstance(value, str):
            return {value}
        if isinstance(value, list):
            return {str(item) for item in value if str(item).strip()}
        return set()

    def _b64decode(self, value: str) -> str:
        return self._b64decode_bytes(value).decode('utf-8')

    def _b64decode_bytes(self, value: str) -> bytes:
        padding_length = (-len(value)) % 4
        padded = f'{value}{"=" * padding_length}'
        try:
            return base64.urlsafe_b64decode(padded.encode('ascii'))
        except Exception as exc:
            raise ValueError('Invalid base64url value.') from exc
