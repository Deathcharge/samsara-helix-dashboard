# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Environment-backed configuration with strict, secret-safe validation."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_ENDPOINTS = 20
MAX_ENDPOINT_NAME_LENGTH = 50
MAX_URL_LENGTH = 2048
MAX_HEADERS = 20
MAX_HEADER_VALUE_LENGTH = 2048
MAX_EXPECTED_STATUSES = 20
DEFAULT_EXPECTED_STATUSES = frozenset(range(200, 300))
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
HEADER_ENV_PATTERN = re.compile(r"SAMSARIX_ENDPOINT_HEADERS_[A-Z0-9_]{1,64}")
HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
FORBIDDEN_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class ConfigError(ValueError):
    """Raised when runtime configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class HealthEndpoint:
    """An operator-controlled HTTP endpoint that the bot may monitor."""

    name: str
    url: str
    expected_statuses: frozenset[int] = DEFAULT_EXPECTED_STATUSES
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class BotConfig:
    """Validated runtime configuration."""

    token: str | None
    endpoints: tuple[HealthEndpoint, ...]
    allowed_guild_ids: frozenset[int]
    allowed_role_ids: frozenset[int]
    request_timeout_seconds: float
    max_concurrency: int
    cache_ttl_seconds: float
    alert_channel_id: int | None
    poll_interval_seconds: float
    failure_threshold: int
    recovery_threshold: int
    log_level: str

    def summary(self) -> str:
        """Return a token-free summary suitable for logs and CLI output."""
        guild_scope = (
            f"{len(self.allowed_guild_ids)} configured guild(s)"
            if self.allowed_guild_ids
            else "all installed guilds"
        )
        role_scope = (
            f"{len(self.allowed_role_ids)} configured role(s)"
            if self.allowed_role_ids
            else "all guild members"
        )
        alerting = (
            f"enabled ({self.failure_threshold} fail/{self.recovery_threshold} recover)"
            if self.alert_channel_id is not None
            else "disabled"
        )
        authenticated_endpoints = sum(bool(endpoint.headers) for endpoint in self.endpoints)
        return (
            f"token={'configured' if self.token else 'missing'}, "
            f"endpoints={len(self.endpoints)}, authenticated_endpoints={authenticated_endpoints}, "
            f"guild_scope={guild_scope}, "
            f"role_scope={role_scope}, timeout={self.request_timeout_seconds:g}s, "
            f"max_concurrency={self.max_concurrency}, cache_ttl={self.cache_ttl_seconds:g}s, "
            f"alerting={alerting}"
        )


def _parse_positive_ids(raw: str, variable: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()

    values: set[int] = set()
    for item in raw.split(","):
        candidate = item.strip()
        try:
            parsed = int(candidate)
        except ValueError as error:
            message = f"{variable} must be a comma-separated list of positive integers"
            raise ConfigError(message) from error
        if parsed <= 0:
            raise ConfigError(f"{variable} must contain only positive integers")
        values.add(parsed)
    return frozenset(values)


def _parse_optional_positive_id(raw: str, variable: str) -> int | None:
    if not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{variable} must be a positive integer") from error
    if value <= 0:
        raise ConfigError(f"{variable} must be a positive integer")
    return value


def _parse_float(
    raw: str,
    variable: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigError(f"{variable} must be a number") from error
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ConfigError(f"{variable} must be between {minimum:g} and {maximum:g}")
    return value


def _parse_int(
    raw: str,
    variable: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    try:
        value = int(raw)
    except ValueError as error:
        raise ConfigError(f"{variable} must be an integer") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{variable} must be between {minimum} and {maximum}")
    return value


def _parse_expected_statuses(value: object, label: str) -> frozenset[int]:
    if value is None:
        return DEFAULT_EXPECTED_STATUSES
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{label}.expected_statuses must be a non-empty JSON array")
    if len(value) > MAX_EXPECTED_STATUSES:
        raise ConfigError(
            f"{label}.expected_statuses supports at most {MAX_EXPECTED_STATUSES} values"
        )
    statuses: set[int] = set()
    for status in value:
        if isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
            raise ConfigError(
                f"{label}.expected_statuses must contain only HTTP status integers from 100 to 599"
            )
        if status in statuses:
            raise ConfigError(f"{label}.expected_statuses must contain unique values")
        statuses.add(status)
    return frozenset(statuses)


def _parse_headers(
    headers_env: object,
    source: Mapping[str, str],
    label: str,
) -> tuple[tuple[str, str], ...]:
    if headers_env is None:
        return ()
    if not isinstance(headers_env, str) or not HEADER_ENV_PATTERN.fullmatch(headers_env.strip()):
        raise ConfigError(
            f"{label}.headers_env must name a SAMSARIX_ENDPOINT_HEADERS_* environment variable"
        )
    variable = headers_env.strip()
    raw = source.get(variable)
    if raw is None or not raw.strip():
        raise ConfigError(f"{label}.headers_env references missing environment variable {variable}")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigError(f"{variable} must be a valid JSON object") from error
    if not isinstance(decoded, dict):
        raise ConfigError(f"{variable} must be a JSON object of HTTP header names and values")
    if len(decoded) > MAX_HEADERS:
        raise ConfigError(f"{variable} supports at most {MAX_HEADERS} headers")

    headers: list[tuple[str, str]] = []
    seen_names: set[str] = set()
    for name, value in decoded.items():
        if not isinstance(name, str) or not HEADER_NAME_PATTERN.fullmatch(name):
            raise ConfigError(f"{variable} contains an invalid HTTP header name")
        normalized_name = name.casefold()
        if normalized_name in seen_names:
            raise ConfigError(f"{variable} header names must be unique ignoring case")
        if normalized_name in FORBIDDEN_HEADERS:
            raise ConfigError(f"{variable} must not set transport-controlled header {name}")
        if not isinstance(value, str):
            raise ConfigError(f"{variable} HTTP header values must be strings")
        if len(value) > MAX_HEADER_VALUE_LENGTH:
            raise ConfigError(
                f"{variable} HTTP header values must be at most "
                f"{MAX_HEADER_VALUE_LENGTH} characters"
            )
        if any(not character.isprintable() for character in value):
            raise ConfigError(
                f"{variable} HTTP header values must contain only printable characters"
            )
        seen_names.add(normalized_name)
        headers.append((name, value))
    return tuple(headers)


def _validate_endpoint(
    index: int,
    item: object,
    source: Mapping[str, str],
) -> HealthEndpoint:
    label = f"SAMSARIX_HEALTH_ENDPOINTS[{index}]"
    if not isinstance(item, dict):
        raise ConfigError(f"{label} must be an object with name and url fields")

    unknown_fields = set(item) - {"name", "url", "expected_statuses", "headers_env"}
    if unknown_fields:
        fields = ", ".join(sorted(str(field) for field in unknown_fields))
        raise ConfigError(f"{label} contains unsupported field(s): {fields}")

    name = item.get("name")
    url = item.get("url")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError(f"{label}.name must be a non-empty string")
    name = name.strip()
    if len(name) > MAX_ENDPOINT_NAME_LENGTH:
        raise ConfigError(f"{label}.name must be at most {MAX_ENDPOINT_NAME_LENGTH} characters")
    if any(not character.isprintable() for character in name):
        raise ConfigError(f"{label}.name must contain only printable characters")

    if not isinstance(url, str) or not url.strip():
        raise ConfigError(f"{label}.url must be a non-empty string")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise ConfigError(f"{label}.url must be at most {MAX_URL_LENGTH} characters")
    if any(character.isspace() or not character.isprintable() for character in url):
        raise ConfigError(f"{label}.url must not contain whitespace or control characters")

    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        _port = parsed.port
    except ValueError as error:
        raise ConfigError(f"{label}.url must be a valid absolute HTTP or HTTPS URL") from error
    if parsed.scheme not in {"http", "https"} or not hostname:
        raise ConfigError(f"{label}.url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(f"{label}.url must not contain credentials")
    if parsed.fragment:
        raise ConfigError(f"{label}.url must not contain a fragment")

    headers = _parse_headers(item.get("headers_env"), source, label)
    if headers and parsed.scheme != "https":
        raise ConfigError(f"{label}.url must use https when headers_env is configured")

    return HealthEndpoint(
        name=name,
        url=url,
        expected_statuses=_parse_expected_statuses(item.get("expected_statuses"), label),
        headers=headers,
    )


def _parse_endpoints(raw: str, source: Mapping[str, str]) -> tuple[HealthEndpoint, ...]:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigError("SAMSARIX_HEALTH_ENDPOINTS must be valid JSON") from error

    if not isinstance(decoded, list):
        raise ConfigError("SAMSARIX_HEALTH_ENDPOINTS must be a JSON array")
    if len(decoded) > MAX_ENDPOINTS:
        raise ConfigError(f"SAMSARIX_HEALTH_ENDPOINTS supports at most {MAX_ENDPOINTS} endpoints")

    endpoints = tuple(
        _validate_endpoint(index, item, source) for index, item in enumerate(decoded)
    )
    names = [endpoint.name.casefold() for endpoint in endpoints]
    if len(names) != len(set(names)):
        raise ConfigError("SAMSARIX_HEALTH_ENDPOINTS names must be unique")
    return endpoints


def load_config(
    env: Mapping[str, str] | None = None,
    *,
    require_token: bool = True,
) -> BotConfig:
    """Load and validate configuration without logging secret values."""
    source = os.environ if env is None else env
    token = source.get("DISCORD_BOT_TOKEN", "").strip() or None
    if require_token and token is None:
        raise ConfigError("DISCORD_BOT_TOKEN is required")

    log_level = source.get("SAMSARIX_LOG_LEVEL", "INFO").strip().upper()
    if log_level not in VALID_LOG_LEVELS:
        choices = ", ".join(sorted(VALID_LOG_LEVELS))
        raise ConfigError(f"SAMSARIX_LOG_LEVEL must be one of {choices}")

    endpoints = _parse_endpoints(source.get("SAMSARIX_HEALTH_ENDPOINTS", "[]"), source)
    alert_channel_id = _parse_optional_positive_id(
        source.get("SAMSARIX_ALERT_CHANNEL_ID", ""), "SAMSARIX_ALERT_CHANNEL_ID"
    )
    if alert_channel_id is not None and not endpoints:
        raise ConfigError("SAMSARIX_ALERT_CHANNEL_ID requires at least one health endpoint")

    return BotConfig(
        token=token,
        endpoints=endpoints,
        allowed_guild_ids=_parse_positive_ids(
            source.get("SAMSARIX_ALLOWED_GUILD_IDS", ""), "SAMSARIX_ALLOWED_GUILD_IDS"
        ),
        allowed_role_ids=_parse_positive_ids(
            source.get("SAMSARIX_ALLOWED_ROLE_IDS", ""), "SAMSARIX_ALLOWED_ROLE_IDS"
        ),
        request_timeout_seconds=_parse_float(
            source.get("SAMSARIX_REQUEST_TIMEOUT_SECONDS", "5"),
            "SAMSARIX_REQUEST_TIMEOUT_SECONDS",
            minimum=1,
            maximum=30,
        ),
        max_concurrency=_parse_int(
            source.get("SAMSARIX_MAX_CONCURRENCY", "5"),
            "SAMSARIX_MAX_CONCURRENCY",
            minimum=1,
            maximum=20,
        ),
        cache_ttl_seconds=_parse_float(
            source.get("SAMSARIX_CACHE_TTL_SECONDS", "15"),
            "SAMSARIX_CACHE_TTL_SECONDS",
            minimum=5,
            maximum=300,
        ),
        alert_channel_id=alert_channel_id,
        poll_interval_seconds=_parse_float(
            source.get("SAMSARIX_POLL_INTERVAL_SECONDS", "60"),
            "SAMSARIX_POLL_INTERVAL_SECONDS",
            minimum=30,
            maximum=3600,
        ),
        failure_threshold=_parse_int(
            source.get("SAMSARIX_FAILURE_THRESHOLD", "2"),
            "SAMSARIX_FAILURE_THRESHOLD",
            minimum=1,
            maximum=10,
        ),
        recovery_threshold=_parse_int(
            source.get("SAMSARIX_RECOVERY_THRESHOLD", "2"),
            "SAMSARIX_RECOVERY_THRESHOLD",
            minimum=1,
            maximum=10,
        ),
        log_level=log_level,
    )
