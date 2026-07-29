# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Environment-backed configuration with strict, secret-safe validation."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

MAX_ENDPOINTS = 20
MAX_ENDPOINT_NAME_LENGTH = 50
MAX_URL_LENGTH = 2048
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class ConfigError(ValueError):
    """Raised when runtime configuration is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class HealthEndpoint:
    """An operator-controlled HTTP endpoint that the bot may monitor."""

    name: str
    url: str


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
        return (
            f"token={'configured' if self.token else 'missing'}, "
            f"endpoints={len(self.endpoints)}, guild_scope={guild_scope}, "
            f"role_scope={role_scope}, timeout={self.request_timeout_seconds:g}s, "
            f"max_concurrency={self.max_concurrency}, cache_ttl={self.cache_ttl_seconds:g}s"
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


def _validate_endpoint(index: int, item: object) -> HealthEndpoint:
    label = f"SAMSARIX_HEALTH_ENDPOINTS[{index}]"
    if not isinstance(item, dict):
        raise ConfigError(f"{label} must be an object with name and url fields")

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

    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{label}.url must be an absolute HTTP or HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError(f"{label}.url must not contain credentials")
    if parsed.fragment:
        raise ConfigError(f"{label}.url must not contain a fragment")

    return HealthEndpoint(name=name, url=url)


def _parse_endpoints(raw: str) -> tuple[HealthEndpoint, ...]:
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ConfigError("SAMSARIX_HEALTH_ENDPOINTS must be valid JSON") from error

    if not isinstance(decoded, list):
        raise ConfigError("SAMSARIX_HEALTH_ENDPOINTS must be a JSON array")
    if len(decoded) > MAX_ENDPOINTS:
        raise ConfigError(f"SAMSARIX_HEALTH_ENDPOINTS supports at most {MAX_ENDPOINTS} endpoints")

    endpoints = tuple(_validate_endpoint(index, item) for index, item in enumerate(decoded))
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

    return BotConfig(
        token=token,
        endpoints=_parse_endpoints(source.get("SAMSARIX_HEALTH_ENDPOINTS", "[]")),
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
        log_level=log_level,
    )
