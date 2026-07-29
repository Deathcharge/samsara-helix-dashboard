# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json

import pytest

from samsarix_discord_bot.config import ConfigError, load_config


def test_load_config_accepts_bounded_operator_configuration() -> None:
    env = {
        "DISCORD_BOT_TOKEN": "secret-token",
        "SAMSARIX_HEALTH_ENDPOINTS": json.dumps(
            [
                {"name": "API", "url": "https://api.example.com/health?full=0"},
                {"name": "Worker", "url": "http://worker.internal:8080/ready"},
            ]
        ),
        "SAMSARIX_ALLOWED_GUILD_IDS": "42, 84,42",
        "SAMSARIX_ALLOWED_ROLE_IDS": "7",
        "SAMSARIX_REQUEST_TIMEOUT_SECONDS": "2.5",
        "SAMSARIX_MAX_CONCURRENCY": "2",
        "SAMSARIX_CACHE_TTL_SECONDS": "30",
        "SAMSARIX_LOG_LEVEL": "warning",
    }

    config = load_config(env)

    assert config.token == "secret-token"
    assert [endpoint.name for endpoint in config.endpoints] == ["API", "Worker"]
    assert config.allowed_guild_ids == frozenset({42, 84})
    assert config.allowed_role_ids == frozenset({7})
    assert config.request_timeout_seconds == 2.5
    assert config.max_concurrency == 2
    assert config.cache_ttl_seconds == 30
    assert config.log_level == "WARNING"
    assert "secret-token" not in config.summary()
    assert "token=configured" in config.summary()


def test_missing_token_is_actionable() -> None:
    with pytest.raises(ConfigError, match="DISCORD_BOT_TOKEN is required"):
        load_config({})


def test_token_can_be_optional_for_non_runtime_callers() -> None:
    config = load_config({}, require_token=False)
    assert config.token is None
    assert config.endpoints == ()
    assert "token=missing" in config.summary()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not json", "must be valid JSON"),
        ('{"name":"API"}', "must be a JSON array"),
        ('[{"name":"API"}]', "url must be a non-empty string"),
        ('[{"name":"API","url":"ftp://example.com"}]', "absolute HTTP or HTTPS"),
        (
            '[{"name":"API","url":"https://user:pass@example.com"}]',
            "must not contain credentials",
        ),
        ('[{"name":"API","url":"https://example.com/#secret"}]', "must not contain a fragment"),
        (
            '[{"name":"API\\nprod","url":"https://example.com"}]',
            "only printable characters",
        ),
        (
            '[{"name":"API","url":"https://example.com/\\tprod"}]',
            "whitespace or control characters",
        ),
        (
            '[{"name":"API","url":"https://a.example"},{"name":"api","url":"https://b.example"}]',
            "names must be unique",
        ),
    ],
)
def test_endpoint_validation_rejects_unsafe_or_ambiguous_values(value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config({"DISCORD_BOT_TOKEN": "token", "SAMSARIX_HEALTH_ENDPOINTS": value})


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (json.dumps(["not-an-object"]), "must be an object"),
        (json.dumps([{"name": "x" * 51, "url": "https://example.com"}]), "at most 50"),
        (json.dumps([{"name": "API", "url": "https://example.com/" + "x" * 2049}]), "at most 2048"),
        (
            json.dumps(
                [{"name": f"API-{index}", "url": "https://example.com"} for index in range(21)]
            ),
            "at most 20",
        ),
    ],
)
def test_endpoint_validation_rejects_shape_and_size_limits(value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config({"DISCORD_BOT_TOKEN": "token", "SAMSARIX_HEALTH_ENDPOINTS": value})


@pytest.mark.parametrize(
    ("variable", "value", "message"),
    [
        ("SAMSARIX_ALLOWED_GUILD_IDS", "1,nope", "comma-separated"),
        ("SAMSARIX_ALLOWED_ROLE_IDS", "-1", "positive integers"),
        ("SAMSARIX_REQUEST_TIMEOUT_SECONDS", "0", "between 1 and 30"),
        ("SAMSARIX_REQUEST_TIMEOUT_SECONDS", "nan", "between 1 and 30"),
        ("SAMSARIX_MAX_CONCURRENCY", "21", "between 1 and 20"),
        ("SAMSARIX_CACHE_TTL_SECONDS", "4", "between 5 and 300"),
        ("SAMSARIX_LOG_LEVEL", "TRACE", "must be one of"),
    ],
)
def test_scalar_validation_is_fail_fast(variable: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config({"DISCORD_BOT_TOKEN": "token", variable: value})
