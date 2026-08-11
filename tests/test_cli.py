# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import json
from types import SimpleNamespace

import discord
import pytest

import samsarix_discord_bot.cli as cli
from samsarix_discord_bot.cli import main
from samsarix_discord_bot.config import BotConfig, HealthEndpoint
from samsarix_discord_bot.health import HealthResult, HealthState


def test_check_config_succeeds_without_network(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "do-not-print-me")
    monkeypatch.delenv("SAMSARIX_HEALTH_ENDPOINTS", raising=False)

    result = main(["check-config"])

    captured = capsys.readouterr()
    assert result == 0
    assert "Configuration valid" in captured.out
    assert "no health endpoints" in captured.out
    assert "do-not-print-me" not in captured.out


def test_check_config_reports_missing_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    result = main(["check-config"])
    captured = capsys.readouterr()
    assert result == 2
    assert "DISCORD_BOT_TOKEN is required" in captured.err


def test_check_endpoints_is_token_independent_and_hides_urls(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.setenv(
        "SAMSARIX_HEALTH_ENDPOINTS",
        '[{"name":"API","url":"https://private.example/health"}]',
    )
    endpoint = HealthEndpoint("API", "https://private.example/health")

    class StubChecker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def check_all(self) -> tuple[HealthResult, ...]:
            return (HealthResult(endpoint, HealthState.HEALTHY, 12, 200),)

    monkeypatch.setattr(cli, "HealthChecker", StubChecker)
    assert main(["check-endpoints"]) == 0
    captured = capsys.readouterr()
    assert "API: healthy · HTTP 200 · 12 ms" in captured.out
    assert "private.example" not in captured.out


def test_check_endpoints_returns_nonzero_for_empty_or_unhealthy_results(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SAMSARIX_HEALTH_ENDPOINTS", raising=False)
    assert main(["check-endpoints"]) == 4
    assert "is empty" in capsys.readouterr().err

    monkeypatch.setenv(
        "SAMSARIX_HEALTH_ENDPOINTS",
        '[{"name":"API","url":"https://private.example/health"}]',
    )
    endpoint = HealthEndpoint("API", "https://private.example/health")

    class StubChecker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def check_all(self) -> tuple[HealthResult, ...]:
            return (HealthResult(endpoint, HealthState.UNHEALTHY, 10, 503),)

    monkeypatch.setattr(cli, "HealthChecker", StubChecker)
    assert main(["check-endpoints"]) == 4


def test_check_endpoints_json_is_stable_and_secret_safe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(
        "SAMSARIX_HEALTH_ENDPOINTS",
        '[{"name":"API","url":"https://private.example/health"}]',
    )
    endpoint = HealthEndpoint("API", "https://private.example/health")

    class StubChecker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def check_all(self) -> tuple[HealthResult, ...]:
            return (HealthResult(endpoint, HealthState.DEGRADED, 12, 302, "redirect not followed"),)

    monkeypatch.setattr(cli, "HealthChecker", StubChecker)

    assert main(["check-endpoints", "--format", "json"]) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "overall": "degraded",
        "results": [
            {
                "detail": "redirect not followed",
                "latency_ms": 12,
                "name": "API",
                "state": "degraded",
                "status_code": 302,
            }
        ],
        "schema_version": 1,
    }
    assert "private.example" not in json.dumps(payload)


def test_check_endpoints_json_reports_empty_configuration(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("SAMSARIX_HEALTH_ENDPOINTS", raising=False)

    assert main(["check-endpoints", "--format", "json"]) == 4
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"] == "unconfigured"
    assert payload["results"] == []


def test_check_endpoints_json_reports_configuration_error_without_secret(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("SAMSARIX_HEALTH_ENDPOINTS", "not-json-private-value")

    assert main(["check-endpoints", "--format", "json"]) == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"] == "configuration_error"
    assert "not-json-private-value" not in json.dumps(payload)


def test_check_endpoints_fails_without_leaking_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(
        "SAMSARIX_HEALTH_ENDPOINTS",
        '[{"name":"API","url":"https://private.example/health"}]',
    )

    class StubChecker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def check_all(self) -> tuple[HealthResult, ...]:
            raise RuntimeError("sensitive upstream detail")

    monkeypatch.setattr(cli, "HealthChecker", StubChecker)
    assert main(["check-endpoints"]) == 5
    captured = capsys.readouterr()
    assert "failed unexpectedly" in captured.err
    assert "sensitive upstream detail" not in captured.err


def test_check_endpoints_json_contains_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(
        "SAMSARIX_HEALTH_ENDPOINTS",
        '[{"name":"API","url":"https://private.example/health"}]',
    )

    class StubChecker:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def check_all(self) -> tuple[HealthResult, ...]:
            raise RuntimeError("sensitive upstream detail")

    monkeypatch.setattr(cli, "HealthChecker", StubChecker)

    assert main(["check-endpoints", "--format", "json"]) == 5
    payload = json.loads(capsys.readouterr().out)
    assert payload["overall"] == "error"
    assert "sensitive upstream detail" not in json.dumps(payload)


def test_run_invokes_bot_without_printing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "do-not-print-me")

    def run(*args: object, **kwargs: object) -> None:
        return None

    fake_bot = SimpleNamespace(run=run)

    def create_bot(config: BotConfig) -> SimpleNamespace:
        return fake_bot

    monkeypatch.setattr(cli, "create_bot", create_bot)
    assert main(["run"]) == 0


def test_run_reports_rejected_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "invalid")

    def reject(*args: object, **kwargs: object) -> None:
        raise discord.LoginFailure("rejected")

    def create_bot(config: BotConfig) -> SimpleNamespace:
        return SimpleNamespace(run=reject)

    monkeypatch.setattr(cli, "create_bot", create_bot)
    assert main(["run"]) == 3
