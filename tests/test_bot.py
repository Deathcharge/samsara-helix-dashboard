# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from discord import app_commands

from samsarix_discord_bot.alerts import AlertEvent, AlertKind
from samsarix_discord_bot.bot import (
    CachedStatusService,
    SamsarixOperatorBot,
    build_alert_embed,
    build_status_embed,
    create_bot,
    interaction_is_authorized,
)
from samsarix_discord_bot.config import BotConfig, HealthEndpoint
from samsarix_discord_bot.health import HealthResult, HealthState


def make_config(**overrides: object) -> BotConfig:
    values: dict[str, object] = {
        "token": "token",
        "endpoints": (),
        "allowed_guild_ids": frozenset(),
        "allowed_role_ids": frozenset(),
        "request_timeout_seconds": 5.0,
        "max_concurrency": 5,
        "cache_ttl_seconds": 15.0,
        "alert_channel_id": None,
        "poll_interval_seconds": 60.0,
        "failure_threshold": 2,
        "recovery_threshold": 2,
        "log_level": "INFO",
    }
    values.update(overrides)
    return BotConfig(**cast(Any, values))


def test_create_bot_registers_only_supported_command_group() -> None:
    bot = create_bot(make_config())
    commands = bot.tree.get_commands()
    assert [command.name for command in commands] == ["samsarix"]
    group = commands[0]
    assert isinstance(group, app_commands.Group)
    assert {command.name for command in group.commands} == {"about", "check", "ping", "status"}
    assert bot.intents.guilds is True
    assert bot.intents.message_content is False


def test_build_status_embed_hides_url_and_escapes_name() -> None:
    endpoint = HealthEndpoint("API *private*", "https://secret.internal/health")
    result = HealthResult(endpoint, HealthState.HEALTHY, 12, 200)
    embed = build_status_embed((result,))
    rendered = str(embed.to_dict())
    assert "secret.internal" not in rendered
    assert embed.fields[0].name == r"API \*private\*"
    assert "HTTP 200" in rendered


def test_build_alert_embed_hides_url_and_mentions() -> None:
    endpoint = HealthEndpoint("API @everyone", "https://secret.internal/health")
    result = HealthResult(endpoint, HealthState.UNHEALTHY, 12, 503, "unexpected response")
    embed = build_alert_embed(AlertEvent(AlertKind.INCIDENT, result))
    rendered = str(embed.to_dict())
    assert "secret.internal" not in rendered
    assert "@everyone" not in rendered
    assert "HTTP 503" in rendered


def test_authorization_requires_guild_and_honors_guild_allowlist() -> None:
    config = make_config(allowed_guild_ids=frozenset({42}))
    allowed = cast(Any, SimpleNamespace(guild_id=42, user=SimpleNamespace()))
    denied = cast(Any, SimpleNamespace(guild_id=7, user=SimpleNamespace()))
    dm = cast(Any, SimpleNamespace(guild_id=None, user=SimpleNamespace()))
    assert interaction_is_authorized(config, allowed)
    assert not interaction_is_authorized(config, denied)
    assert not interaction_is_authorized(config, dm)


@pytest.mark.asyncio
async def test_cached_status_service_is_single_flight() -> None:
    calls = 0
    endpoint = HealthEndpoint("API", "https://example.com")
    expected = (HealthResult(endpoint, HealthState.HEALTHY, 1, 200),)

    async def checker() -> tuple[HealthResult, ...]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return expected

    service = CachedStatusService(checker, ttl_seconds=30)
    first, second = await asyncio.gather(service.get(), service.get())
    assert first == second == expected
    assert await service.get() == expected
    assert calls == 1


@pytest.mark.asyncio
async def test_forced_refresh_coalesces_concurrent_callers() -> None:
    calls = 0
    release = asyncio.Event()
    endpoint = HealthEndpoint("API", "https://example.com")
    expected = (HealthResult(endpoint, HealthState.HEALTHY, 1, 200),)

    async def checker() -> tuple[HealthResult, ...]:
        nonlocal calls
        calls += 1
        await release.wait()
        return expected

    service = CachedStatusService(checker, ttl_seconds=30)
    first = asyncio.create_task(service.refresh())
    second = asyncio.create_task(service.refresh())
    await asyncio.sleep(0)
    release.set()

    assert await first == await second == expected
    assert calls == 1


@pytest.mark.asyncio
async def test_forced_refresh_has_a_shared_minimum_interval() -> None:
    calls = 0
    now = 100.0
    endpoint = HealthEndpoint("API", "https://example.com")

    async def checker() -> tuple[HealthResult, ...]:
        nonlocal calls
        calls += 1
        return (HealthResult(endpoint, HealthState.HEALTHY, calls, 200),)

    service = CachedStatusService(
        checker,
        ttl_seconds=300,
        clock=lambda: now,
        refresh_cooldown_seconds=5,
    )
    await service.get()
    await service.refresh()
    await service.refresh()
    assert calls == 2

    now += 5
    await service.refresh()
    assert calls == 3


class FakeInteraction:
    def __init__(self, guild_id: int | None = 42) -> None:
        self.guild_id = guild_id
        self.user = SimpleNamespace()
        self.response = SimpleNamespace(
            send_message=AsyncMock(),
            defer=AsyncMock(),
        )
        self.followup = SimpleNamespace(send=AsyncMock())


def get_callback(bot: SamsarixOperatorBot, name: str) -> Callable[[Any], Awaitable[None]]:
    group = bot.tree.get_command("samsarix")
    assert isinstance(group, app_commands.Group)
    command = group.get_command(name)
    assert isinstance(command, app_commands.Command)
    return cast(Callable[[Any], Awaitable[None]], command.callback)


@pytest.mark.asyncio
async def test_command_callbacks_cover_empty_success_and_denied_states() -> None:
    empty_bot = create_bot(make_config())
    interaction = FakeInteraction()
    await get_callback(empty_bot, "ping")(interaction)
    await get_callback(empty_bot, "about")(interaction)
    await get_callback(empty_bot, "status")(interaction)
    await get_callback(empty_bot, "check")(interaction)
    assert interaction.response.send_message.await_count == 4
    assert "No services" in interaction.response.send_message.await_args_list[-1].args[0]

    endpoint = HealthEndpoint("API", "https://example.com")
    expected = (HealthResult(endpoint, HealthState.HEALTHY, 1, 200),)

    async def checker() -> tuple[HealthResult, ...]:
        return expected

    success_bot = create_bot(make_config(endpoints=(endpoint,)), checker=checker)
    success = FakeInteraction()
    await get_callback(success_bot, "status")(success)
    success.response.defer.assert_awaited_once_with(ephemeral=True, thinking=True)
    assert success.followup.send.await_args.kwargs["ephemeral"] is True

    restricted_bot = create_bot(
        make_config(endpoints=(endpoint,), allowed_guild_ids=frozenset({99})),
        checker=checker,
    )
    denied = FakeInteraction(guild_id=42)
    await get_callback(restricted_bot, "status")(denied)
    assert "not available" in denied.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_forced_check_bypasses_a_warm_cache() -> None:
    calls = 0
    endpoint = HealthEndpoint("API", "https://example.com")

    async def checker() -> tuple[HealthResult, ...]:
        nonlocal calls
        calls += 1
        return (HealthResult(endpoint, HealthState.HEALTHY, calls, 200),)

    bot = create_bot(make_config(endpoints=(endpoint,)), checker=checker)
    await bot.status_service.get()
    interaction = FakeInteraction()
    await get_callback(bot, "check")(interaction)

    assert calls == 2
    interaction.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_status_command_fails_safely(caplog: pytest.LogCaptureFixture) -> None:
    endpoint = HealthEndpoint("API", "https://example.com")

    async def checker() -> tuple[HealthResult, ...]:
        raise RuntimeError("sensitive upstream detail")

    bot = create_bot(make_config(endpoints=(endpoint,)), checker=checker)
    interaction = FakeInteraction()
    await get_callback(bot, "status")(interaction)
    assert "failed safely" in interaction.followup.send.await_args.args[0]
    assert "Unexpected health-check failure" in caplog.text


@pytest.mark.asyncio
async def test_setup_hook_syncs_global_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = create_bot(make_config())
    sync = AsyncMock(return_value=[])
    monkeypatch.setattr(bot.tree, "sync", sync)
    await bot.setup_hook()
    sync.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_pending_alert_is_delivered_without_url_or_mentions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = HealthEndpoint("API", "https://secret.internal/health")
    result = HealthResult(endpoint, HealthState.UNHEALTHY, 10, 503, "unexpected response")
    bot = create_bot(make_config(alert_channel_id=123, endpoints=(endpoint,)))
    event = AlertEvent(AlertKind.INCIDENT, result)
    bot._pending_alerts["api"] = event
    send = AsyncMock()
    monkeypatch.setattr(bot, "get_channel", lambda channel_id: SimpleNamespace(send=send))

    await bot._deliver_pending_alerts()

    send.assert_awaited_once()
    await_call = send.await_args
    assert await_call is not None
    embed = await_call.kwargs["embed"]
    assert "secret.internal" not in str(embed.to_dict())
    assert await_call.kwargs["allowed_mentions"].everyone is False
    assert bot._pending_alerts == {}


@pytest.mark.asyncio
async def test_unavailable_alert_channel_keeps_one_bounded_pending_transition(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint = HealthEndpoint("API", "https://secret.internal/health")
    result = HealthResult(endpoint, HealthState.UNHEALTHY, 10, 503)
    bot = create_bot(make_config(alert_channel_id=123, endpoints=(endpoint,)))
    bot._pending_alerts["api"] = AlertEvent(AlertKind.INCIDENT, result)
    monkeypatch.setattr(bot, "get_channel", lambda channel_id: None)

    await bot._deliver_pending_alerts()
    await bot._deliver_pending_alerts()

    assert list(bot._pending_alerts) == ["api"]
    assert caplog.text.count("Configured alert channel is unavailable") == 1


@pytest.mark.asyncio
async def test_alert_loop_polls_once_and_queues_transition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = HealthEndpoint("API", "https://secret.internal/health")
    result = HealthResult(endpoint, HealthState.UNHEALTHY, 10, 503)
    bot = create_bot(
        make_config(
            alert_channel_id=123,
            endpoints=(endpoint,),
            failure_threshold=1,
        )
    )
    monkeypatch.setattr(bot, "wait_until_ready", AsyncMock())
    monkeypatch.setattr(bot, "is_closed", lambda: False)
    monkeypatch.setattr(bot.status_service, "refresh", AsyncMock(return_value=(result,)))

    async def deliver() -> None:
        bot._alert_stop.set()

    monkeypatch.setattr(bot, "_deliver_pending_alerts", deliver)

    await bot._alert_loop()

    assert bot._pending_alerts["api"].kind is AlertKind.INCIDENT


@pytest.mark.asyncio
async def test_alert_loop_contains_unexpected_poll_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint = HealthEndpoint("API", "https://secret.internal/health")
    bot = create_bot(make_config(alert_channel_id=123, endpoints=(endpoint,)))
    monkeypatch.setattr(bot, "wait_until_ready", AsyncMock())
    monkeypatch.setattr(bot, "is_closed", lambda: False)

    async def fail_refresh() -> tuple[HealthResult, ...]:
        bot._alert_stop.set()
        raise RuntimeError("private upstream text")

    monkeypatch.setattr(bot.status_service, "refresh", fail_refresh)

    await bot._alert_loop()

    assert "Alert polling failed safely (RuntimeError)" in caplog.text
    assert "private upstream text" not in caplog.text


@pytest.mark.asyncio
async def test_setup_hook_syncs_allowed_guilds_and_starts_alert_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint = HealthEndpoint("API", "https://example.com")
    bot = create_bot(
        make_config(
            allowed_guild_ids=frozenset({42, 7}),
            alert_channel_id=123,
            endpoints=(endpoint,),
        )
    )
    sync = AsyncMock(return_value=[])
    alert_loop = AsyncMock()
    monkeypatch.setattr(bot.tree, "sync", sync)
    monkeypatch.setattr(bot, "_alert_loop", alert_loop)

    await bot.setup_hook()
    assert [call.kwargs["guild"].id for call in sync.await_args_list] == [7, 42]
    assert bot._alert_task is not None
    await bot._alert_task
    alert_loop.assert_awaited_once()


@pytest.mark.asyncio
async def test_close_cancels_alert_task(monkeypatch: pytest.MonkeyPatch) -> None:
    bot = create_bot(make_config())

    async def wait_forever() -> None:
        await asyncio.Event().wait()

    bot._alert_task = asyncio.create_task(wait_forever())
    parent_close = AsyncMock()
    monkeypatch.setattr("discord.Client.close", parent_close)

    await bot.close()

    assert getattr(bot, "_alert_task", object()) is None
    assert bot._alert_stop.is_set()
    parent_close.assert_awaited_once()
