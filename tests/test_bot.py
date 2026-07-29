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

from samsarix_discord_bot.bot import (
    CachedStatusService,
    SamsarixOperatorBot,
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
    assert {command.name for command in group.commands} == {"about", "ping", "status"}
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
    assert calls == 1


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
    assert interaction.response.send_message.await_count == 3
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
