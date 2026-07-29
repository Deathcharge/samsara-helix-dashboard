# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Minimal Discord gateway client and slash-command experience."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable

import discord
from discord import app_commands

from . import __version__
from .config import BotConfig
from .health import HealthChecker, HealthResult, HealthState, overall_state

logger = logging.getLogger(__name__)
Checker = Callable[[], Awaitable[tuple[HealthResult, ...]]]

STATE_EMOJI = {
    HealthState.HEALTHY: "✅",
    HealthState.DEGRADED: "⚠️",
    HealthState.UNHEALTHY: "❌",
}
STATE_COLOR = {
    HealthState.HEALTHY: discord.Color.green(),
    HealthState.DEGRADED: discord.Color.orange(),
    HealthState.UNHEALTHY: discord.Color.red(),
}


class CachedStatusService:
    """Single-flight endpoint checks with a short shared cache."""

    def __init__(self, checker: Checker, ttl_seconds: float) -> None:
        self._checker = checker
        self._ttl_seconds = ttl_seconds
        self._lock = asyncio.Lock()
        self._cached_at = 0.0
        self._cached: tuple[HealthResult, ...] | None = None

    async def get(self) -> tuple[HealthResult, ...]:
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < self._ttl_seconds:
            return self._cached

        async with self._lock:
            now = time.monotonic()
            if self._cached is not None and now - self._cached_at < self._ttl_seconds:
                return self._cached
            self._cached = await self._checker()
            self._cached_at = time.monotonic()
            return self._cached


def interaction_is_authorized(config: BotConfig, interaction: discord.Interaction) -> bool:
    """Apply guild and optional role restrictions server-side."""
    if interaction.guild_id is None:
        return False
    if config.allowed_guild_ids and interaction.guild_id not in config.allowed_guild_ids:
        return False
    if not config.allowed_role_ids:
        return True

    user = interaction.user
    if isinstance(user, discord.Member):
        if user.guild_permissions.administrator:
            return True
        return any(role.id in config.allowed_role_ids for role in user.roles)
    return False


def build_status_embed(results: tuple[HealthResult, ...]) -> discord.Embed:
    """Render endpoint results without exposing configured URLs or response bodies."""
    state = overall_state(results)
    embed = discord.Embed(
        title=f"{STATE_EMOJI[state]} Service status: {state.value}",
        color=STATE_COLOR[state],
        timestamp=discord.utils.utcnow(),
    )
    for result in results:
        status = str(result.status_code) if result.status_code is not None else "no response"
        detail = f" — {result.detail}" if result.detail else ""
        embed.add_field(
            name=discord.utils.escape_markdown(result.endpoint.name),
            value=(
                f"{STATE_EMOJI[result.state]} {result.state.value} · "
                f"HTTP {status} · {result.latency_ms} ms{detail}"
            ),
            inline=False,
        )
    embed.set_footer(text="Results are cached briefly to bound outbound traffic")
    return embed


class SamsarixOperatorBot(discord.Client):
    """Discord client exposing the supported operator command surface."""

    def __init__(self, config: BotConfig, *, checker: Checker | None = None) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        super().__init__(
            intents=intents,
            allowed_mentions=discord.AllowedMentions.none(),
        )
        self.config = config
        self.tree = app_commands.CommandTree(self)
        self.started_at = time.monotonic()

        if checker is None:
            health_checker = HealthChecker(
                config.endpoints,
                timeout_seconds=config.request_timeout_seconds,
                max_concurrency=config.max_concurrency,
            )
            checker = health_checker.check_all
        self.status_service = CachedStatusService(checker, config.cache_ttl_seconds)
        self._register_commands()

    def _register_commands(self) -> None:
        group = app_commands.Group(
            name="samsarix",
            description="Service health and bot diagnostics",
        )

        @group.command(name="ping", description="Check whether the bot is responsive")
        async def ping(interaction: discord.Interaction) -> None:
            latency_ms = round(self.latency * 1000) if math.isfinite(self.latency) else "connecting"
            await interaction.response.send_message(
                f"✅ Samsarix operator bot is online · gateway {latency_ms} ms",
                ephemeral=True,
            )

        @group.command(name="about", description="Show version and privacy behavior")
        async def about(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(
                f"Samsarix operator bot v{__version__}. "
                "It stores no Discord message content and health checks "
                "never read response bodies.",
                ephemeral=True,
            )

        @group.command(name="status", description="Check configured service health endpoints")
        async def status(interaction: discord.Interaction) -> None:
            if not interaction_is_authorized(self.config, interaction):
                await interaction.response.send_message(
                    "This command is not available in this guild or for your role.",
                    ephemeral=True,
                )
                return
            if not self.config.endpoints:
                await interaction.response.send_message(
                    "No services are configured. Set SAMSARIX_HEALTH_ENDPOINTS "
                    "and restart the bot.",
                    ephemeral=True,
                )
                return

            await interaction.response.defer(ephemeral=True, thinking=True)
            try:
                results = await self.status_service.get()
            except Exception:
                logger.exception("Unexpected health-check failure")
                await interaction.followup.send(
                    "The status check failed safely. Review the bot logs and try again.",
                    ephemeral=True,
                )
                return
            await interaction.followup.send(embed=build_status_embed(results), ephemeral=True)

        self.tree.add_command(group)

    async def setup_hook(self) -> None:
        if self.config.allowed_guild_ids:
            for guild_id in sorted(self.config.allowed_guild_ids):
                guild = discord.Object(id=guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info("Synced %d command(s) to guild %s", len(synced), guild_id)
        else:
            synced = await self.tree.sync()
            logger.info("Synced %d global command(s)", len(synced))

    async def on_ready(self) -> None:
        logger.info(
            "Connected to Discord as %s in %d guild(s)",
            self.user,
            len(self.guilds),
        )


def create_bot(config: BotConfig, *, checker: Checker | None = None) -> SamsarixOperatorBot:
    """Create the supported bot runtime."""
    return SamsarixOperatorBot(config, checker=checker)
