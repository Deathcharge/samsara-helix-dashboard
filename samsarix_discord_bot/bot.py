# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Minimal Discord gateway client and slash-command experience."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress

import discord
from discord import app_commands

from . import __version__
from .alerts import AlertEvent, AlertKind, AlertTracker
from .config import BotConfig
from .health import HealthChecker, HealthResult, HealthState, overall_state

logger = logging.getLogger(__name__)
Checker = Callable[[], Awaitable[tuple[HealthResult, ...]]]
Clock = Callable[[], float]

MIN_FORCED_REFRESH_INTERVAL_SECONDS = 5.0

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


def _safe_endpoint_name(name: str) -> str:
    return discord.utils.escape_markdown(discord.utils.escape_mentions(name))


class CachedStatusService:
    """Single-flight endpoint checks with a short shared cache."""

    def __init__(
        self,
        checker: Checker,
        ttl_seconds: float,
        *,
        clock: Clock = time.monotonic,
        refresh_cooldown_seconds: float = MIN_FORCED_REFRESH_INTERVAL_SECONDS,
    ) -> None:
        self._checker = checker
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._refresh_cooldown_seconds = refresh_cooldown_seconds
        self._lock = asyncio.Lock()
        self._cached_at = 0.0
        self._cached: tuple[HealthResult, ...] | None = None
        self._generation = 0
        self._last_forced_refresh_at: float | None = None

    async def _check_and_cache(self) -> tuple[HealthResult, ...]:
        self._cached = await self._checker()
        self._cached_at = self._clock()
        self._generation += 1
        return self._cached

    async def get(self) -> tuple[HealthResult, ...]:
        now = self._clock()
        if self._cached is not None and now - self._cached_at < self._ttl_seconds:
            return self._cached

        async with self._lock:
            now = self._clock()
            if self._cached is not None and now - self._cached_at < self._ttl_seconds:
                return self._cached
            return await self._check_and_cache()

    async def refresh(self) -> tuple[HealthResult, ...]:
        """Force a check while coalescing callers that arrived during the same refresh."""
        observed_generation = self._generation
        async with self._lock:
            if self._generation != observed_generation and self._cached is not None:
                return self._cached
            now = self._clock()
            if (
                self._cached is not None
                and self._last_forced_refresh_at is not None
                and now - self._last_forced_refresh_at < self._refresh_cooldown_seconds
            ):
                return self._cached
            result = await self._check_and_cache()
            self._last_forced_refresh_at = self._clock()
            return result


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
            name=_safe_endpoint_name(result.endpoint.name),
            value=(
                f"{STATE_EMOJI[result.state]} {result.state.value} · "
                f"HTTP {status} · {result.latency_ms} ms{detail}"
            ),
            inline=False,
        )
    embed.set_footer(text="Results are cached briefly to bound outbound traffic")
    return embed


def build_alert_embed(event: AlertEvent) -> discord.Embed:
    """Render a channel-safe incident transition without destination or response data."""
    result = event.result
    recovered = event.kind is AlertKind.RECOVERY
    title_prefix = "✅ Recovered" if recovered else "❌ Incident"
    color = discord.Color.green() if recovered else STATE_COLOR[result.state]
    status = str(result.status_code) if result.status_code is not None else "no response"
    detail = f" · {result.detail}" if result.detail else ""
    embed = discord.Embed(
        title=f"{title_prefix}: {_safe_endpoint_name(result.endpoint.name)}",
        description=(
            f"{result.state.value} · HTTP {status} · {result.latency_ms} ms{detail}"
        ),
        color=color,
        timestamp=discord.utils.utcnow(),
    )
    embed.set_footer(text="Transition confirmed by consecutive health checks")
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
        self.alert_tracker = AlertTracker(
            failure_threshold=config.failure_threshold,
            recovery_threshold=config.recovery_threshold,
        )
        self._alert_task: asyncio.Task[None] | None = None
        self._alert_stop = asyncio.Event()
        self._pending_alerts: dict[str, AlertEvent] = {}
        self._alert_delivery_error_reported = False
        self._register_commands()

    async def _respond_with_status(
        self,
        interaction: discord.Interaction,
        *,
        force_refresh: bool,
    ) -> None:
        if not interaction_is_authorized(self.config, interaction):
            await interaction.response.send_message(
                "This command is not available in this guild or for your role.",
                ephemeral=True,
            )
            return
        if not self.config.endpoints:
            await interaction.response.send_message(
                "No services are configured. Set SAMSARIX_HEALTH_ENDPOINTS and restart the bot.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(ephemeral=True, thinking=True)
        try:
            results = (
                await self.status_service.refresh()
                if force_refresh
                else await self.status_service.get()
            )
        except Exception as error:
            logger.error("Unexpected health-check failure (%s)", type(error).__name__)
            await interaction.followup.send(
                "The status check failed safely. Review the bot logs and try again.",
                ephemeral=True,
            )
            return
        await interaction.followup.send(embed=build_status_embed(results), ephemeral=True)

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
            await self._respond_with_status(interaction, force_refresh=False)

        @group.command(name="check", description="Force a fresh service health check")
        async def check(interaction: discord.Interaction) -> None:
            await self._respond_with_status(interaction, force_refresh=True)

        self.tree.add_command(group)

    async def _deliver_pending_alerts(self) -> None:
        if not self._pending_alerts or self.config.alert_channel_id is None:
            return
        channel = self.get_channel(self.config.alert_channel_id)
        send = getattr(channel, "send", None)
        if not callable(send):
            if not self._alert_delivery_error_reported:
                logger.error("Configured alert channel is unavailable or not messageable")
                self._alert_delivery_error_reported = True
            return
        self._alert_delivery_error_reported = False

        for key, event in tuple(self._pending_alerts.items()):
            try:
                await send(
                    embed=build_alert_embed(event),
                    allowed_mentions=discord.AllowedMentions.none(),
                )
            except discord.HTTPException as error:
                if not self._alert_delivery_error_reported:
                    logger.error("Discord alert delivery failed (%s)", type(error).__name__)
                    self._alert_delivery_error_reported = True
                return
            if self._pending_alerts.get(key) is event:
                del self._pending_alerts[key]
            self._alert_delivery_error_reported = False

    async def _alert_loop(self) -> None:
        await self.wait_until_ready()
        while not self.is_closed() and not self._alert_stop.is_set():
            try:
                results = await self.status_service.refresh()
                for event in self.alert_tracker.update(results):
                    self._pending_alerts[event.result.endpoint.name.casefold()] = event
                await self._deliver_pending_alerts()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("Alert polling failed safely (%s)", type(error).__name__)

            try:
                await asyncio.wait_for(
                    self._alert_stop.wait(),
                    timeout=self.config.poll_interval_seconds,
                )
            except TimeoutError:
                continue

    async def close(self) -> None:
        self._alert_stop.set()
        if self._alert_task is not None:
            self._alert_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._alert_task
            self._alert_task = None
        await super().close()

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
        if self.config.alert_channel_id is not None and self._alert_task is None:
            self._alert_task = asyncio.create_task(
                self._alert_loop(),
                name="samsarix-alert-loop",
            )

    async def on_ready(self) -> None:
        logger.info(
            "Connected to Discord as %s in %d guild(s)",
            self.user,
            len(self.guilds),
        )


def create_bot(config: BotConfig, *, checker: Checker | None = None) -> SamsarixOperatorBot:
    """Create the supported bot runtime."""
    return SamsarixOperatorBot(config, checker=checker)
