# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Bounded, body-free HTTP health checks for operator-configured endpoints."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, cast

import aiohttp

from .config import HealthEndpoint


class HealthState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthResult:
    endpoint: HealthEndpoint
    state: HealthState
    latency_ms: int
    status_code: int | None = None
    detail: str | None = None


class ResponseLike(Protocol):
    status: int

    async def __aenter__(self) -> ResponseLike: ...

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None: ...


class SessionLike(Protocol):
    def get(self, url: str, **kwargs: object) -> ResponseLike: ...


def overall_state(results: tuple[HealthResult, ...]) -> HealthState:
    """Return the worst state in a collection of endpoint results."""
    if any(result.state is HealthState.UNHEALTHY for result in results):
        return HealthState.UNHEALTHY
    if any(result.state is HealthState.DEGRADED for result in results):
        return HealthState.DEGRADED
    return HealthState.HEALTHY


async def check_endpoint(
    endpoint: HealthEndpoint,
    session: SessionLike,
    semaphore: asyncio.Semaphore,
) -> HealthResult:
    """Check one endpoint without following redirects or reading its body."""
    started = time.perf_counter()
    try:
        async with (
            semaphore,
            session.get(
                endpoint.url,
                allow_redirects=False,
                headers={
                    "User-Agent": "samsarix-discord-bot/0.1",
                    **dict(endpoint.headers),
                },
            ) as response,
        ):
            latency_ms = round((time.perf_counter() - started) * 1000)
            if response.status in endpoint.expected_statuses:
                state = HealthState.HEALTHY
                detail = None
            elif 300 <= response.status < 400:
                state = HealthState.DEGRADED
                detail = "redirect not followed"
            else:
                state = HealthState.UNHEALTHY
                detail = "unexpected response"
            return HealthResult(
                endpoint=endpoint,
                state=state,
                latency_ms=latency_ms,
                status_code=response.status,
                detail=detail,
            )
    except TimeoutError:
        return HealthResult(
            endpoint=endpoint,
            state=HealthState.UNHEALTHY,
            latency_ms=round((time.perf_counter() - started) * 1000),
            detail="request timed out",
        )
    except aiohttp.ClientError:
        return HealthResult(
            endpoint=endpoint,
            state=HealthState.UNHEALTHY,
            latency_ms=round((time.perf_counter() - started) * 1000),
            detail="connection failed",
        )


class HealthChecker:
    """Check a bounded endpoint set with a shared timeout and connection pool."""

    def __init__(
        self,
        endpoints: tuple[HealthEndpoint, ...],
        *,
        timeout_seconds: float,
        max_concurrency: int,
    ) -> None:
        self.endpoints = endpoints
        self.timeout_seconds = timeout_seconds
        self.max_concurrency = max_concurrency

    async def check_all(self) -> tuple[HealthResult, ...]:
        if not self.endpoints:
            return ()

        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        connector = aiohttp.TCPConnector(limit=self.max_concurrency)
        semaphore = asyncio.Semaphore(self.max_concurrency)
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            trust_env=False,
            raise_for_status=False,
        ) as session:
            results = await asyncio.gather(
                *(
                    check_endpoint(endpoint, cast(SessionLike, session), semaphore)
                    for endpoint in self.endpoints
                )
            )
        return tuple(results)
