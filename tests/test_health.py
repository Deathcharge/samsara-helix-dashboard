# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiohttp
import pytest

from samsarix_discord_bot.config import HealthEndpoint
from samsarix_discord_bot.health import (
    HealthChecker,
    HealthResult,
    HealthState,
    check_endpoint,
    overall_state,
)


@dataclass
class FakeResponse:
    status: int

    async def __aenter__(self) -> FakeResponse:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


class FakeSession:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, kwargs))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "state", "detail"),
    [
        (204, HealthState.HEALTHY, None),
        (302, HealthState.DEGRADED, "redirect not followed"),
        (503, HealthState.UNHEALTHY, "non-success response"),
    ],
)
async def test_check_endpoint_classifies_status_without_following_redirects(
    status: int, state: HealthState, detail: str | None
) -> None:
    endpoint = HealthEndpoint("API", "https://example.com/health")
    session = FakeSession(FakeResponse(status))

    result = await check_endpoint(endpoint, session, asyncio.Semaphore(1))

    assert result.state is state
    assert result.detail == detail
    assert result.status_code == status
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "detail"),
    [
        (TimeoutError(), "request timed out"),
        (aiohttp.ClientConnectionError(), "connection failed"),
    ],
)
async def test_check_endpoint_returns_safe_failure(error: Exception, detail: str) -> None:
    endpoint = HealthEndpoint("API", "https://example.com/health")
    result = await check_endpoint(endpoint, FakeSession(error), asyncio.Semaphore(1))
    assert result.state is HealthState.UNHEALTHY
    assert result.status_code is None
    assert result.detail == detail


def test_overall_state_uses_worst_result() -> None:
    endpoint = HealthEndpoint("API", "https://example.com")
    healthy = HealthResult(endpoint, HealthState.HEALTHY, 1, 200)
    degraded = HealthResult(endpoint, HealthState.DEGRADED, 1, 302)
    unhealthy = HealthResult(endpoint, HealthState.UNHEALTHY, 1, 500)
    assert overall_state(()) is HealthState.HEALTHY
    assert overall_state((healthy,)) is HealthState.HEALTHY
    assert overall_state((healthy, degraded)) is HealthState.DEGRADED
    assert overall_state((healthy, unhealthy)) is HealthState.UNHEALTHY


@pytest.mark.asyncio
async def test_health_checker_uses_real_http_interface_without_reading_body() -> None:
    async def handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    checker = HealthChecker(
        (HealthEndpoint("Local test", f"http://127.0.0.1:{port}/health"),),
        timeout_seconds=2,
        max_concurrency=1,
    )
    try:
        results = await checker.check_all()
    finally:
        server.close()
        await server.wait_closed()

    assert results[0].state is HealthState.HEALTHY
    assert results[0].status_code == 204


@pytest.mark.asyncio
async def test_health_checker_handles_empty_configuration() -> None:
    checker = HealthChecker((), timeout_seconds=2, max_concurrency=1)
    assert await checker.check_all() == ()
