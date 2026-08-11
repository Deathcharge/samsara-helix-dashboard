# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from samsarix_discord_bot.alerts import AlertKind, AlertTracker
from samsarix_discord_bot.config import HealthEndpoint
from samsarix_discord_bot.health import HealthResult, HealthState


def result(state: HealthState, status: int | None = None) -> HealthResult:
    return HealthResult(
        HealthEndpoint("API", "https://private.example/health"),
        state,
        12,
        status,
    )


def test_tracker_debounces_incidents_and_recoveries() -> None:
    tracker = AlertTracker(failure_threshold=2, recovery_threshold=2)

    assert tracker.update((result(HealthState.HEALTHY, 200),)) == ()
    assert tracker.update((result(HealthState.UNHEALTHY, 503),)) == ()
    incident = tracker.update((result(HealthState.UNHEALTHY, 503),))
    assert [event.kind for event in incident] == [AlertKind.INCIDENT]

    assert tracker.update((result(HealthState.UNHEALTHY, 503),)) == ()
    assert tracker.update((result(HealthState.HEALTHY, 200),)) == ()
    recovery = tracker.update((result(HealthState.HEALTHY, 200),))
    assert [event.kind for event in recovery] == [AlertKind.RECOVERY]


def test_tracker_treats_degraded_as_a_failure_and_keeps_endpoints_independent() -> None:
    tracker = AlertTracker(failure_threshold=1, recovery_threshold=1)
    api = result(HealthState.DEGRADED, 302)
    worker_endpoint = HealthEndpoint("Worker", "https://private.example/worker")
    worker = HealthResult(worker_endpoint, HealthState.HEALTHY, 8, 204)

    events = tracker.update((api, worker))

    assert [(event.kind, event.result.endpoint.name) for event in events] == [
        (AlertKind.INCIDENT, "API")
    ]
