# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Low-noise in-memory incident and recovery transition tracking."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .health import HealthResult, HealthState


class AlertKind(StrEnum):
    INCIDENT = "incident"
    RECOVERY = "recovery"


@dataclass(frozen=True, slots=True)
class AlertEvent:
    kind: AlertKind
    result: HealthResult


@dataclass(slots=True)
class _EndpointState:
    incident_open: bool = False
    consecutive_failures: int = 0
    consecutive_successes: int = 0


class AlertTracker:
    """Emit one incident/recovery event only after configured consecutive results."""

    def __init__(self, *, failure_threshold: int, recovery_threshold: int) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        self._states: dict[str, _EndpointState] = {}

    def update(self, results: tuple[HealthResult, ...]) -> tuple[AlertEvent, ...]:
        events: list[AlertEvent] = []
        for result in results:
            state = self._states.setdefault(result.endpoint.name.casefold(), _EndpointState())
            if result.state is HealthState.HEALTHY:
                state.consecutive_failures = 0
                state.consecutive_successes += 1
                if state.incident_open and state.consecutive_successes >= self.recovery_threshold:
                    state.incident_open = False
                    state.consecutive_successes = 0
                    events.append(AlertEvent(AlertKind.RECOVERY, result))
                continue

            state.consecutive_successes = 0
            state.consecutive_failures += 1
            if not state.incident_open and state.consecutive_failures >= self.failure_threshold:
                state.incident_open = True
                state.consecutive_failures = 0
                events.append(AlertEvent(AlertKind.INCIDENT, result))
        return tuple(events)
