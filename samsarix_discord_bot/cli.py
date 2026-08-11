# Copyright (c) 2026 Samsarix LLC
# SPDX-License-Identifier: MPL-2.0

"""Command-line entry point for configuration checks and the bot runtime."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Sequence

import discord

from . import __version__
from .bot import create_bot
from .config import ConfigError, load_config
from .health import HealthChecker, HealthResult, HealthState, overall_state


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="samsarix-discord-bot",
        description="Run or validate the standalone Samsarix operator bot.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-config", help="Validate environment variables without connecting")
    endpoint_parser = subcommands.add_parser(
        "check-endpoints",
        help="Check configured endpoints without connecting to Discord",
    )
    endpoint_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        dest="output_format",
        help="Output human-readable text or a stable JSON document",
    )
    subcommands.add_parser("run", help="Connect to Discord and serve slash commands")
    return parser


def _endpoint_payload(
    results: tuple[HealthResult, ...],
    *,
    overall: str,
    error: str | None = None,
) -> dict[str, object]:
    serialized_results: list[dict[str, object]] = []
    for result in results:
        serialized_results.append(
            {
                "name": result.endpoint.name,
                "state": result.state.value,
                "status_code": result.status_code,
                "latency_ms": result.latency_ms,
                "detail": result.detail,
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "overall": overall,
        "results": serialized_results,
    }
    if error is not None:
        payload["error"] = error
    return payload


def _print_json_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, separators=(",", ":"), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(require_token=args.command != "check-endpoints")
    except ConfigError as error:
        if args.command == "check-endpoints" and args.output_format == "json":
            _print_json_payload(
                _endpoint_payload((), overall="configuration_error", error=str(error))
            )
            return 2
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.command == "check-config":
        print(f"Configuration valid: {config.summary()}")
        if not config.endpoints:
            print(
                "Notice: no health endpoints are configured; "
                "/samsarix status will explain how to add one."
            )
        return 0

    if args.command == "check-endpoints":
        if not config.endpoints:
            if args.output_format == "json":
                _print_json_payload(
                    _endpoint_payload(
                        (),
                        overall="unconfigured",
                        error="SAMSARIX_HEALTH_ENDPOINTS is empty",
                    )
                )
                return 4
            print("Endpoint check failed: SAMSARIX_HEALTH_ENDPOINTS is empty", file=sys.stderr)
            return 4
        checker = HealthChecker(
            config.endpoints,
            timeout_seconds=config.request_timeout_seconds,
            max_concurrency=config.max_concurrency,
        )
        try:
            results = asyncio.run(checker.check_all())
        except Exception:
            if args.output_format == "json":
                _print_json_payload(
                    _endpoint_payload(
                        (),
                        overall="error",
                        error="endpoint check failed unexpectedly",
                    )
                )
                return 5
            print("Endpoint check failed unexpectedly", file=sys.stderr)
            return 5
        aggregate = overall_state(results)
        if args.output_format == "json":
            _print_json_payload(_endpoint_payload(results, overall=aggregate.value))
            return 0 if aggregate is HealthState.HEALTHY else 4
        for result in results:
            status = str(result.status_code) if result.status_code is not None else "no response"
            detail = f" · {result.detail}" if result.detail else ""
            print(
                f"{result.endpoint.name}: {result.state.value} · "
                f"HTTP {status} · {result.latency_ms} ms{detail}"
            )
        return 0 if aggregate is HealthState.HEALTHY else 4

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger(__name__).info("Starting with %s", config.summary())
    bot = create_bot(config)
    try:
        bot.run(config.token or "", log_handler=None, reconnect=True)
    except discord.LoginFailure:
        logging.getLogger(__name__).error("Discord rejected DISCORD_BOT_TOKEN")
        return 3
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Shutdown requested")
    return 0
