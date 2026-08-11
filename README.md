# Samsarix Discord Operator Bot

Samsarix Discord Operator Bot gives self-hosting teams a private Discord operating loop for service
health. Configure up to 20 HTTP endpoints, run one small Python process, query current state
ephemerally, and optionally post low-noise incident and recovery transitions without exposing
endpoint URLs, request credentials, or response bodies.

The supported runtime is the `samsarix_discord_bot` package. It stands alone and has no runtime
dependency on another Samsarix or Helix-era checkout. The older `discord_bot_src` tree is retained
as an unshipped historical extraction and is not required at runtime.

Samsarix LLC is the project steward. The canonical repository is
`Deathcharge/samsarix-discord-bot`; package, CLI, environment, Discord command, company, and product
branding use Samsarix.

## Status

Version `0.1.0` is alpha software and a release candidate for local and self-hosted evaluation. The
core journey, tests, package build, CI, security checks, and license policy are implemented. A live
owner-controlled Discord smoke test and legal confirmation of code ownership remain publication
gates; see [Releasing](docs/RELEASING.md).

## What it does

- Registers `/samsarix ping`, `/samsarix about`, `/samsarix status`, and `/samsarix check` as native
  Discord slash commands.
- Provides `check-config` and token-independent `check-endpoints` preflight commands, including a
  stable secret-safe JSON format for automation.
- Checks only operator-configured HTTP or HTTPS endpoints.
- Supports per-endpoint expected status codes and request headers loaded from separate secret
  environment variables.
- Runs at most 20 checks with configurable timeouts and concurrency.
- Never follows redirects and never reads response bodies.
- Caches results briefly so concurrent Discord requests do not amplify outbound traffic.
- Optionally polls in the background and posts one incident/recovery transition only after
  configurable consecutive results.
- Returns status responses ephemerally and never requests the privileged message-content intent.
- Optionally restricts commands to specific guilds and `/samsarix status` to specific roles.

It deliberately does not include LLM calls, arbitrary code execution, moderation, account linking,
voice features, a database, or private Samsarix/Helix-era service dependencies.

## Quick start

Prerequisites:

- Python 3.11, 3.12, or 3.13
- A Discord application with a bot token
- A Discord server where you can install the application

Create a virtual environment and install the package:

```bash
python -m venv .venv
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install .
```

Or on PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install .
```

Set configuration. This example uses one public placeholder; replace it with your own health URL.

```bash
export DISCORD_BOT_TOKEN="replace-me"
export SAMSARIX_ALLOWED_GUILD_IDS="123456789012345678"
export SAMSARIX_HEALTH_ENDPOINTS='[{"name":"API","url":"https://example.com/health"}]'
```

PowerShell equivalent:

```powershell
$env:DISCORD_BOT_TOKEN = "replace-me"
$env:SAMSARIX_ALLOWED_GUILD_IDS = "123456789012345678"
$env:SAMSARIX_HEALTH_ENDPOINTS = '[{"name":"API","url":"https://example.com/health"}]'
```

Validate without connecting to Discord or making health requests:

```bash
samsarix-discord-bot check-config
```

Verify configured endpoints without connecting to Discord. This prints names and status metadata,
never URLs or response bodies:

```bash
samsarix-discord-bot check-endpoints
```

Use the stable JSON schema in CI or deployment gates. Exit code `0` means every endpoint is healthy;
`4` means unconfigured, degraded, or unhealthy:

```bash
samsarix-discord-bot check-endpoints --format json
```

Start the bot:

```bash
samsarix-discord-bot run
```

In Discord, run `/samsarix ping`, then `/samsarix status`. Use `/samsarix check` when you explicitly
need a fresh result rather than the short shared cache. Overlapping calls coalesce, and a shared
five-second minimum interval also collapses immediate sequential forced checks. Guild-scoped
commands appear quickly when `SAMSARIX_ALLOWED_GUILD_IDS` is set; global command propagation can
take longer.

See [Getting Started](docs/GETTING_STARTED.md) for the Discord Developer Portal and installation
steps.

## Configuration

The process reads environment variables directly; it does not automatically load `.env` files.
Use [.env.example](.env.example) as a deployment template.

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `DISCORD_BOT_TOKEN` | Yes | — | Discord bot credential; never logged. |
| `SAMSARIX_HEALTH_ENDPOINTS` | No | `[]` | JSON array of endpoint objects, maximum 20. |
| `SAMSARIX_ALLOWED_GUILD_IDS` | No | all installed guilds | Comma-separated guild IDs; also enables fast guild-scoped sync. |
| `SAMSARIX_ALLOWED_ROLE_IDS` | No | all guild members | Roles allowed to run status/check; administrators are always allowed. |
| `SAMSARIX_REQUEST_TIMEOUT_SECONDS` | No | `5` | Total request timeout, from 1 through 30 seconds. |
| `SAMSARIX_MAX_CONCURRENCY` | No | `5` | Concurrent health requests, from 1 through 20. |
| `SAMSARIX_CACHE_TTL_SECONDS` | No | `15` | Shared result cache, from 5 through 300 seconds. |
| `SAMSARIX_ALERT_CHANNEL_ID` | No | disabled | Channel for proactive incident/recovery embeds. Requires at least one endpoint. |
| `SAMSARIX_POLL_INTERVAL_SECONDS` | No | `60` | Alert polling interval, from 30 through 3600 seconds. |
| `SAMSARIX_FAILURE_THRESHOLD` | No | `2` | Consecutive nonhealthy checks before an incident, from 1 through 10. |
| `SAMSARIX_RECOVERY_THRESHOLD` | No | `2` | Consecutive healthy checks before recovery, from 1 through 10. |
| `SAMSARIX_LOG_LEVEL` | No | `INFO` | Standard Python log level. |

An endpoint may add `expected_statuses` and `headers_env`. Header values stay in a separate variable
so the endpoint list remains safe to inspect:

```bash
export SAMSARIX_HEALTH_ENDPOINTS='[{"name":"Private API","url":"https://api.example.com/ready","expected_statuses":[200,204],"headers_env":"SAMSARIX_ENDPOINT_HEADERS_API"}]'
export SAMSARIX_ENDPOINT_HEADERS_API='{"Authorization":"Bearer replace-me"}'
```

Header variable names must start with `SAMSARIX_ENDPOINT_HEADERS_`. Transport-controlled headers
such as `Host`, `Content-Length`, and `Transfer-Encoding` are rejected. Header-bearing endpoints
must use HTTPS, and secret values are never included in configuration summaries or status output.

The complete configuration and Python API contracts are in
[API Reference](docs/API_REFERENCE.md).

## Discord installation

In the Discord Developer Portal:

1. Create an application and bot, then copy the bot token into your secret manager.
2. Leave privileged gateway intents disabled; this bot uses only the standard guild intent.
3. Install the app to your server with the `bot` and `applications.commands` scopes.
4. Grant no administrative Discord permissions. On-demand commands need no privileged gateway
   intent. If proactive alerting is enabled, grant only View Channel, Send Messages, and Embed Links
   in the configured alert channel.

Discord documents application commands as the primary app invocation model, and privileged intents
must be explicitly enabled. The release candidate follows those defaults:
[Interactions & Commands](https://docs.discord.com/developers/platform/interactions) and
[Gateway Intents](https://docs.discord.com/developers/events/gateway#gateway-intents).

## Development and verification

```bash
python -m pip install --requirement requirements-dev.txt
python -m ruff check samsarix_discord_bot tests
python -m mypy
python -m pytest
python -m bandit -q -r samsarix_discord_bot
python -m pip_audit --strict --requirement requirements.txt
python -m compileall -q discord_bot_src
python -m build
python -m twine check dist/*
```

CI runs those checks on Python 3.11–3.13, audits the pinned runtime dependency, and verifies that
the wheel contains the typed supported package and legal notices but not the historical extraction.

## Architecture

```text
Discord slash command
        |
        v
SamsarixOperatorBot ---- server-side guild/role check
        |                     ^
        |                     | optional thresholded channel alerts
        v
CachedStatusService ---- coalesced fresh checks + short cache
        |
        v
HealthChecker ---- timeout + concurrency cap + no redirects/body reads
        |
        v
Operator-configured HTTP(S) endpoints
```

`config.py` owns validation, `health.py` owns bounded network I/O, `alerts.py` owns transition
debouncing, `bot.py` owns Discord behavior, and `cli.py` owns schemas, exit codes, and startup. No
persistent store is needed.

## Security, privacy, reliability, and cost

- Treat the Discord token as a secret. The CLI reports only whether it exists.
- Endpoint URLs are trusted operator configuration and may intentionally address private services;
  Discord users cannot supply or change them.
- Status output includes names, state, HTTP status, and latency only. It omits configured URLs and
  response content.
- Authentication headers are loaded from separately named secret variables and never rendered in
  summaries, JSON output, Discord messages, or expected validation errors.
- Redirects are reported as degraded rather than followed, preventing destination changes during a
  check.
- Empty configuration produces an actionable Discord message instead of a startup crash.
- On-demand traffic is coalesced, cached, and forced checks share a five-second minimum interval.
  Optional polling is bounded by
  `20 endpoints × 3600 / interval` requests per hour; there are no metered AI or data-storage costs.
- The bot stores no message content, member profile, health history, or telemetry.

## Known limitations

- Health is intentionally HTTP-status based; response-body assertions are not supported.
- Alert thresholds and pending delivery state are in memory. A process restart starts a new
  observation window and does not reconstruct prior incidents.
- Configuration changes require a process restart.
- No container or hosted deployment is supplied; run it under the process supervisor you already
  trust.
- Discord delivery has not been validated with owner credentials in this environment.
- Python 3.14 is intentionally rejected until the runtime dependency stack is validated there.

## Historical source snapshot

`discord_bot_src/` contains the earlier 10,000-line Helix-integrated bot extraction. It depends on
unreleased `apps.backend` modules, contains many unverified command surfaces, and is excluded from
the wheel and supported startup path. It is preserved for deliberate future extraction work, not
presented as functional standalone code. See [Legacy Snapshot](discord_bot_README.md).

## Contributing and product decisions

See [CONTRIBUTING.md](CONTRIBUTING.md) for the verified workflow and
[Productization Record](docs/PRODUCTIZATION.md) for baseline evidence, scope decisions, acceptance
criteria, and deferred work. Support paths are in [SUPPORT.md](SUPPORT.md); suspected
vulnerabilities must follow [SECURITY.md](SECURITY.md).

The current product comparison and deliberately narrow competitive wedge are documented in
[Competitive Research](docs/COMPETITIVE_RESEARCH.md).

## License

The current tree is licensed under the unmodified
[Mozilla Public License 2.0](LICENSE), SPDX `MPL-2.0`. This file-level copyleft permits commercial
and proprietary larger works while requiring distributed modifications to covered source files to
remain available under MPL-2.0. See [LICENSING.md](LICENSING.md) and [NOTICE](NOTICE).

The source license does not grant rights to Samsarix names or logos. See
[TRADEMARKS.md](TRADEMARKS.md). General questions go to
[contact@samsarix.com](mailto:contact@samsarix.com); product support and private security reports go
to [support@samsarix.com](mailto:support@samsarix.com).

Copyright © 2026 Samsarix LLC.
