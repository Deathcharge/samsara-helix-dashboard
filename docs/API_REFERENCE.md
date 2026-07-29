# API and Configuration Reference

The supported public surface is intentionally small. Everything under `discord_bot_src/` is a
historical, unshipped snapshot and is not part of this API.

## CLI

### `samsarix-discord-bot --version`

Print the package version and exit `0`.

### `samsarix-discord-bot check-config`

Validate all environment variables without connecting to Discord or health endpoints.

- Exit `0`: valid.
- Exit `2`: missing or invalid configuration.
- Output redacts the Discord token and does not list endpoint URLs.

Equivalent module form:

```bash
python -m samsarix_discord_bot check-config
```

### `samsarix-discord-bot check-endpoints`

Validate endpoint reachability without connecting to Discord. The Discord token is optional for
this subcommand.

- Exit `0`: every configured endpoint returned HTTP 2xx.
- Exit `2`: another environment variable is invalid.
- Exit `4`: no endpoints are configured or at least one is degraded/unhealthy.
- Exit `5`: an unexpected check failure was contained.
- Output contains names, state, HTTP status, latency, and generic detail only; it never includes
  configured URLs or response bodies.

Equivalent module form:

```bash
python -m samsarix_discord_bot check-endpoints
```

### `samsarix-discord-bot run`

Validate configuration, sync commands, and connect to the Discord Gateway.

- Exit `2`: configuration error.
- Exit `3`: Discord rejected the bot token.
- `Ctrl+C`: graceful client shutdown through `discord.py`.

## Environment variables

### `DISCORD_BOT_TOKEN`

Required secret string. Whitespace-only values are rejected. It is passed only to `discord.py` and
never included in the configuration summary.

### `SAMSARIX_HEALTH_ENDPOINTS`

Optional JSON array with at most 20 entries:

```json
[
  {"name": "API", "url": "https://api.example.com/health"},
  {"name": "Worker", "url": "http://worker.internal:8080/ready"}
]
```

Rules:

- `name` is required, unique case-insensitively, and at most 50 characters.
- `name` cannot contain control or non-printable characters.
- `url` is an absolute HTTP(S) URL at most 2048 characters.
- URL credentials and fragments are rejected.
- Raw whitespace and control characters in URLs are rejected.
- Queries are allowed.
- Redirects are not followed.
- Response bodies are not read.

URLs are trusted operator input, not Discord-user input. Private destinations are allowed because
monitoring private services is a primary use case.

### Access controls

- `SAMSARIX_ALLOWED_GUILD_IDS`: optional comma-separated positive Discord guild IDs. When present,
  commands sync only to those guilds and status requests from other guilds are rejected.
- `SAMSARIX_ALLOWED_ROLE_IDS`: optional comma-separated positive role IDs. When present, `/samsarix
  status` requires one listed role or guild Administrator permission.

DM status requests are rejected server-side.

### Resource controls

- `SAMSARIX_REQUEST_TIMEOUT_SECONDS`: finite number from 1 through 30; default `5`.
- `SAMSARIX_MAX_CONCURRENCY`: integer from 1 through 20; default `5`.
- `SAMSARIX_CACHE_TTL_SECONDS`: finite number from 5 through 300; default `15`.
- `SAMSARIX_LOG_LEVEL`: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`; default `INFO`.

## Discord commands

All replies are ephemeral and use no message-content intent.

| Command | Behavior |
| --- | --- |
| `/samsarix ping` | Reports bot availability and Gateway latency. |
| `/samsarix about` | Reports version and privacy behavior. |
| `/samsarix status` | Applies guild/role checks, then reports cached endpoint health. |

Health state mapping:

- `healthy`: HTTP 2xx.
- `degraded`: HTTP 3xx; redirect is not followed.
- `unhealthy`: HTTP 4xx/5xx, timeout, or connection failure.

## Python API

Stable for `0.1.x`:

```python
from samsarix_discord_bot import BotConfig, ConfigError, HealthEndpoint, load_config
from samsarix_discord_bot.bot import create_bot
from samsarix_discord_bot.health import HealthChecker, HealthResult, HealthState
```

`load_config()` validates the current environment. Pass a string mapping in tests. `create_bot()`
returns the configured Discord client without connecting. `HealthChecker.check_all()` returns an
ordered tuple matching configured endpoint order.

Internal caching, rendering helpers, and legacy modules may change within `0.1.x`.
