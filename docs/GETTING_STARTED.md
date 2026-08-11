# Getting Started

This guide takes a new operator from a clone to a fresh `/samsarix check` response and an optional
thresholded incident/recovery alert.

## 1. Create the Discord application

1. Open the [Discord Developer Portal](https://discord.com/developers/applications).
2. Create an application, open **Bot**, and create its bot user.
3. Reset/copy the token and store it as `DISCORD_BOT_TOKEN`. Never paste it into source, an issue,
   or a support message.
4. Keep Message Content, Server Members, and Presence privileged intents disabled.
5. Under installation settings, install the app to your test server with `bot` and
   `applications.commands` scopes. Do not grant Administrator.

If proactive alerts will be enabled, grant the bot View Channel, Send Messages, and Embed Links in
one dedicated operator channel. On-demand commands do not require privileged gateway intents.

For the fastest development sync, copy the server ID and configure
`SAMSARIX_ALLOWED_GUILD_IDS`. Discord requires Developer Mode to expose **Copy Server ID**.

## 2. Install locally

From the repository root:

```bash
python -m venv .venv
```

Use Python 3.11, 3.12, or 3.13. Package metadata intentionally rejects unverified Python 3.14.

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install .
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install .
```

## 3. Configure

The app reads the process environment; `.env` files are not loaded automatically. The following
PowerShell example monitors two services:

```powershell
$env:DISCORD_BOT_TOKEN = "your-token"
$env:SAMSARIX_ALLOWED_GUILD_IDS = "123456789012345678"
$env:SAMSARIX_ALLOWED_ROLE_IDS = "234567890123456789"
$env:SAMSARIX_HEALTH_ENDPOINTS = '[{"name":"API","url":"https://api.example.com/health","expected_statuses":[200,204]},{"name":"Worker","url":"https://worker.internal:8443/ready","headers_env":"SAMSARIX_ENDPOINT_HEADERS_WORKER"}]'
$env:SAMSARIX_ENDPOINT_HEADERS_WORKER = '{"Authorization":"Bearer your-private-readiness-token"}'
```

Header-bearing endpoints must use HTTPS. Plain HTTP remains available only for endpoints that do
not send configured credentials.

Guild administrators can always use `/samsarix status`. If no role IDs are configured, every member
of an allowed/installed guild can use it. Responses are ephemeral.

Validate syntax and bounds without connecting or sending requests:

```bash
samsarix-discord-bot check-config
```

The command exits `0` when configuration is valid and `2` when it is invalid. It never prints the
token or endpoint URLs.

Before connecting to Discord, check endpoint reachability:

```bash
samsarix-discord-bot check-endpoints
```

This command does not require `DISCORD_BOT_TOKEN`. It exits `0` only when every endpoint is healthy,
`4` when configuration is empty or any endpoint is degraded/unhealthy, and `5` after an unexpected
safe failure. Output contains endpoint names, states, status codes, latency, and generic failure
details—never URLs or response bodies.

For CI and deployment gates, request the stable URL- and credential-free JSON schema:

```bash
samsarix-discord-bot check-endpoints --format json
```

To enable proactive alerts, copy the dedicated channel ID and add:

```powershell
$env:SAMSARIX_ALERT_CHANNEL_ID = "345678901234567890"
$env:SAMSARIX_POLL_INTERVAL_SECONDS = "60"
$env:SAMSARIX_FAILURE_THRESHOLD = "2"
$env:SAMSARIX_RECOVERY_THRESHOLD = "2"
```

Alerting is opt-in. A default incident requires two consecutive degraded/unhealthy checks; recovery
requires two consecutive healthy checks. Repeated identical states do not send another message.

## 4. Run and verify

```bash
samsarix-discord-bot run
```

Wait for the log line confirming command sync, then in Discord:

1. Run `/samsarix ping` and confirm an online response.
2. Run `/samsarix about` and confirm version `0.1.0`.
3. Run `/samsarix status` and confirm every configured endpoint has a state, HTTP status, and latency.
4. Run `/samsarix check` and confirm it performs a fresh check rather than returning the warm
   ordinary cache. Immediate repeated forced checks share the first result for five seconds.
5. If alerting is enabled, use a nonproduction fixture to verify the configured number of failures
   produces one incident and the configured number of successes produces one recovery.
6. Remove the bot's Send Messages permission temporarily and confirm delivery fails safely without
   exposing a token, URL, header value, or response body in logs.

## Common failures

### `DISCORD_BOT_TOKEN is required`

Set the variable in the same process environment/session that launches the bot. `.env.example` is
a template only.

### Discord rejects the token

Reset the token in the Developer Portal, update your secret manager, and restart. Do not log or
share the rejected value.

### Commands do not appear

Set your test server ID in `SAMSARIX_ALLOWED_GUILD_IDS` and restart for guild-scoped sync. Confirm the
application was installed with `applications.commands`.

### Status says no services are configured

Set `SAMSARIX_HEALTH_ENDPOINTS` to a JSON array. Run `check-config` before restarting.

### A service reports a redirect

Redirects are deliberately not followed. Configure the final canonical health URL.

### A valid service reports an unexpected response

Add its accepted status to that endpoint's `expected_statuses` list. Do not broadly accept failure
statuses merely to make a check green.

### Proactive alerts do not arrive

Confirm `SAMSARIX_ALERT_CHANNEL_ID` identifies a channel visible to the installed bot and that the
bot has View Channel, Send Messages, and Embed Links there. Run `/samsarix check` to separate
endpoint health from channel-delivery configuration. Threshold counters reset after a restart.

### Timeouts or connection failures

Check DNS/firewall access from the machine running the bot. Increase
`SAMSARIX_REQUEST_TIMEOUT_SECONDS` only within the supported 1–30 second range.

## Upgrade and rollback

Install a reviewed tag in a fresh virtual environment, rerun `check-config`, and restart the process.
Rollback by restoring the prior tag/environment; there is no database or migration state.
