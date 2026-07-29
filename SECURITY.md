# Security policy

## Supported surface

The supported surface is the `samsarix_discord_bot/` Python package and the
`samsarix-discord-bot` CLI. Version `0.1.x` is alpha software; there is not yet a published,
production-supported release.

The `discord_bot_src/` directory is an unsupported historical extraction. It is not imported by the
supported package or included in the wheel and should not be deployed. Reports about it may still
be useful for future cleanup, but they do not describe the released package unless a supported path
actually reaches the affected code.

## Trust boundaries and invariants

- Discord users, interaction fields, endpoint responses, and Discord-visible text are untrusted.
- The bot token and all process environment variables are trusted operator configuration and must
  be protected by the deployment environment.
- Discord users must never select health-check destinations, headers, credentials, executable
  code, filesystem paths, or process arguments.
- Credentials must not appear in endpoint URLs, logs, exceptions, command arguments, or Discord
  responses.
- Every outbound request must remain bounded by endpoint count, timeout, concurrency, redirects,
  response processing, and cache behavior.
- Guild and role policy must be enforced before starting a status check.
- The supported runtime must not acquire a dependency on private Samsarix or Helix-era repositories.

## Reporting

Prefer the repository's private GitHub Security Advisory interface when it is available. Otherwise,
email support@samsarix.com with `[SECURITY] Samsarix Discord Operator Bot` in the subject.

Include the affected version or commit, the smallest reproducer, impact, required preconditions,
and whether a real credential or external service was involved. Never submit live credentials,
private endpoint URLs, response bodies, or private Discord content.

Do not disclose vulnerability details in a public issue before coordination. Samsarix LLC does not
promise a bounty or fixed response deadline, but will make a good-faith effort to acknowledge a
complete report, assess it, and coordinate disclosure appropriate to the risk.
