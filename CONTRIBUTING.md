# Contributing

The supported product is `samsarix_discord_bot/`. Keep changes small, permission-minimal, and
independent of private Samsarix repositories and historical Helix-era services.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --requirement requirements-dev.txt
```

## Required checks

```bash
python -m ruff check .
python -m mypy
python -m pytest
python -m bandit -q -r samsarix_discord_bot
python -m pip_audit --strict --requirement requirements.txt
python -m compileall -q discord_bot_src
python -m build
python -m twine check dist/*
```

Add focused tests for configuration boundaries, endpoint behavior, authorization, and user-visible
failure states. Tests must not require Discord credentials or public network access.

## Product boundaries

- Prefer native application commands over message parsing.
- Do not request privileged intents without a documented, tested need.
- Never accept health-check destinations from Discord users.
- Keep secret headers in separately referenced environment variables; never render or log values.
- Keep endpoint count, timeouts, redirects, concurrency, and retries bounded.
- Keep background polling opt-in, pending delivery bounded, and unchanged states notification-free.
- Never log tokens, URL credentials, response bodies, or private Discord content.
- Do not add a database, AI provider, authentication system, or cloud dependency without a concrete
  primary-journey requirement.
- Do not import from `apps.backend` or another repository.

`discord_bot_src/` is a historical snapshot. A legacy extraction proposal must define the new
public API, remove private dependencies, add representative tests, and update the productization
record before the code becomes supported.

## Pull requests

Explain the user problem, security/privacy impact, verification commands, and any operator action.
Do not include credentials or generated build/coverage artifacts.

Unless explicitly agreed otherwise before acceptance, contributions are submitted under MPL-2.0,
the same license as the project. Contributors retain copyright in their work. Contributions do not
grant rights in Samsarix names or logos; see [LICENSING.md](LICENSING.md) and
[TRADEMARKS.md](TRADEMARKS.md).

General questions can go to contact@samsarix.com. Product or private security support goes to
support@samsarix.com; follow [SECURITY.md](SECURITY.md) for vulnerability reports.
