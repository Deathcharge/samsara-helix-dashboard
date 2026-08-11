# Releasing Samsarix Discord Operator Bot

The repository is release-candidate quality but is not yet published. Releases require an owner;
CI must not publish automatically from an arbitrary branch or pull request.

## One-time owner decisions

1. Confirm Samsarix LLC owns or has adequate rights to license the current source, including the
   historical extraction. Have counsel review `LICENSE`, `LICENSING.md`, and `TRADEMARKS.md` before
   relying on them in a dispute or commercial licensing program.
2. Keep package metadata and documentation pointed at the renamed canonical repository,
   `Deathcharge/samsarix-discord-bot`.
3. Confirm and register the `samsarix-discord-bot` distribution name before publishing.
4. Configure PyPI Trusted Publishing for this repository and a protected release environment.
5. Enable GitHub private vulnerability reporting and branch protection for `main`.

## Release gate

1. Install the bot into an owner-controlled test guild with no privileged intents or administrative
   permissions.
2. Verify `/samsarix ping`, `/samsarix about`, `/samsarix status`, and `/samsarix check` as an
   allowed user. Confirm simultaneous fresh checks coalesce.
3. Verify status denial for the wrong guild, wrong role, and direct messages.
4. Verify one healthy, redirecting, failing, timing-out, and unreachable endpoint without using
   production secrets.
5. Verify a protected HTTPS fixture receives headers without logging/rendering their values and
   accepts only its configured HTTP statuses. Confirm the same header mapping is rejected for HTTP.
6. If proactive alerts are enabled, verify failure/recovery thresholds, no repeated unchanged-state
   alert, missing channel, lost Send Messages permission, reconnect, and restart behavior.
7. Record the Discord application ID, tested commit, Python version, and operator responsible for
   the smoke test in the release notes. Never record the token.

## Build verification

From a clean checkout and isolated Python 3.11, 3.12, or 3.13 environment:

```bash
python -m pip install --requirement requirements-dev.txt
python -m ruff check .
python -m mypy
python -m pytest
python -m bandit -q -r samsarix_discord_bot
python -m pip_audit --strict --requirement requirements.txt
python -m compileall -q samsarix_discord_bot tests discord_bot_src
python -m build
python -m twine check dist/*
```

Install the wheel into a second clean environment and run:

Set `DISCORD_BOT_TOKEN` to a non-production placeholder before `check-config`. Either configure
`SAMSARIX_HEALTH_ENDPOINTS` with a controlled HTTPS fixture before the endpoint checks, or treat
exit code 4 and the `unconfigured` JSON state as the expected empty-configuration result.

```bash
samsarix-discord-bot --version
samsarix-discord-bot check-config
samsarix-discord-bot check-endpoints
samsarix-discord-bot check-endpoints --format json
python -c "import samsarix_discord_bot; print(samsarix_discord_bot.__version__)"
```

Inspect the wheel and source distribution. They must include the supported package, `py.typed`,
MPL license/notice/trademark files, and no `discord_bot_src` path.

## Version and publication

1. Update the matching version in `pyproject.toml`, `samsarix_discord_bot/__init__.py`,
   `CITATION.cff`, and `CHANGELOG.md`.
2. Commit the release, merge through protected `main`, and tag the exact merge commit as `vX.Y.Z`.
3. Build from the tag and publish with the protected PyPI environment through Trusted Publishing.
4. Create a GitHub release containing the changelog, tested platforms, wheel/sdist hashes, live
   Discord smoke-test receipt, known limitations, and upgrade notes.
5. Verify the published wheel in a clean environment before announcing availability.

Do not publish `0.1.0` until the licensing/ownership and live Discord gates are closed.
