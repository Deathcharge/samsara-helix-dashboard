# Releasing Samsarix Discord Operator Bot

The repository is release-candidate quality but is not yet published. Releases require an owner;
CI must not publish automatically from an arbitrary branch or pull request.

## One-time owner decisions

1. Confirm Samsarix LLC owns or has adequate rights to license the current source, including the
   historical extraction. Have counsel review `LICENSE`, `LICENSING.md`, and `TRADEMARKS.md` before
   relying on them in a dispute or commercial licensing program.
2. Decide whether to retain the historical `helix-discord-bot` GitHub slug. Product, package, CLI,
   and command branding do not depend on changing that stable repository address.
3. Confirm and register the `samsarix-discord-bot` distribution name before publishing.
4. Configure PyPI Trusted Publishing for this repository and a protected release environment.
5. Enable GitHub private vulnerability reporting and branch protection for `main`.

## Release gate

1. Install the bot into an owner-controlled test guild with no privileged intents or administrative
   permissions.
2. Verify `/samsarix ping`, `/samsarix about`, and `/samsarix status` as an allowed user.
3. Verify status denial for the wrong guild, wrong role, and direct messages.
4. Verify one healthy, redirecting, failing, timing-out, and unreachable endpoint without using
   production secrets.
5. Record the Discord application ID, tested commit, Python version, and operator responsible for
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

```bash
samsarix-discord-bot --version
samsarix-discord-bot check-config
samsarix-discord-bot check-endpoints
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
