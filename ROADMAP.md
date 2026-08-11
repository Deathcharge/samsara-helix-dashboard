# Samsarix Discord Operator Bot roadmap

This roadmap separates four gates: merge, release, publication, and flagship adoption. Passing one does not imply the next.

## Product boundary

Portfolio role: **integration or extension**. Keep its platform-specific packaging and release lifecycle separate. Any flagship integration should use a documented HTTP, event, or package contract with explicit auth, privacy, and failure ownership.
Planned repository identity: `Deathcharge/samsarix-discord-bot` (ready).

Current disposition: the standalone productization and canonical repository rename are merged.
Competitive operator workflows are in active release-candidate hardening; release and flagship
adoption remain separate decisions.

## Stabilize the productized default

- Keep the default branch buildable from a clean checkout and preserve exact-head CI evidence.
- Keep Samsarix LLC branding, package identity, license metadata, and compatibility aliases internally consistent.
- Preserve the pre-productization default under a rollback ref before merging; do not delete legacy history.
- Review priority: keep direct HTTP and Discord runtime dependencies explicit and audited.
- Review priority: green Python 3.11–3.13 CI and clean-wheel verification.
- Review priority: live minimum-permission Discord smoke.
- Review priority: approve MPL.
- Review priority: publish 0.1.0 prerelease.

## Release candidate

- Test the exact distributable on its target platform, including failure and upgrade paths.
- Review permissions, data retention, privacy copy, signing, and store or platform ownership.
- Release a prerelease to a bounded pilot before broad distribution.

Current hardening backlog:

- No live Discord installation, command sync, permissions, or reconnect test was observed.
- No release/publish automation or versioned release exists.
- Authenticated headers, exact expected statuses, forced-fresh checks, JSON preflight output, and
  thresholded incident/recovery notifications are implemented for the competitive operator loop.
- HTTP status remains the deliberately bounded health definition; response-body assertions are not
  implemented.
- Repository size and legacy directories can still mislead users and security tools despite artifact exclusion.
- MPL relicensing and ownership of the extracted legacy code require explicit approval.

Highest-value next validation:

- Run a minimum-permission Discord pilot covering command sync, reconnect, missing-channel
  permissions, two-failure incident delivery, two-success recovery, and rate-limit behavior.
- Decide between scheduled-maintenance suppression and alert reminders using pilot feedback.
- Add a process-supervisor or container example once an owner chooses the supported deployment path.
- Publish an authenticated prerelease only after the live Discord and ownership gates close.

## Samsarix adoption

- Define a public API, event, schema, artifact, or deployment contract before connecting to Samsarix Unified.
- Add a consumer-owned contract fixture covering authentication, privacy, limits, errors, and version compatibility.
- Make one implementation canonical; remove or freeze duplicate behavior only after parity and rollback are proven.
- Record an owner, support level, compatibility window, and measurable adoption signal.

## Completion evidence

A milestone is complete only when its exact commit, commands and results, artifact digest, consumer or deployment, and rollback path are recorded in a pull request or release record. README claims must not exceed that evidence.
