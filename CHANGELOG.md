# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Standalone `samsarix_discord_bot` package with `/samsarix ping`, `/samsarix about`, and `/samsarix status`.
- Strict token, endpoint, guild, role, timeout, concurrency, cache, and log-level validation.
- Bounded HTTP checks that do not follow redirects or read response bodies.
- No-network configuration validation CLI and console entry point.
- Token-independent endpoint preflight with automation-friendly exit codes and secret-safe output.
- Focused unit/integration tests, package build metadata, and Python 3.11–3.13 CI.
- Productization record and accurate operator documentation.
- Samsarix support, security, release, citation, ownership-notice, and trademark policies.
- Source-distribution manifest that includes operator/project documentation and excludes legacy code.
- Bandit, dependency-audit, distribution-metadata, typed-package, and pinned-action CI checks.
- `/samsarix check` for a forced-fresh health result with concurrent-request coalescing.
- Optional proactive incident and recovery embeds with configurable polling and consecutive-result
  thresholds.
- Per-endpoint expected HTTP statuses and secret request-header mappings referenced through
  `SAMSARIX_ENDPOINT_HEADERS_*` environment variables.
- Stable `check-endpoints --format json` output for CI and deployment gates.
- Current competitive research and an explicit narrow product wedge.
- TLS-only enforcement for endpoints that carry configured secret headers.
- A shared minimum interval for sequential forced checks, in addition to overlap coalescing.

### Changed

- Reframed the independent product as a small service-health operator bot.
- Replaced the unrelated dashboard dependency files with the bot's reproducible dependency set.
- Excluded the Helix-coupled `discord_bot_src` snapshot from the shipped wheel.
- Renamed the supported package, CLI, slash-command group, environment variables, and product
  metadata from Helix to Samsarix under the stewardship of Samsarix LLC.
- Standardized the current tree on MPL-2.0 with explicit Samsarix attribution and working contact
  addresses.
- Declared direct `aiohttp` use, bounded supported Python to 3.11–3.13, and refreshed build, test,
  lint, coverage, and GitHub Actions pins.
- Replaced generic 2xx classification with endpoint-specific expected-status contracts while
  retaining 2xx as the default.

### Removed

- Removed the nonfunctional Node manifest, which referenced an absent `index.js` implementation.
- Removed the contradictory historical proprietary license from the current tree; it remains in Git
  history for provenance.

## [1.0.0] - 2026-03-31 (historical extraction claim)

### Added
- Initial release of Helix Discord Bot
- 48 Python files with 11,117 lines of production code
- 50+ Discord commands across 15+ command modules
- Multi-agent coordination and swarm integration
- Real-time performance monitoring
- Advanced content generation
- Voice features and audio integration
- Memory management system
- Webhook integration
- Portal deployment capabilities
- Admin and moderation tools
- Fun minigames and entertainment
- Comprehensive documentation
- Apache 2.0 + Proprietary licensing

### Features
- Agent bot factory for dynamic bot creation
- Agent memory service for persistent context
- Performance tracking and optimization
- Multi-agent enhancement system
- Discord slash commands support
- Cog-based command organization
- Webhook sender with hybrid support
- Voice channel management
- Guild storage and configuration
- Internationalization (i18n) support
- Web bridge for external integration
- Account linking system
