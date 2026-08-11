# Competitive product research

Research date: 2026-08-08

## Market shape

The mature uptime-monitoring market is broad. Uptime Kuma provides dashboards, public status
pages, incident history, notification suppression, and more than 90 notification providers.
Gatus supports repeated checks across several network protocols, condition expressions, failure
and recovery thresholds, reminders, and multiple alert providers. Better Stack combines hosted
monitors with incident creation and on-call escalation.

Primary sources:

- [Uptime Kuma repository](https://github.com/louislam/uptime-kuma) and
  [notification methods](https://github.com/louislam/uptime-kuma/wiki/Notification-Methods)
- [Gatus endpoints](https://gatus.io/docs/endpoints),
  [conditions](https://gatus.io/docs/conditions), and
  [alerting](https://gatus.io/docs/alerting-getting-started)
- [Better Stack monitoring introduction](https://betterstack.com/docs/uptime/monitoring-start/)
- [Discord interactions](https://docs.discord.com/developers/platform/interactions),
  [application-command permissions](https://docs.discord.com/developers/interactions/application-commands),
  and [rate limits](https://docs.discord.com/developers/topics/rate-limits)

Samsarix should not imitate those products' dashboard breadth. A small team that already operates
inside Discord has a narrower problem: safely check a bounded private service set, see current
state without exposing destinations, and receive a low-noise incident transition without
deploying a database or public status page.

## Chosen competitive wedge

Samsarix Discord Operator Bot is a private, Discord-native operator companion rather than a general
monitoring platform. The competitive release slice consists of:

1. **Pull status:** ephemeral `/samsarix status` responses backed by a short shared cache.
2. **Fresh diagnostics:** `/samsarix check` bypasses a warm cache while coalescing concurrent calls.
3. **Proactive transitions:** optional polling posts one incident after consecutive failures and one
   recovery after consecutive successes. Repeated identical states do not create alert noise.
4. **Private-service compatibility:** an endpoint may reference a separate secret JSON environment
   variable for request headers; values are never included in summaries, CLI output, Discord, or
   expected configuration errors.
5. **Real health contracts:** operators may list accepted HTTP statuses per endpoint instead of
   assuming every valid readiness response is any 2xx status.
6. **Automation:** `check-endpoints --format json` provides a stable, URL-free schema and meaningful
   exit codes for CI, deployment gates, and process-supervisor probes.

This keeps the runtime to one Python process, one Discord connection, and outbound HTTP requests.
There is no database, public listener, hosted dependency, telemetry service, or metered API.

## Deliberate non-goals

- Public dashboards and status pages
- Persistent uptime percentages or incident history
- Paging schedules, acknowledgement workflows, or multi-level escalation
- Arbitrary response-body capture or expression evaluation
- User-supplied destinations, headers, commands, or executable checks
- Dozens of notification providers
- Automatic discovery of private network services

These are established product categories with materially different security, privacy, persistence,
and support obligations. Samsarix can interoperate with a larger monitor later, but its independent
value is the small Discord-native operating loop.

## Future evidence to gather

- Pilot feedback on alert usefulness and false-positive frequency
- Whether operators need opt-in bounded JSON-field assertions
- Whether scheduled-maintenance suppression is more valuable than alert reminders
- Whether a container image or native process-supervisor examples cover most deployment demand
- Live Discord evidence for channel permissions, reconnect behavior, and delivery under rate limits

No demand or product-market fit is claimed; these are hypotheses for bounded validation.
