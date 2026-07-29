# Historical Helix Discord Extraction

`discord_bot_src/` is preserved source history from the larger Helix ecosystem. It is not the
supported standalone product in this repository.

The snapshot:

- imports unreleased `apps.backend` modules from another checkout;
- combines moderation, LLM, code-execution, account, voice, webhook, and monitoring experiments;
- requests privileged message content in several entry points;
- has no independently reproducible dependency or test contract;
- is excluded from the `samsarix-discord-bot` wheel and CI type/lint scope.

Do not run `discord_bot_src.discord_bot_helix` as a standalone deployment. The maintained product is
documented in [README.md](README.md) and lives in `samsarix_discord_bot/`.

Future work may extract individual legacy capabilities only when they can be permission-minimal,
independently configured, tested through a real interface, and documented without private Helix
dependencies. Until then, syntax compilation is the only CI guarantee for this tree.
