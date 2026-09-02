# Unified MLB migration and rollback

## Current state

The unified engine is a development shadow. `static-deployment` and its legacy
entrypoint remain authoritative. The daily workflow builds
`unified_predictions.json` only after all legacy products have completed.

## Migration gate

Authority may move only after:

1. unified and affected legacy tests pass in CI;
2. artifacts in web, dist, and private targets are byte-equivalent;
3. representative exact/reconstructable point-in-time slates have been
   compared with the legacy output;
4. locked validation and a prospective shadow period support promotion;
5. the frontend has been visually verified with populated and abstaining 2-,
   3-, 4-leg and SGP states;
6. no selected candidate lacks probability, price, identity, support or role;
7. no selected leg has non-positive conservative EV;
8. an approved migration change explicitly switches authority.

No current code path performs step 8.

## Rollback

Before authority migration, rollback is automatic: remove or ignore
`unified_predictions.json`; the legacy files and renderers remain unchanged.

After a future explicit migration:

1. revert the migration commit only;
2. restore the legacy frontend source flag/entrypoint;
3. rebuild the static site;
4. validate the legacy daily, same-game, pitcher and exotic artifacts;
5. publish the rebuild.

Do not delete historical unified shadow artifacts during rollback; they are
research evidence and should remain auditable.
