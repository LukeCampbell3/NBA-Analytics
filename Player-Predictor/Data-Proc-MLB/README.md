# Data-Proc-MLB

Sample MLB-native processed data and schema contract for cloning the
existing `Player-Predictor` architecture pattern without cross-sport bridging.

## Layout

- `schema/mlb_native_player_schema_v1.json`: machine-readable MLB contract.
- `Mookie_Betts/2026_processed_processed.csv`: sample hitter file in MLB-native format.
- `update_manifest_2026.json`: sample build/update manifest.

## Validation

```bash
python scripts/validate_mlb_processed_contract.py --data-dir Data-Proc-MLB
```
