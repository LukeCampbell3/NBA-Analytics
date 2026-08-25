# Onboarding a New Sport

Spec section 40 acceptance condition: adding a sport must not require
editing the Transformer/MoE/DRM internals. This is enforced by the
`SportAdapter` contract (`adapters/base.py`) and demonstrated by a real
test fixture (`tests/test_universal_model.py::test_I_hypothetical_new_sport_needs_no_model_change`,
a hypothetical "cricket" adapter defined entirely inside the test file with
zero imports from `model/`, `train/`, or `drm_controller/`).

## Required steps

1. **Implement `SportAdapter`** (`adapters/<sport>.py`): `discover_sources`,
   `build_observations` (return `(list[UniversalEvent], SourceCoverage)` --
   report `sufficient_for_training=False` honestly if there isn't enough
   real settled history; see `adapters/nba.py`/`golf.py`/`f1.py` for the
   pattern), `map_universal_features`, `map_namespaced_features`,
   `build_targets`.
2. **Register it** in `adapters/registry.py`'s `ALL_ADAPTERS` dict.
3. **Register any real, verified pregame features** in
   `data/feature_registry.py`'s per-sport column spec (classify each real
   source column -- do not hand-invent features that don't exist in real
   data; see the module docstring for the classification vocabulary and
   the standard of evidence used for MLB/NFL).
4. **Register target mappings**: extend `data/schema.py`'s
   `TARGET_FAMILIES` if the sport introduces a genuinely new target family.
5. **Compile the dataset**: `python -m sports.universal_model.data.compiler`
   picks up any adapter reporting `sufficient_for_training=True`
   automatically -- no changes needed there.
6. **Run validation**: `python -m sports.universal_model.validation.run_full_validation`.

## What you do NOT need to touch

`model/`, `train/`, `drm_controller/` -- the tokenizer, dense/Switch/Top-2
stem, and DRM controller are all sport-agnostic by construction (they
consume `UniversalEvent`/typed tensors, never a sport-specific field
directly). This has been true in practice, not just in principle: five
real adapters (mlb, nfl, nba, golf, f1) exist today and only two of them
(`sufficient_for_training=True`) ever reach the model -- the other three
prove the *interface* works end-to-end without a single model-layer change,
even though their real data doesn't support training yet.
