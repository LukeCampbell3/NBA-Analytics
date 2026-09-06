# Sequential PA H/TB integration

`sequential_pa_contact_model_v1` is the advanced H/TB structural-probability path.

Production contract:

- each hitter plate appearance is simulated sequentially as `K | BB | HBP | HR | NON_HR_CONTACT | OTHER`;
- non-HR contact resolves as `OUT | 1B | 2B | 3B | ROE_OTHER`;
- PA and AB are tracked separately;
- `P(H over 0.5) = 1 - P(H=0)` and `P(TB over 1.5) = P(TB>=2)` are read directly from simulated nights;
- Statcast expected contact values are average-context baselines; specific defense is a zero-centered residual and is never fabricated when OAA/location evidence is unavailable;
- advanced data are timestamped/as-of constrained and fail closed when freshness or identity cannot be established;
- the model has negative publication authority until an independently validated recalibrator exists: it may lower or veto legacy H/TB confidence, but may not increase it;
- the legacy H/TB path remains the declared fallback when advanced evidence is unavailable or weak;
- current canonical adapters carry structural, calibrated, usable, lower-bound, uncertainty, support, lineup, quote and identity facts into the frozen V2.1 policy without changing that policy's thresholds.

This document is descriptive only and grants no staking or certification authority.
