# Ownership Regression Fix Report

Status: PVR_EC_OWNERSHIP_TOP1_LATENCY_MATCHES_CANARY

Root cause: regular ownership_top1 was benchmarked through the richer diagnostic route and paid first-call CUDA warmup, while the canary used the warmed tight path

| model | latency_ms | loss | mae | accuracy |
|---|---:|---:|---:|---:|
| fixed_moe_vectorized | 1.9365 | 0.650065 | 0.635804 | 0.6562 |
| pvr_ec_deploy_top1 | 0.0297 | 1.187547 | 0.719799 | 0.6250 |
| pvr_ec_ownership_top1 | 0.0359 | 1.187547 | 0.719799 | 0.6250 |
| pvr_ec_ownership_top1_disabled | 0.0412 | 1.187547 | 0.719799 | 0.6250 |
| pvr_ec_ownership_top1_shadow | 0.0390 | 1.187547 | 0.719799 | 0.6250 |
| pvr_ec_ownership_top1_frozen_production | 0.6067 | 1.187547 | 0.719799 | 0.6250 |
| pvr_ec_ownership_top1_frozen_candidate | 0.1331 | 1.187547 | 0.719799 | 0.6250 |
| pvr_ec_ownership_top1_candidate_canary | 0.1127 | 1.187547 | 0.719799 | 0.6250 |

Before ownership_top1 latency: 204.79148699996585
After ownership_top1 latency: 0.03589999960240675

Hot-path code moved out of forward:
- replay/oracle probes
- report writing
- candidate map validation
- challenger diagnostics

Final statuses: PVR_EC_OWNERSHIP_TOP1_LATENCY_MATCHES_CANARY, PVR_EC_DO_NOT_PROMOTE
