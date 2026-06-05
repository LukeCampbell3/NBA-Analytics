# Ownership Regression Fix Report

Status: PVR_EC_OWNERSHIP_TOP1_LATENCY_MATCHES_CANARY

Root cause: regular ownership_top1 was benchmarked through the richer diagnostic route and paid first-call CUDA warmup, while the canary used the warmed tight path

| model | latency_ms | loss | mae | accuracy |
|---|---:|---:|---:|---:|
| fixed_moe_vectorized | 0.6048 | 1.148112 | 0.862780 | 0.6250 |
| pvr_ec_deploy_top1 | 0.1937 | 1.983451 | 0.994538 | 0.6211 |
| pvr_ec_ownership_top1 | 0.3834 | 1.983451 | 0.994538 | 0.6211 |
| pvr_ec_ownership_top1_candidate_canary | 7.2052 | 1.933416 | 0.973656 | 0.6250 |

Before ownership_top1 latency: 204.79148699996585
After ownership_top1 latency: 0.38336099987645866

Hot-path code moved out of forward:
- replay/oracle probes
- report writing
- candidate map validation
- challenger diagnostics

Final statuses: PVR_EC_OWNERSHIP_TOP1_LATENCY_MATCHES_CANARY, PVR_EC_DO_NOT_PROMOTE
