# Ownership Regression Fix Report

Status: PVR_EC_OWNERSHIP_HOT_PATH_REGRESSION

Root cause: regular ownership_top1 was benchmarked through the richer diagnostic route and paid first-call CUDA warmup, while the canary used the warmed tight path

| model | latency_ms | loss | mae | accuracy |
|---|---:|---:|---:|---:|
| pvr_ec_deploy_top1 | 0.0143 | 0.992414 | 0.613781 | 0.7500 |
| pvr_ec_ownership_top1_frozen_candidate | 0.1245 | 0.992414 | 0.613781 | 0.7500 |

Before ownership_top1 latency: 204.79148699996585
After ownership_top1 latency: None

Hot-path code moved out of forward:
- replay/oracle probes
- report writing
- candidate map validation
- challenger diagnostics

Final statuses: PVR_EC_OWNERSHIP_HOT_PATH_REGRESSION, PVR_EC_DO_NOT_PROMOTE
