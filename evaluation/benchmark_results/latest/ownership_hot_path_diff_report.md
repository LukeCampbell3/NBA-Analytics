# Ownership Hot Path Diff Report

Status: PVR_EC_OWNERSHIP_TOP1_LATENCY_MATCHES_CANARY

| model | latency_ms | ownership_lookup_ms | ownership_score_ms | argmax_owner_ms | purity | dominant_source |
|---|---:|---:|---:|---:|---:|---|
| pvr_ec_deploy_top1 | 0.0297 | 0.0000 | 0.0000 | 0.0065 | 1.0000 | none |
| pvr_ec_ownership_top1_disabled | 0.0412 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | none |
| pvr_ec_ownership_top1_shadow | 0.0390 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | none |
| pvr_ec_ownership_top1_frozen_production | 0.6067 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | none |
| pvr_ec_ownership_top1_frozen_candidate | 0.1331 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | none |
| pvr_ec_ownership_top1_candidate_canary | 0.1127 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | none |
| pvr_ec_ownership_top1 | 0.0359 | 0.0000 | 0.0000 | 0.0064 | 1.0000 | none |
