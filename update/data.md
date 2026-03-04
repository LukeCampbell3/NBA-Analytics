# data.md — Glossary + Example Unified Output Schema

This file contains:
- A **glossary** (definition contract) for core components.
- An **example schema JSON** you can standardize across all tools.

---

## Glossary

Each term includes:
- **Meaning** (plain language)
- **Inputs** (which card fields / tool outputs it uses)
- **Scale**
- **Interpretation**
- **Failure modes** (how it can be wrong)

### spacing_gravity_skill
- **Meaning:** Proxy for how much a player forces defensive attention away from the ball because of shooting profile and deployment (gravity).
- **Inputs:** `shot_profile.three_point_frequency`, `traditional.three_point_pct`, `creation_profile.assisted_rate`, archetype
- **Scale:** 0–1
- **Interpretation:**  
  - High = strong spacing value that can exist even without high box score volume.  
  - Low = spacing unlikely to be a meaningful driver.
- **Failure modes:** Missing/unstable 3P% samples; movement-action gravity not captured; team usage of gravity not represented.

### suppression_relief
- **Meaning:** Estimated improvement in utilization of a player’s hidden/ecosystem skills when moving from current team style to candidate team style.
- **Inputs:** `hidden_skill_total`, `current_team_utilization.hidden_system_total`, `candidate_team_utilization.hidden_system_total`
- **Scale:** -1–1 (typically 0–1 when framed as “relief”)
- **Interpretation:**  
  - Positive = candidate environment uses the player’s hidden skills better.  
  - Near zero = environment change is unlikely to unlock new value.
- **Failure modes:** Team style proxies too thin; missing scheme representation; player skill decomposition incomplete.

### portability_score
- **Meaning:** How well a player’s defensive value survives matchup pressure (switch hunting, size targeting, foul risk) and translates across playoff contexts.
- **Inputs:** position/size, defensive proxies (steal/block/foul rates if present), on/off defense proxies (if present), defensive role inference
- **Scale:** 0–1
- **Interpretation:**  
  - High = fewer playoff “attack points”; role survives switching and targeting.  
  - Low = likely to be hunted / requires scheme protection.
- **Failure modes:** Missing tracking/lineup context; box-only proxies can mislead; unusual schemes can break inference.

### impact_prior_disagreement
- **Meaning:** Distance between your system’s direction/valuation and an external impact prior (EPM/LEBRON/DARKO/RAPM family).
- **Inputs:** `impact_prior.value`, `your_model.projected_value`, `impact_prior.uncertainty`
- **Scale:** points/100 (preferred) + normalized 0–1 severity
- **Interpretation:**  
  - Large = flag for review + reduce confidence unless justified.  
  - Small = confidence bonus.
- **Failure modes:** Prior stale or injured season; role changed; season-window mismatch; priors differ in modeling choices.

### comp_density
- **Meaning:** How “anchorable” the comps are (how many near neighbors exist).
- **Inputs:** top-k similarity scores from portfolio clustering
- **Scale:** 0–1
- **Interpretation:**  
  - Low = unique role → comps weak → widen uncertainty.  
  - High = archetype well-represented → comps stabilize projections.
- **Failure modes:** Similarity features incomplete; clustering biased toward box stats; missing role/defense traits.

### clamp_severity
- **Meaning:** How strongly constraints restrict the counterfactual (role expansion, defense survivability, contract feasibility, team role budgets).
- **Inputs:** scenario-screen violations/caps, defense portability clamps, contract/roster feasibility checks
- **Scale:** 0–1
- **Interpretation:**  
  - High = “best case” still constrained.  
  - Low = projection more feasible and less forced.
- **Failure modes:** Clamp definitions too conservative/loose; missing constraints (coach/system, roster composition).

---

## example_schema.json

Below is an **example JSON object** (not a strict JSON Schema Draft) that you can standardize across all tools.
Tools can add fields, but they should not break this shape. Use `null` when unavailable.

```json
{
  "header": {
    "player_id": "12345",
    "player_name": "Example Player",
    "team": "DEN",
    "season": "2025",
    "generated_at": "2026-03-04T00:00:00Z",
    "tool_versions": {
      "scouting_report": "1.0.0",
      "context_offense_eval": "1.0.0",
      "risk_decision_support": "1.0.0",
      "scenario_screening": "1.0.0",
      "isolated_value": "1.0.0",
      "breakout_detector": "1.0.0",
      "impact_sanity_check": "0.1.0",
      "defense_portability": "0.1.0",
      "contract_valuation": "0.1.0"
    }
  },

  "identity": {
    "position": "SF",
    "height_in": null,
    "weight_lb": null,
    "role_identity": {
      "primary_role": "connector_wing",
      "usage_band": "low_to_mid",
      "shot_mix": {
        "rim_freq": null,
        "mid_freq": null,
        "three_freq": 0.38
      },
      "creation_profile": {
        "self_created_rate": null,
        "assisted_rate": 0.72
      }
    },
    "archetype": {
      "primary_archetype": "connector",
      "archetype_confidence": 0.64
    },
    "comparables": {
      "top_comps": [
        { "player_name": "Comp A", "similarity": 0.81 },
        { "player_name": "Comp B", "similarity": 0.78 }
      ],
      "comp_density": 0.62,
      "notes": "Moderate comp density; comps are usable anchors."
    }
  },

  "signals": {
    "value_vector": {
      "on_ball": 0.28,
      "off_ball": 0.62,
      "spacing": 0.70,
      "creation": 0.30,
      "finishing": 0.44,
      "rim_defense": 0.22,
      "poa_defense": 0.48,
      "switchability": 0.55,
      "rebounding": 0.40,
      "foul_risk": 0.22
    },

    "hidden_ecosystem": {
      "spacing_gravity_skill": 0.74,
      "connector_skill": 0.61,
      "finisher_ecosystem_skill": 0.42,
      "hidden_skill_total": 0.64,
      "current_team_utilization": {
        "spacing_system_util": 0.44,
        "connector_system_util": 0.52,
        "finisher_system_util": 0.40,
        "hidden_system_total": 0.46
      }
    },

    "context_adjusted_offense": {
      "production_vs_schedule_strength": null,
      "team_help_hurt": -0.04,
      "context_influence_per100": -1.2,
      "notes": "Player appears mildly suppressed by current environment."
    }
  },

  "counterfactuals": {
    "best_team_fit": {
      "candidate_team": "IND",
      "opportunity_score_0_100": 78.2,
      "projected_metrics": {
        "mpg": 31.8,
        "usage_rate": 0.22,
        "ppg": 15.4,
        "apg": 4.2,
        "rpg": 5.8
      },
      "deltas": {
        "minutes_delta": 2.6,
        "usage_delta": 0.03,
        "points_delta": 1.8,
        "assists_delta": 0.6,
        "hidden_value_delta": 0.14,
        "suppression_relief": 0.12
      },
      "clamps": {
        "scenario_feasible": true,
        "clamps_applied": [],
        "clamp_severity": 0.18
      }
    }
  },

  "anchors": {
    "impact_prior": {
      "source": "EPM_like_proxy",
      "value_points_per100": 1.8,
      "uncertainty": 0.35
    },
    "impact_sanity_check": {
      "your_projected_impact_points_per100": 2.4,
      "impact_prior_disagreement_points_per100": 0.6,
      "disagreement_severity_0_1": 0.22,
      "verdict": "supportive_or_neutral",
      "flags": []
    },
    "defense_portability": {
      "defensive_role_identity": {
        "primary": "wing",
        "secondary": "help",
        "poa": 0.48,
        "rim": 0.22,
        "screen_nav": null
      },
      "portability_score": 0.58,
      "playoff_risk_flags": ["may_be_targeted_in_switch_heavy_series"],
      "defense_clamps": {
        "limits_role_expansion": false,
        "notes": "Playable but may need scheme support vs elite creators."
      }
    }
  },

  "valuation": {
    "contract": {
      "aav_millions": null,
      "years_remaining": null,
      "contract_tier": null,
      "notes": "Populate from your contract tool."
    },
    "aging_curve": {
      "age": null,
      "archetype_curve_id": "connector_v1",
      "expected_delta_next_year": null,
      "notes": "Archetype-specific aging adjustment placeholder."
    },
    "surplus_value": {
      "projected_value_units": null,
      "cost_units": null,
      "surplus_units": null,
      "notes": "SurplusValue = ProjectedValue - ContractCost"
    }
  },

  "confidence": {
    "overall_score_0_1": 0.66,
    "level": "medium",
    "components": {
      "data_quality": 0.72,
      "extrapolation_inverse": 0.78,
      "model_agreement": 0.70,
      "isolation_inverse": 0.62,
      "clamps_inverse": 0.82,
      "comp_density": 0.62
    },
    "warnings": [
      "Defense proxies limited; portability has moderate uncertainty."
    ]
  },

  "decision": {
    "verdicts": {
      "visible_breakout": false,
      "ecosystem_breakout": true,
      "fit_upside": true
    },
    "flags": [
      "ecosystem_value_uplift_validated"
    ],
    "review_questions": [
      "Does the candidate team actually run enough off-ball actions to realize spacing gravity?"
    ]
  }
}
```
