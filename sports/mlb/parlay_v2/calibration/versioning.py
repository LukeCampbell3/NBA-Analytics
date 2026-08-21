from __future__ import annotations

"""Version identifiers for the calibration stream (mission section 3/19-N:
"changing predictive/calibration/policy version creates a new compatible
stream or causes replay/admission refusal"). Bumping CALIBRATION_VERSION
is a deliberate, frozen decision -- observations/snapshots are tagged with
the version active when they were written, and admission/replay refuse to
silently mix incompatible versions."""

CALIBRATION_VERSION = "CALIBRATION_LEDGER_V1"
SUPPORT_RULE_VERSION = "SUPPORT_RULE_V1"


def assert_compatible(observed_version: str, *, expected: str = CALIBRATION_VERSION, context: str = "") -> None:
    if observed_version != expected:
        raise ValueError(
            f"calibration version mismatch{' (' + context + ')' if context else ''}: "
            f"observed {observed_version!r} != expected {expected!r} -- refuse to mix streams across versions"
        )
