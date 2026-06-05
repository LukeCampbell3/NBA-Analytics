#!/usr/bin/env python3
"""
Production-Shadow Daemon

Continuously runs the production-shadow evidence cycle:
  predecision → close → settle → status → evidence rebuild → training export

Usage:
  python Player-Predictor/scripts/run_production_shadow_daemon.py
  python Player-Predictor/scripts/run_production_shadow_daemon.py --once
  python Player-Predictor/scripts/run_production_shadow_daemon.py --once --dry-run
  python Player-Predictor/scripts/run_production_shadow_daemon.py --poll-minutes 10

Safety:
  - staking_enabled = false always
  - production_status = blocked unless gates pass → ready_for_manual_review
  - never auto-enables staking
  - never runs deprecated runners
  - never counts mocked/replay as live evidence
"""
from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

# Paths
WORKSPACE = Path(__file__).resolve().parents[2]
RUNNER_SCRIPT = WORKSPACE / "Player-Predictor" / "scripts" / "run_production_shadow_daily.py"
SHADOW_DIR = WORKSPACE / "sports" / "validation" / "production_shadow"
STATE_PATH = SHADOW_DIR / "daemon_state.json"
PROVIDER_STATE_PATH = SHADOW_DIR / "provider_runtime_state.json"
LOCK_PATH = SHADOW_DIR / "production_shadow_daemon.lock"
LOG_DIR = SHADOW_DIR / "logs"
LEDGER_PATH = SHADOW_DIR / "live_ledger.csv"
EXPORT_DIR = SHADOW_DIR / "training_exports"
EVIDENCE_STATUS_PATH = SHADOW_DIR / "evidence_accumulation_status.json"
ACTIVE_CONFIG_PATH = SHADOW_DIR / "active_production_config.json"
PRODUCTION_STATUS_PATH = SHADOW_DIR / "production_status.json"
PROMOTION_REPORT_PATH = SHADOW_DIR / "promotion_decision_report.json"
LIVE_SLATE_HISTORY_PATH = SHADOW_DIR / "live_slate_history.csv"
CLV_SURROGATE_DIR = SHADOW_DIR / "clv_surrogate"
CLV_TRAIN_SCRIPT = CLV_SURROGATE_DIR / "train_clv_surrogate.py"
CLV_EVAL_SCRIPT = CLV_SURROGATE_DIR / "evaluate_clv_surrogate.py"
TARGETED_EVIDENCE_PLAN_PATH = SHADOW_DIR / "targeted_evidence_plan.json"
GATE_ACCELERATION_REPORT_PATH = SHADOW_DIR / "gate_acceleration_report.json"

# Ensure directories
LOG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup file + console logging."""
    logger = logging.getLogger("shadow_daemon")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    # Console: concise
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    # File: detailed
    log_file = LOG_DIR / f"production_shadow_daemon_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return logger


# ─── Lock Management ─────────────────────────────────────────────

def acquire_lock(logger: logging.Logger) -> bool:
    """Acquire daemon lock. Returns True if acquired."""
    if LOCK_PATH.exists():
        try:
            lock_data = json.loads(LOCK_PATH.read_text())
            pid = lock_data.get("pid", 0)
            # Check if process is alive
            if pid and _pid_alive(pid):
                logger.error(f"Another daemon is running (PID {pid}). Exiting.")
                return False
            else:
                logger.warning(f"Stale lock found (PID {pid} not alive). Removing.")
                LOCK_PATH.unlink()
        except (json.JSONDecodeError, OSError):
            LOCK_PATH.unlink(missing_ok=True)

    lock_data = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "hostname": os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown")),
    }
    LOCK_PATH.write_text(json.dumps(lock_data, indent=2))
    return True


def release_lock():
    """Release daemon lock."""
    LOCK_PATH.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    """Check if a PID is alive."""
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# ─── Provider Cooldown ───────────────────────────────────────────

class ProviderCooldown:
    """Manages provider request cooldown to avoid quota waste."""

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if PROVIDER_STATE_PATH.exists():
            try:
                return json.loads(PROVIDER_STATE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "provider": "sportsgameodds",
            "last_attempt_at": None,
            "last_success_at": None,
            "last_status": None,
            "blocked_until": None,
            "cooldown_seconds": 300,
            "requests_today": 0,
            "errors_today": 0,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        }

    def save(self):
        PROVIDER_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROVIDER_STATE_PATH.write_text(json.dumps(self.state, indent=2))

    def can_request(self) -> bool:
        """Check if we can make a provider request now."""
        now = datetime.now(timezone.utc)

        # Reset daily counters
        today = now.strftime("%Y-%m-%d")
        if self.state.get("date") != today:
            self.state["date"] = today
            self.state["requests_today"] = 0
            self.state["errors_today"] = 0

        # Check blocked_until
        blocked = self.state.get("blocked_until")
        if blocked:
            blocked_dt = datetime.fromisoformat(blocked)
            if now < blocked_dt:
                return False

        # Check cooldown
        last = self.state.get("last_attempt_at")
        if last:
            last_dt = datetime.fromisoformat(last)
            cooldown = self.state.get("cooldown_seconds", 300)
            if (now - last_dt).total_seconds() < cooldown:
                return False

        return True

    def record_attempt(self, status: str):
        """Record a provider attempt result."""
        now = datetime.now(timezone.utc)
        self.state["last_attempt_at"] = now.isoformat()
        self.state["last_status"] = status
        self.state["requests_today"] = self.state.get("requests_today", 0) + 1

        if status == "success":
            self.state["last_success_at"] = now.isoformat()
            self.state["cooldown_seconds"] = 300  # Normal 5-min cooldown
            self.state["blocked_until"] = None
        elif status == "no_props":
            self.state["cooldown_seconds"] = 900  # 15 min if no props
        elif status in ("missing_credentials", "quota_exhausted"):
            self.state["blocked_until"] = (now + timedelta(hours=1)).isoformat()
            self.state["errors_today"] = self.state.get("errors_today", 0) + 1
        elif status == "api_error":
            self.state["cooldown_seconds"] = 600  # 10 min on error
            self.state["errors_today"] = self.state.get("errors_today", 0) + 1
        else:
            self.state["cooldown_seconds"] = 300

        self.save()


# ─── Daemon State ────────────────────────────────────────────────

class DaemonState:
    """Manages daemon state persistence."""

    def __init__(self):
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if STATE_PATH.exists():
            try:
                return json.loads(STATE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "daemon_started_at": datetime.now(timezone.utc).isoformat(),
            "last_loop_at": None,
            "loop_count": 0,
            "mode": "production_shadow",
            "staking_enabled": False,
            "last_predecision_at": None,
            "last_close_at": None,
            "last_settle_at": None,
            "last_status_at": None,
            "last_evidence_rebuild_at": None,
            "last_phase_result": None,
            "active_provider": "sportsgameodds",
            "provider_status": None,
            "pending_close_rows": 0,
            "pending_settle_rows": 0,
            "settled_class_a_rows": 0,
            "unique_live_slates": 0,
            "failed_gates": [],
            "next_action": None,
            "next_wakeup_at": None,
        }

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state, indent=2, default=str))

    def update(self, **kwargs):
        self.state.update(kwargs)
        self.save()


# ─── Phase Runner ────────────────────────────────────────────────

def run_phase(phase: str, prediction_mode: str = "real", dry_run: bool = False,
              provider: str = "sportsgameodds",
              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Run a single phase of the production-shadow runner via subprocess."""
    cmd = [
        sys.executable, str(RUNNER_SCRIPT),
        "--phase", phase,
        "--prediction-mode", prediction_mode,
    ]
    if provider:
        cmd.extend(["--provider", provider])

    if logger:
        logger.info(f"running {phase}")

    if dry_run:
        if logger:
            logger.info(f"  [dry-run] would execute: {' '.join(cmd)}")
        return {"phase": phase, "status": "dry_run", "exit_code": 0}

    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=120,
            cwd=str(WORKSPACE), env={**os.environ, "PYTHONUTF8": "1"},
            encoding="utf-8", errors="replace"
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        exit_code = result.returncode

        # Parse key metrics from stdout
        metrics = _parse_phase_output(stdout, phase)
        metrics["exit_code"] = exit_code
        metrics["phase"] = phase

        if exit_code != 0 and logger:
            logger.warning(f"  {phase} exited with code {exit_code}")
            if stderr:
                logger.debug(f"  stderr: {stderr[:200]}")

        return metrics

    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"  {phase} timed out")
        return {"phase": phase, "status": "timeout", "exit_code": -1}
    except Exception as e:
        if logger:
            logger.error(f"  {phase} error: {e}")
        return {"phase": phase, "status": "error", "exit_code": -1, "error": str(e)}


def _parse_phase_output(stdout: str, phase: str) -> Dict[str, Any]:
    """Parse key metrics from runner stdout."""
    metrics: Dict[str, Any] = {"status": "unknown"}

    for line in stdout.splitlines():
        line_stripped = line.strip()
        if "Appended rows:" in line_stripped:
            try:
                metrics["appended_rows"] = int(line_stripped.split("Appended rows:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Decisions settled:" in line_stripped:
            try:
                metrics["settled_rows"] = int(line_stripped.split("Decisions settled:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Decisions closed:" in line_stripped or "Closed decisions:" in line_stripped:
            try:
                val = line_stripped.split(":")[-1].strip()
                metrics["closed_rows"] = int(val)
            except (ValueError, IndexError):
                pass
        elif "NO FRESH ODDS" in line_stripped:
            metrics["status"] = "no_fresh_odds"
        elif "Fresh odds obtained" in line_stripped:
            metrics["status"] = "fresh_odds_obtained"
        elif "PREDECISION PHASE COMPLETE" in line_stripped:
            metrics["status"] = "success"
        elif "CLOSE PHASE COMPLETE" in line_stripped:
            metrics["status"] = "success"
        elif "SETTLE PHASE COMPLETE" in line_stripped:
            metrics["status"] = "success"
        elif "No pending decisions" in line_stripped:
            metrics["status"] = "no_pending"

    return metrics


# ─── Ledger Inspection ───────────────────────────────────────────

def inspect_ledger() -> Dict[str, Any]:
    """Inspect the live ledger for pending rows."""
    import pandas as pd

    if not LEDGER_PATH.exists():
        return {"total": 0, "pending_close": 0, "pending_settle": 0, "settled": 0, "gold_real_clv_rows": 0}

    try:
        df = pd.read_csv(LEDGER_PATH)
    except Exception:
        return {"total": 0, "pending_close": 0, "pending_settle": 0, "settled": 0, "gold_real_clv_rows": 0}

    if df.empty:
        return {"total": 0, "pending_close": 0, "pending_settle": 0, "settled": 0, "gold_real_clv_rows": 0}

    n = len(df)

    # Pending close: has entry but no close
    pending_close = df[
        ((df["close_snapshot_id"].isna()) | (df["close_snapshot_id"] == "")) &
        (df["entry_snapshot_id"].notna()) & (df["entry_snapshot_id"] != "")
    ]

    # Pending settle: has close but not settled
    pending_settle = df[
        (df["close_snapshot_id"].notna()) & (df["close_snapshot_id"] != "") &
        ((df["settled_at"].isna()) | (df["settled_at"] == ""))
    ]

    # Settled
    settled = df[(df["settled_at"].notna()) & (df["settled_at"] != "")]
    class_a_settled = settled[settled.get("class_a_candidate", pd.Series([False] * len(settled))) == True]
    gold_real = settled[
        (settled.get("side_aware_prob_clv", pd.Series([pd.NA] * len(settled))).notna()) &
        (settled.get("close_snapshot_id", pd.Series([""] * len(settled))).fillna("").astype(str) != "") &
        (~settled.get("evidence_source", pd.Series([""] * len(settled))).fillna("").astype(str).str.lower().isin(["mocked", "replay", "historical_backtest"]))
    ]

    return {
        "total": n,
        "pending_close": len(pending_close),
        "pending_settle": len(pending_settle),
        "settled": len(settled),
        "settled_class_a": len(class_a_settled),
        "gold_real_clv_rows": len(gold_real),
        "unique_slates": int(settled["game_id"].nunique()) if not settled.empty else 0,
    }


# ─── Training Export ─────────────────────────────────────────────

def export_training_datasets(logger: Optional[logging.Logger] = None, include_cross_sport_market_lapse: bool = False):
    """Export training/calibration/CLV datasets from settled live evidence."""
    import pandas as pd

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    def _run_sparse_exports() -> None:
        try:
            sys.path.insert(0, str(SHADOW_DIR))
            from sparse_season_data_mode import run_sparse_season_exports
            sparse_report = run_sparse_season_exports(include_cross_sport_market_lapse=include_cross_sport_market_lapse)
            if logger:
                logger.info(
                    "  sparse mode: %s | historical=%s | silver=%s | gold=%s",
                    sparse_report.get("mode"),
                    sparse_report.get("training_rows_gained"),
                    sparse_report.get("silver_proxy_rows"),
                    sparse_report.get("gold_CLV_rows"),
                )
        except Exception as exc:
            if logger:
                logger.warning(f"  sparse-season export failed: {exc}")

    if not LEDGER_PATH.exists():
        _run_sparse_exports()
        return

    try:
        df = pd.read_csv(LEDGER_PATH)
    except Exception:
        _run_sparse_exports()
        return

    if df.empty:
        _run_sparse_exports()
        return

    # Filter to real live settled rows
    settled_mask = (df["settled_at"].notna()) & (df["settled_at"] != "")
    live_mask = df.get("live_evidence", pd.Series([False] * len(df))) == True
    real_model_mask = df.get("real_model", pd.Series([False] * len(df))) == True
    real_odds_mask = df.get("real_odds", pd.Series([False] * len(df))) == True

    # Training dataset: settled + live + real model + real odds
    training = df[settled_mask & live_mask & real_model_mask & real_odds_mask]
    if not training.empty:
        training.to_csv(EXPORT_DIR / "production_shadow_training_dataset_latest.csv", index=False)

    # Calibration dataset: settled rows with required fields
    calib_cols = ["p_probability_champion", "p_model_raw", "market_no_vig", "hit_loss_push",
                  "brier", "class_a_candidate", "shadow_candidate", "market", "side",
                  "model_edge_raw", "decision_tier", "actual_value"]
    calib = training[[c for c in calib_cols if c in training.columns]].copy()
    if not calib.empty:
        calib.to_csv(EXPORT_DIR / "production_shadow_calibration_dataset_latest.csv", index=False)

    # CLV dataset: rows with close snapshots
    clv_mask = (df["close_snapshot_id"].notna()) & (df["close_snapshot_id"] != "")
    clv_cols = ["player", "market", "line", "side", "book", "entry_snapshot_id",
                "close_snapshot_id", "side_aware_prob_clv", "side_aware_line_clv",
                "market_no_vig", "odds"]
    clv = df[clv_mask & live_mask][[c for c in clv_cols if c in df.columns]].copy()
    if not clv.empty:
        clv.to_csv(EXPORT_DIR / "production_shadow_clv_dataset_latest.csv", index=False)

    if logger:
        logger.info(f"  exports: training={len(training)}, calibration={len(calib)}, clv={len(clv)}")

    _run_sparse_exports()


def run_maintenance_command(script: Path, dry_run: bool, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Run a maintenance script such as CLV surrogate train/evaluate."""
    cmd = [sys.executable, str(script)]
    if dry_run:
        if logger:
            logger.info(f"  [dry-run] would execute: {' '.join(cmd)}")
        return {"script": str(script), "status": "dry_run", "exit_code": 0}
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=180,
            cwd=str(WORKSPACE),
            env={**os.environ, "PYTHONUTF8": "1"},
            encoding="utf-8",
            errors="replace",
        )
        if logger and result.returncode != 0:
            logger.warning(f"  {script.name} exited with code {result.returncode}")
        return {"script": str(script), "status": "success" if result.returncode == 0 else "error", "exit_code": result.returncode}
    except Exception as exc:
        if logger:
            logger.warning(f"  {script.name} failed: {exc}")
        return {"script": str(script), "status": "error", "exit_code": -1, "error": str(exc)}


def evaluate_operational_state(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Evaluate gates, staking controller, and write operational reports."""
    sys.path.insert(0, str(SHADOW_DIR))
    try:
        from cumulative_live_evidence_builder import build_cumulative_evidence
        evidence = build_cumulative_evidence()
    except Exception as exc:
        evidence = {"overall": {}, "error": str(exc)}

    from production_gate_evaluator import ProductionGateEvaluator
    from staking_controller import StakingController, load_active_config

    evaluator = ProductionGateEvaluator()
    gate_status = evaluator.evaluate_all()
    try:
        from targeted_evidence_planner import build_targeted_evidence_plan
        targeted_plan = build_targeted_evidence_plan()
    except Exception as exc:
        targeted_plan = {"error": str(exc), "next_exact_action": "targeted_plan_unavailable"}
    staking_status = StakingController().resolve(gate_status)
    config = load_active_config()
    overall = evidence.get("overall", {})
    metrics = gate_status.get("metrics", {})

    # Prefer direct gate metrics, but retain cumulative fields when available.
    merged_metrics = {**overall, **{k: v for k, v in metrics.items() if v is not None}}
    failed = gate_status.get("failed_gates", []) + gate_status.get("micro_failed_gates", [])
    kill_switches = gate_status.get("kill_switches", [])
    stage = gate_status.get("stage", "production_shadow")
    production_status = "blocked"
    if stage == "production_shadow_accumulating":
        production_status = "production_shadow_accumulating"
    elif stage == "production_candidate":
        production_status = "production_candidate"
    elif stage == "micro_live_ready":
        production_status = "micro_live_ready"
    elif stage == "micro_live":
        production_status = "micro_live"

    status = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "system_mode": config.get("system_mode", "production_shadow"),
        "stage": stage,
        "production_status": production_status,
        "staking_enabled": bool(staking_status.get("staking_enabled", False)),
        "manual_approval_required": bool(config.get("manual_approval_required", True)),
        "settled_live_class_a_rows": int(merged_metrics.get("settled_live_class_a_rows", 0) or 0),
        "gold_real_clv_rows": int(merged_metrics.get("gold_real_clv_rows", merged_metrics.get("total_production_countable_rows", 0)) or 0),
        "unique_live_slates": int(merged_metrics.get("unique_live_slates", 0) or 0),
        "brier": merged_metrics.get("brier"),
        "bss": merged_metrics.get("bss"),
        "ece": merged_metrics.get("ece"),
        "roi": merged_metrics.get("roi"),
        "mean_clv": merged_metrics.get("mean_clv"),
        "positive_clv_rate": merged_metrics.get("positive_clv_rate"),
        "clv_join_rate": merged_metrics.get("clv_join_rate"),
        "settlement_join_rate": merged_metrics.get("settlement_join_rate"),
        "concentration_warnings": [
            name for name in [
                "side_concentration_violation",
                "market_concentration_violation",
                "book_concentration_violation",
            ] if merged_metrics.get(name)
        ],
        "kill_switches": kill_switches,
        "blocker_breakdown": gate_status.get("blocker_breakdown", {}),
        "failed_gates": failed,
        "next_action": gate_status.get("recommended_action", "accumulate_live_evidence"),
        "targeted_next_action": targeted_plan.get("next_exact_action"),
        "targeted_evidence_plan_path": str(TARGETED_EVIDENCE_PLAN_PATH),
        "gate_acceleration_report_path": str(GATE_ACCELERATION_REPORT_PATH),
        "stake_block_reason": staking_status.get("stake_block_reason", ""),
        "class_a_ece": merged_metrics.get("class_a_ece"),
        "class_a_ece_status": merged_metrics.get("class_a_ece_status"),
        "class_a_ece_rows": merged_metrics.get("class_a_ece_rows"),
        "class_a_ece_min_rows": merged_metrics.get("class_a_ece_min_rows"),
        "class_a_ece_gate": merged_metrics.get("class_a_ece_gate"),
        "gate_ece_source": merged_metrics.get("gate_ece_source"),
        "manual_review_checklist": [
            "Confirm >=100 settled live Class A rows",
            "Confirm >=100 gold real CLV rows",
            "Confirm ROI, Brier skill, CLV, and concentration gates",
            "Confirm synthetic/proxy labels are not production evidence",
        ] if stage == "production_candidate" else [],
    }
    PRODUCTION_STATUS_PATH.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    promotion = {
        "created_at": status["computed_at"],
        "current_stage": stage,
        "staking_enabled": status["staking_enabled"],
        "manual_approval_required": status["manual_approval_required"],
        "production_candidate_pass": gate_status.get("production_candidate_pass", False),
        "micro_live_pass": gate_status.get("micro_live_pass", False),
        "failed_gates": failed,
        "passed_gates": gate_status.get("passed_gates", []),
        "micro_passed_gates": gate_status.get("micro_passed_gates", []),
        "evidence_summary": merged_metrics,
        "risk_summary": {
            "kill_switches_active": kill_switches,
            "concentration_warnings": status["concentration_warnings"],
        },
        "blocker_breakdown": gate_status.get("blocker_breakdown", {}),
        "targeted_evidence_plan": targeted_plan,
        "recommended_action": status["next_action"],
        "reason": staking_status.get("stake_block_reason", ""),
        "full_production_auto_enabled": False,
    }
    PROMOTION_REPORT_PATH.write_text(json.dumps(promotion, indent=2, default=str), encoding="utf-8")
    if logger:
        logger.info(f"  stage={stage} staking={status['staking_enabled']} failed_gates={len(failed)}")
    return status


def append_live_slate_history(loop_result: Dict[str, Any]) -> None:
    """Append one operational loop summary to live_slate_history.csv."""
    import pandas as pd

    row = pd.DataFrame([loop_result])
    LIVE_SLATE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LIVE_SLATE_HISTORY_PATH.exists():
        row.to_csv(LIVE_SLATE_HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LIVE_SLATE_HISTORY_PATH, index=False)


# ─── Main Daemon Loop ────────────────────────────────────────────

def daemon_loop(args, logger: logging.Logger):
    """Main daemon scheduling loop."""
    cooldown = ProviderCooldown()
    state = DaemonState()
    state.update(daemon_started_at=datetime.now(timezone.utc).isoformat())

    poll_seconds = args.poll_minutes * 60
    prediction_mode = args.prediction_mode.replace("-", "_")
    max_runtime = timedelta(hours=args.max_runtime_hours) if args.max_runtime_hours > 0 else None
    start_time = datetime.now(timezone.utc)

    logger.info(f"daemon started | poll={args.poll_minutes}m | mode={prediction_mode} | dry_run={args.dry_run}")

    try:
        while True:
            loop_start = datetime.now(timezone.utc)
            state.update(last_loop_at=loop_start.isoformat(), loop_count=state.state["loop_count"] + 1)

            # 1. Inspect ledger
            ledger_info = inspect_ledger()
            state.update(
                pending_close_rows=ledger_info["pending_close"],
                pending_settle_rows=ledger_info["pending_settle"],
                settled_class_a_rows=ledger_info.get("settled_class_a", 0),
                gold_real_clv_rows=ledger_info.get("gold_real_clv_rows", 0),
                unique_live_slates=ledger_info.get("unique_slates", 0),
                mode="production_shadow",
                staking_enabled=False,
            )

            # 2. Decide which phases to run
            ran_something = False

            # PREDECISION: if provider cooldown allows and we want fresh data
            if cooldown.can_request():
                result = run_phase("predecision", prediction_mode, args.dry_run, args.provider, logger)
                state.update(last_predecision_at=loop_start.isoformat(), last_phase_result=result.get("status"))

                # Record provider attempt
                if args.dry_run:
                    pass
                elif result.get("status") == "no_fresh_odds":
                    cooldown.record_attempt("no_props")
                elif result.get("status") in ("success", "fresh_odds_obtained"):
                    cooldown.record_attempt("success")
                    ran_something = True
                    appended = result.get("appended_rows", 0)
                    if appended:
                        logger.info(f"  rows appended: {appended}")
                elif result.get("exit_code", 0) != 0:
                    cooldown.record_attempt("api_error")
                else:
                    cooldown.record_attempt(result.get("status", "unknown"))

            # CLOSE: if pending close rows exist
            ledger_info = inspect_ledger()  # Re-inspect after predecision
            if ledger_info["pending_close"] > 0:
                result = run_phase("close", prediction_mode, args.dry_run, args.provider, logger)
                state.update(last_close_at=loop_start.isoformat())
                closed = result.get("closed_rows", 0)
                if closed:
                    logger.info(f"  closed rows: {closed}")
                    ran_something = True

            # SETTLE: if pending settle rows exist
            ledger_info = inspect_ledger()
            if ledger_info["pending_settle"] > 0:
                result = run_phase("settle", prediction_mode, args.dry_run, args.provider, logger)
                state.update(last_settle_at=loop_start.isoformat())
                settled = result.get("settled_rows", 0)
                if settled:
                    logger.info(f"  settled rows: {settled}")
                    ran_something = True

            # STATUS: always run
            run_phase("status", prediction_mode, args.dry_run, args.provider, logger)
            state.update(last_status_at=loop_start.isoformat())

            # EVIDENCE REBUILD + EXPORT: after settle or periodically
            if ran_something or state.state["loop_count"] % 6 == 0 or args.once:
                export_training_datasets(logger, include_cross_sport_market_lapse=args.include_cross_sport_market_lapse)
                if not args.dry_run:
                    train_result = run_maintenance_command(CLV_TRAIN_SCRIPT, args.dry_run, logger)
                    eval_result = run_maintenance_command(CLV_EVAL_SCRIPT, args.dry_run, logger)
                    state.update(last_clv_surrogate_train=train_result, last_clv_surrogate_eval=eval_result)
                state.update(last_evidence_rebuild_at=loop_start.isoformat())

            operational_status = evaluate_operational_state(logger)

            # Update state with next wakeup
            next_wakeup = (loop_start + timedelta(seconds=poll_seconds)).isoformat()
            state.update(
                next_wakeup_at=next_wakeup,
                provider_status=cooldown.state.get("last_status"),
                stage=operational_status.get("stage", "production_shadow"),
                staking_enabled=False,
                terminal_state="PRODUCTION_SHADOW_RUNNING" if operational_status.get("stage") == "production_shadow" else operational_status.get("stage", "BLOCKED").upper(),
            )

            # Read failed gates from evidence status if available
            if EVIDENCE_STATUS_PATH.exists():
                try:
                    ev = json.loads(EVIDENCE_STATUS_PATH.read_text())
                    state.update(
                        failed_gates=ev.get("failed_gates", []),
                        next_action=ev.get("next_action", ""),
                    )
                except (json.JSONDecodeError, OSError):
                    pass

            logger.info(f"  next wakeup: {next_wakeup}")
            append_live_slate_history({
                "timestamp": loop_start.isoformat(),
                "dry_run": bool(args.dry_run),
                "provider": args.provider,
                "prediction_mode": prediction_mode,
                "stage": operational_status.get("stage"),
                "staking_enabled": False,
                "pending_close_rows": inspect_ledger().get("pending_close", 0),
                "pending_settle_rows": inspect_ledger().get("pending_settle", 0),
                "settled_live_class_a_rows": operational_status.get("settled_live_class_a_rows", 0),
                "gold_real_clv_rows": operational_status.get("gold_real_clv_rows", 0),
                "unique_live_slates": operational_status.get("unique_live_slates", 0),
                "failed_gate_count": len(operational_status.get("failed_gates", [])),
                "kill_switch_count": len(operational_status.get("kill_switches", [])),
                "next_action": operational_status.get("next_action"),
            })

            # Exit conditions
            if args.once:
                logger.info("--once mode, exiting after single loop")
                break

            if max_runtime and (datetime.now(timezone.utc) - start_time) > max_runtime:
                logger.info(f"max runtime {args.max_runtime_hours}h reached, exiting")
                break

            # Sleep
            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received, shutting down gracefully")
    finally:
        state.update(last_loop_at=datetime.now(timezone.utc).isoformat())
        state.save()


# ─── Entry Point ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Production-Shadow Daemon")
    parser.add_argument("--date", default=None, help="Override date (YYYY-MM-DD)")
    parser.add_argument("--timezone", default="America/New_York", help="Timezone for scheduling")
    parser.add_argument("--poll-minutes", type=int, default=5, help="Poll interval in minutes")
    parser.add_argument("--provider", default="sportsgameodds", help="Provider name")
    parser.add_argument("--prediction-mode", default="real", choices=["real", "simulated_for_pipeline_test"])
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--once", action="store_true", help="Run one loop and exit")
    parser.add_argument("--max-runtime-hours", type=float, default=0, help="Max runtime (0=unlimited)")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--state-path", default=str(STATE_PATH), help="State file path")
    parser.add_argument("--include-cross-sport-market-lapse", action="store_true", help="Include optional cross-sport market-state rows for CLV surrogate training only")

    args = parser.parse_args()

    # Setup
    logger = setup_logging(args.log_level)

    # Acquire lock
    if not acquire_lock(logger):
        sys.exit(1)
    atexit.register(release_lock)

    # Handle signals
    def _signal_handler(sig, frame):
        logger.info("Signal received, shutting down")
        release_lock()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    # Run
    daemon_loop(args, logger)

    # Cleanup
    release_lock()
    logger.info("daemon stopped")


if __name__ == "__main__":
    main()
