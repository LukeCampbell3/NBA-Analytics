#!/usr/bin/env python3
"""
MLB Production-Shadow Daemon

Continuously runs the MLB production-shadow evidence cycle:
  predecision → close → settle → status → evidence rebuild → training exports

Usage:
  python sports/mlb/predictions/scripts/run_mlb_production_shadow_daemon.py
  python sports/mlb/predictions/scripts/run_mlb_production_shadow_daemon.py --once
  python sports/mlb/predictions/scripts/run_mlb_production_shadow_daemon.py --once --dry-run

Safety:
  - staking_enabled = false always
  - live_action_enabled = false always
  - never auto-enables staking
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
from typing import Any, Dict, Optional

WORKSPACE = Path(__file__).resolve().parents[4]

# Load .env credentials early
sys.path.insert(0, str(WORKSPACE / "sports" / "mlb" / "predictions" / "odds"))
try:
    from provider_credentials import load_repo_env
    load_repo_env()
except ImportError:
    pass

RUNNER_SCRIPT = WORKSPACE / "sports" / "mlb" / "predictions" / "scripts" / "run_mlb_production_shadow_daily.py"
MLB_SHADOW_DIR = WORKSPACE / "sports" / "mlb" / "validation" / "production_shadow"
STATE_PATH = MLB_SHADOW_DIR / "daemon_state.json"
PROVIDER_STATE_PATH = MLB_SHADOW_DIR / "provider_runtime_state.json"
LOCK_PATH = MLB_SHADOW_DIR / "mlb_production_shadow_daemon.lock"
LOG_DIR = MLB_SHADOW_DIR / "logs"
LEDGER_PATH = MLB_SHADOW_DIR / "mlb_live_ledger.csv"
EXPORT_DIR = MLB_SHADOW_DIR / "training_exports"
PRODUCTION_STATUS_PATH = MLB_SHADOW_DIR / "production_status.json"
LIVE_SLATE_HISTORY_PATH = MLB_SHADOW_DIR / "live_slate_history.csv"
DATA_MODE_STATUS_PATH = MLB_SHADOW_DIR / "data_mode_status.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("mlb_shadow_daemon")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(asctime)s] MLB %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    log_file = LOG_DIR / f"mlb_production_shadow_daemon_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)

    return logger


def acquire_lock(logger: logging.Logger) -> bool:
    if LOCK_PATH.exists():
        try:
            lock_data = json.loads(LOCK_PATH.read_text())
            pid = lock_data.get("pid", 0)
            if pid and _pid_alive(pid):
                logger.error(f"Another MLB daemon running (PID {pid}). Exiting.")
                return False
            else:
                logger.warning(f"Stale lock (PID {pid}). Removing.")
                LOCK_PATH.unlink()
        except (json.JSONDecodeError, OSError):
            LOCK_PATH.unlink(missing_ok=True)

    lock_data = {
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sport": "MLB",
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock_data, indent=2))
    return True


def release_lock():
    LOCK_PATH.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)
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


class MlbProviderCooldown:
    """Manages MLB provider request cooldown."""

    def __init__(self):
        self.state = self._load()

    def _load(self) -> Dict[str, Any]:
        if PROVIDER_STATE_PATH.exists():
            try:
                return json.loads(PROVIDER_STATE_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "provider": "sportsgameodds",
            "sport": "MLB",
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
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        if self.state.get("date") != today:
            self.state["date"] = today
            self.state["requests_today"] = 0
            self.state["errors_today"] = 0

        blocked = self.state.get("blocked_until")
        if blocked:
            if now < datetime.fromisoformat(blocked):
                return False

        last = self.state.get("last_attempt_at")
        if last:
            cooldown = self.state.get("cooldown_seconds", 300)
            if (now - datetime.fromisoformat(last)).total_seconds() < cooldown:
                return False
        return True

    def record_attempt(self, status: str):
        now = datetime.now(timezone.utc)
        self.state["last_attempt_at"] = now.isoformat()
        self.state["last_status"] = status
        self.state["requests_today"] = self.state.get("requests_today", 0) + 1

        if status == "success":
            self.state["last_success_at"] = now.isoformat()
            self.state["cooldown_seconds"] = 300
            self.state["blocked_until"] = None
        elif status == "no_props":
            self.state["cooldown_seconds"] = 900
        elif status in ("missing_credentials", "quota_exhausted"):
            self.state["blocked_until"] = (now + timedelta(hours=1)).isoformat()
            self.state["errors_today"] = self.state.get("errors_today", 0) + 1
        elif status == "api_error":
            self.state["cooldown_seconds"] = 600
            self.state["errors_today"] = self.state.get("errors_today", 0) + 1
        else:
            self.state["cooldown_seconds"] = 300
        self.save()


class MlbDaemonState:
    """Manages MLB daemon state."""

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
            "sport": "MLB",
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
            "gold_real_clv_rows": 0,
            "failed_gates": [],
            "next_action": None,
            "next_wakeup_at": None,
            "stage": "production_shadow_accumulating",
            "terminal_state": "MLB_WAITING_FOR_FRESH_PROPS",
        }

    def save(self):
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATE_PATH.write_text(json.dumps(self.state, indent=2, default=str))

    def update(self, **kwargs):
        self.state.update(kwargs)
        self.save()


def run_phase(phase: str, prediction_mode: str = "real", dry_run: bool = False,
              provider: str = "sportsgameodds",
              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Run a single phase via subprocess."""
    cmd = [sys.executable, str(RUNNER_SCRIPT), "--phase", phase, "--provider", provider]

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
        metrics = _parse_phase_output(result.stdout or "", phase)
        metrics["exit_code"] = result.returncode
        metrics["phase"] = phase
        if result.returncode != 0 and logger:
            logger.warning(f"  {phase} exited with code {result.returncode}")
        return metrics
    except subprocess.TimeoutExpired:
        if logger:
            logger.error(f"  {phase} timed out")
        return {"phase": phase, "status": "timeout", "exit_code": -1}
    except Exception as e:
        if logger:
            logger.error(f"  {phase} error: {e}")
        return {"phase": phase, "status": "error", "exit_code": -1}


def _parse_phase_output(stdout: str, phase: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {"status": "unknown"}
    for line in stdout.splitlines():
        s = line.strip()
        if "Appended rows:" in s:
            try:
                metrics["appended_rows"] = int(s.split("Appended rows:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        elif "Decisions settled:" in s:
            try:
                metrics["settled_rows"] = int(s.split("Decisions settled:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Closed decisions:" in s:
            try:
                metrics["closed_rows"] = int(s.split(":")[-1].strip())
            except (ValueError, IndexError):
                pass
        elif "NO FRESH ODDS" in s:
            metrics["status"] = "no_fresh_odds"
        elif "Fresh odds obtained" in s:
            metrics["status"] = "fresh_odds_obtained"
        elif "PREDECISION PHASE COMPLETE" in s:
            metrics["status"] = "success"
        elif "CLOSE PHASE COMPLETE" in s:
            metrics["status"] = "success"
        elif "SETTLE PHASE COMPLETE" in s:
            metrics["status"] = "success"
        elif "No pending decisions" in s:
            metrics["status"] = "no_pending"
    return metrics


def inspect_mlb_ledger() -> Dict[str, Any]:
    """Inspect the MLB live ledger."""
    import pandas as pd
    if not LEDGER_PATH.exists():
        return {"total": 0, "pending_close": 0, "pending_settle": 0, "settled": 0, "gold_real_clv_rows": 0}
    try:
        df = pd.read_csv(LEDGER_PATH)
    except Exception:
        return {"total": 0, "pending_close": 0, "pending_settle": 0, "settled": 0, "gold_real_clv_rows": 0}
    if df.empty:
        return {"total": 0, "pending_close": 0, "pending_settle": 0, "settled": 0, "gold_real_clv_rows": 0}

    pending_close = df[
        ((df["close_snapshot_id"].isna()) | (df["close_snapshot_id"] == "")) &
        (df["entry_snapshot_id"].notna()) & (df["entry_snapshot_id"] != "")
    ]
    pending_settle = df[
        (df["close_snapshot_id"].notna()) & (df["close_snapshot_id"] != "") &
        ((df["settled_at"].isna()) | (df["settled_at"] == ""))
    ]
    settled = df[(df["settled_at"].notna()) & (df["settled_at"] != "")]
    class_a_settled = settled[settled.get("class_a_candidate", pd.Series([False] * len(settled))) == True] if not settled.empty else pd.DataFrame()
    gold_real = settled[
        (settled.get("side_aware_prob_clv", pd.Series([pd.NA] * len(settled))).notna()) &
        (settled.get("close_snapshot_id", pd.Series([""] * len(settled))).fillna("").astype(str) != "")
    ] if not settled.empty else pd.DataFrame()

    return {
        "total": len(df),
        "pending_close": len(pending_close),
        "pending_settle": len(pending_settle),
        "settled": len(settled),
        "settled_class_a": len(class_a_settled),
        "gold_real_clv_rows": len(gold_real),
        "unique_slates": int(settled["game_id"].nunique()) if not settled.empty else 0,
    }


def export_mlb_training_datasets(logger: Optional[logging.Logger] = None):
    """Export MLB training datasets."""
    import pandas as pd
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Run historical outcome export
    sys.path.insert(0, str(MLB_SHADOW_DIR))
    try:
        from mlb_sparse_season_data_mode import run_mlb_sparse_season_exports
        report = run_mlb_sparse_season_exports()
        if logger:
            logger.info(f"  sparse exports: mode={report.get('mode')} historical={report.get('historical_mlb_rows')}")
    except Exception as exc:
        if logger:
            logger.warning(f"  sparse export failed: {exc}")


def evaluate_mlb_operational_state(logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Evaluate MLB gates and write status."""
    sys.path.insert(0, str(MLB_SHADOW_DIR))
    from production_gate_evaluator import MlbProductionGateEvaluator
    from staking_controller import MlbStakingController

    evaluator = MlbProductionGateEvaluator()
    gate_result = evaluator.evaluate_all()
    staking = MlbStakingController().resolve(gate_result)

    metrics = gate_result.get("metrics", {})
    failed = gate_result.get("failed_gates", [])
    stage = gate_result.get("stage", "production_shadow_accumulating")

    status = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "sport": "MLB",
        "system_mode": "production_shadow",
        "stage": stage,
        "production_status": stage,
        "staking_enabled": False,
        "live_action_enabled": False,
        "settled_live_class_a_rows": metrics.get("settled_live_class_a_rows", 0),
        "gold_real_clv_rows": metrics.get("gold_real_clv_rows", 0),
        "unique_live_slates": metrics.get("unique_live_slates", 0),
        "brier": metrics.get("brier"),
        "bss": metrics.get("bss"),
        "roi": metrics.get("roi"),
        "mean_clv": metrics.get("mean_clv"),
        "positive_clv_rate": metrics.get("positive_clv_rate"),
        "failed_gates": failed,
        "next_action": "accumulate MLB live evidence",
        "reason": "MLB system must collect gold live CLV + settled outcomes before staking",
    }

    PRODUCTION_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRODUCTION_STATUS_PATH.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")

    if logger:
        logger.info(f"  stage={stage} staking=False failed_gates={len(failed)}")
    return status


def daemon_loop(args, logger: logging.Logger):
    """Main MLB daemon loop."""
    cooldown = MlbProviderCooldown()
    state = MlbDaemonState()
    state.update(daemon_started_at=datetime.now(timezone.utc).isoformat())

    poll_seconds = args.poll_minutes * 60
    start_time = datetime.now(timezone.utc)

    logger.info(f"MLB daemon started | poll={args.poll_minutes}m | dry_run={args.dry_run}")

    try:
        while True:
            loop_start = datetime.now(timezone.utc)
            state.update(last_loop_at=loop_start.isoformat(), loop_count=state.state["loop_count"] + 1)

            ledger_info = inspect_mlb_ledger()
            state.update(
                pending_close_rows=ledger_info["pending_close"],
                pending_settle_rows=ledger_info["pending_settle"],
                settled_class_a_rows=ledger_info.get("settled_class_a", 0),
                gold_real_clv_rows=ledger_info.get("gold_real_clv_rows", 0),
                unique_live_slates=ledger_info.get("unique_slates", 0),
                mode="production_shadow",
                staking_enabled=False,
            )

            ran_something = False

            # PREDECISION
            if cooldown.can_request():
                result = run_phase("predecision", "real", args.dry_run, args.provider, logger)
                state.update(last_predecision_at=loop_start.isoformat(), last_phase_result=result.get("status"))

                if not args.dry_run:
                    if result.get("status") == "no_fresh_odds":
                        cooldown.record_attempt("no_props")
                    elif result.get("status") in ("success", "fresh_odds_obtained"):
                        cooldown.record_attempt("success")
                        ran_something = True
                    elif result.get("exit_code", 0) != 0:
                        cooldown.record_attempt("api_error")
                    else:
                        cooldown.record_attempt(result.get("status", "unknown"))

            # CLOSE
            ledger_info = inspect_mlb_ledger()
            if ledger_info["pending_close"] > 0:
                result = run_phase("close", "real", args.dry_run, args.provider, logger)
                state.update(last_close_at=loop_start.isoformat())
                if result.get("closed_rows", 0):
                    ran_something = True

            # SETTLE
            ledger_info = inspect_mlb_ledger()
            if ledger_info["pending_settle"] > 0:
                result = run_phase("settle", "real", args.dry_run, args.provider, logger)
                state.update(last_settle_at=loop_start.isoformat())
                if result.get("settled_rows", 0):
                    ran_something = True

            # STATUS
            run_phase("status", "real", args.dry_run, args.provider, logger)
            state.update(last_status_at=loop_start.isoformat())

            # EVIDENCE REBUILD + EXPORT
            if ran_something or state.state["loop_count"] % 6 == 0 or args.once:
                export_mlb_training_datasets(logger)
                state.update(last_evidence_rebuild_at=loop_start.isoformat())

            operational_status = evaluate_mlb_operational_state(logger)

            # Determine terminal state
            if ran_something:
                terminal = "MLB_PRODUCTION_SHADOW_RUNNING"
            elif cooldown.state.get("last_status") == "no_props":
                terminal = "MLB_WAITING_FOR_FRESH_PROPS"
            elif cooldown.state.get("last_status") in ("missing_credentials", "api_error"):
                terminal = "MLB_EXTERNAL_RESOURCE_BLOCKER"
            else:
                terminal = "MLB_WAITING_FOR_FRESH_PROPS"

            next_wakeup = (loop_start + timedelta(seconds=poll_seconds)).isoformat()
            state.update(
                next_wakeup_at=next_wakeup,
                provider_status=cooldown.state.get("last_status"),
                stage=operational_status.get("stage", "production_shadow_accumulating"),
                staking_enabled=False,
                terminal_state=terminal,
                failed_gates=operational_status.get("failed_gates", []),
                next_action=operational_status.get("next_action", ""),
            )

            logger.info(f"  terminal={terminal} next_wakeup={next_wakeup}")

            # Append to slate history
            _append_slate_history(loop_start, args, operational_status)

            if args.once:
                logger.info("--once mode, exiting after single loop")
                break

            if args.max_runtime_hours > 0:
                if (datetime.now(timezone.utc) - start_time) > timedelta(hours=args.max_runtime_hours):
                    logger.info("max runtime reached, exiting")
                    break

            time.sleep(poll_seconds)

    except KeyboardInterrupt:
        logger.info("Ctrl+C received, shutting down")
    finally:
        state.update(last_loop_at=datetime.now(timezone.utc).isoformat())
        state.save()


def _append_slate_history(loop_start: datetime, args, status: Dict[str, Any]):
    import pandas as pd
    row = pd.DataFrame([{
        "timestamp": loop_start.isoformat(),
        "sport": "MLB",
        "dry_run": bool(args.dry_run),
        "provider": args.provider,
        "stage": status.get("stage"),
        "staking_enabled": False,
        "settled_live_class_a_rows": status.get("settled_live_class_a_rows", 0),
        "gold_real_clv_rows": status.get("gold_real_clv_rows", 0),
        "unique_live_slates": status.get("unique_live_slates", 0),
        "failed_gate_count": len(status.get("failed_gates", [])),
    }])
    LIVE_SLATE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LIVE_SLATE_HISTORY_PATH.exists():
        row.to_csv(LIVE_SLATE_HISTORY_PATH, mode="a", header=False, index=False)
    else:
        row.to_csv(LIVE_SLATE_HISTORY_PATH, index=False)


def main():
    parser = argparse.ArgumentParser(description="MLB Production-Shadow Daemon")
    parser.add_argument("--poll-minutes", type=int, default=5)
    parser.add_argument("--provider", default="sportsgameodds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--max-runtime-hours", type=float, default=0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logger = setup_logging(args.log_level)

    if not acquire_lock(logger):
        sys.exit(1)
    atexit.register(release_lock)

    def _signal_handler(sig, frame):
        logger.info("Signal received, shutting down")
        release_lock()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    daemon_loop(args, logger)
    release_lock()
    logger.info("MLB daemon stopped")


if __name__ == "__main__":
    main()
