"""Command line entry points for PAR/PAR-F builds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import build_parf_validation_report, build_player_metrics, prove_metrics_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and validate PAR/PAR-F player metrics.")
    sub = parser.add_subparsers(dest="command", required=True)

    build_par = sub.add_parser("build-par", help="Build current-season PAR artifacts.")
    build_par.add_argument("--season", required=True)
    build_par.add_argument("--out", required=True)
    build_par.add_argument("--player-limit", type=int, default=None)
    build_par.add_argument("--copy-to-web", action="store_true")

    build_parf = sub.add_parser("build-par-f", help="Build PAR-F forecast artifacts.")
    build_parf.add_argument("--season-from", required=True)
    build_parf.add_argument("--season-to", required=True)
    build_parf.add_argument("--out", required=True)
    build_parf.add_argument("--player-limit", type=int, default=None)
    build_parf.add_argument("--copy-to-web", action="store_true")

    build_all = sub.add_parser("build-player-metrics", help="Build PAR, PAR-F, and frontend JSON artifacts.")
    build_all.add_argument("--season", required=True)
    build_all.add_argument("--forecast-season", required=True)
    build_all.add_argument("--out", required=True)
    build_all.add_argument("--player-limit", type=int, default=None)
    build_all.add_argument("--copy-to-web", action="store_true")

    prove = sub.add_parser("prove-par-product", help="Validate a built PAR product directory.")
    prove.add_argument("--metrics-dir", required=True)

    validate = sub.add_parser("validate-par-f", help="Create PAR-F validation report placeholders/backtest output.")
    validate.add_argument("--metrics-dir", required=True)
    validate.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-par":
        result = build_player_metrics(args.season, args.season, Path(args.out), player_limit=args.player_limit, copy_to_web=args.copy_to_web)
    elif args.command == "build-par-f":
        result = build_player_metrics(args.season_from, args.season_to, Path(args.out), player_limit=args.player_limit, copy_to_web=args.copy_to_web)
    elif args.command == "build-player-metrics":
        result = build_player_metrics(args.season, args.forecast_season, Path(args.out), player_limit=args.player_limit, copy_to_web=args.copy_to_web)
    elif args.command == "prove-par-product":
        result = prove_metrics_dir(Path(args.metrics_dir))
    elif args.command == "validate-par-f":
        result = build_parf_validation_report(Path(args.metrics_dir), Path(args.out))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
