# MLB latent parlay training

This image trains the MLB H-over-0.5 latent leg ensemble and the two-to-four-leg set-attention model on an NVIDIA GPU. Training excludes same-game outcome-derived projection fields and splits complete slate dates into development, calibration, and locked model holdout periods.

From the repository root on Windows:

```powershell
./sports/mlb/training/latent-parlay/train.ps1 -BeforeDate 2026-08-06
```

On Linux:

```bash
./sports/mlb/training/latent-parlay/train.sh 2026-08-06
```

The container writes a framework-free JSON artifact to `sports/mlb/data/predictions/calibration/latent_parlay_model_2026.json` and a readable validation report to `sports/mlb/data/predictions/backtests/latent_parlay_model_2026_report.json`. The daily CPU selector loads the JSON with NumPy; PyTorch and CUDA are not production dependencies.

The artifact is shadow evidence only. Its historical rows use synthetic H 0.5 lines without complete executable price history, so the report supports hit-chain ranking diagnostics, not ROI certification.
