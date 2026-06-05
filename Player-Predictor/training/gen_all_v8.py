"""
Master generator for all NBA v8 files.
Run: python Player-Predictor/training/gen_all_v8.py
"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
INFERENCE = ROOT / "inference"
SCRIPTS   = ROOT / "scripts"

def w(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  wrote {path.relative_to(ROOT)}")

print("=== NBA v8 Generator ===")
print('appended')
