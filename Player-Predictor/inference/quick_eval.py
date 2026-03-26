#!/usr/bin/env python3
"""
Quick evaluation - just run the training script's main but stop after loading model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Run the training script to initialize everything
from training.integrate_moe_improvements import MoEImprovedTrainer

print("Initializing trainer...")
trainer = MoEImprovedTrainer()

# Call train() which will load data and initialize everything
# We'll modify it to just evaluate
print("\nCalling trainer.train() to load data and build model...")
trainer.train()
