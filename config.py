#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for LLM Persuasion Experiment
"""

import os
from pathlib import Path

# API Configuration
OPENROUTER_API_KEY = "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"

# Set environment variable
os.environ["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY

# Experiment Configuration
DEFAULT_MODELS = ["gpt-4o-mini", "claude-3-sonnet", "llama-3.1-8b-instruct"]
DEFAULT_ITERATIONS = 5
DEFAULT_N_DILEMMAS = 3
DEFAULT_MAX_ROUNDS = 3

# Output directories
RESULTS_DIR = Path("persuasion_results")
CONSISTENCY_DIR = Path("runs_consistency")

# Create directories if they don't exist
RESULTS_DIR.mkdir(exist_ok=True)
CONSISTENCY_DIR.mkdir(exist_ok=True)

print("✅ Configuration loaded successfully!")
print(f"✅ API Key set: {OPENROUTER_API_KEY[:15]}...")
print(f"✅ Results directory: {RESULTS_DIR}")
print(f"✅ Consistency directory: {CONSISTENCY_DIR}")
