#!/bin/bash
# Example script to run LLM persuasion experiments

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-your-key-here"

# Create output directory
mkdir -p persuasion_results

# Run basic experiment
echo "Running basic persuasion experiment..."
python llm_persuasion_experiment.py \
    --models gpt-4o-mini claude-3-sonnet \
    --iterations 5 \
    --n_dilemmas 3 \
    --out_dir persuasion_results/basic_experiment \
    --max_rounds 3 \
    --identity_disclosure_prob 0.5 \
    --fake_identity_prob 0.3 \
    --seed 42

# Run extended experiment with more models
echo "Running extended experiment..."
python llm_persuasion_experiment.py \
    --models gpt-4o-mini claude-3-sonnet llama-3.1-8b-instruct \
    --iterations 10 \
    --n_dilemmas 5 \
    --out_dir persuasion_results/extended_experiment \
    --max_rounds 5 \
    --identity_disclosure_prob 0.6 \
    --fake_identity_prob 0.2 \
    --seed 123

# Run identity disclosure focused experiment
echo "Running identity disclosure experiment..."
python llm_persuasion_experiment.py \
    --models gpt-4o-mini claude-3-sonnet \
    --iterations 15 \
    --n_dilemmas 3 \
    --out_dir persuasion_results/identity_experiment \
    --max_rounds 3 \
    --identity_disclosure_prob 0.8 \
    --fake_identity_prob 0.4 \
    --seed 456

echo "All experiments completed!"
echo "Check the results in persuasion_results/ directory"
