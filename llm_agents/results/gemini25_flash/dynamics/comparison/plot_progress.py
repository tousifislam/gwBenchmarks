
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import glob

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import gwbenchmarks.plot_settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
COMPARISON_DIR = BASE_DIR

os.makedirs(COMPARISON_DIR, exist_ok=True)

def plot_progress(benchmark_name="dynamics"):
    scorecards = []
    
    # Collect all scorecard.json files
    for model_dir in glob.glob(os.path.join(MODELS_DIR, '*')):
        scorecard_path = os.path.join(model_dir, 'scorecard.json')
        if os.path.exists(scorecard_path):
            with open(scorecard_path, 'r') as f:
                scorecards.append(json.load(f))

    if not scorecards:
        print("No scorecards found to plot progress.")
        return

    # Sort scorecards by approach number for consistent plotting
    scorecards.sort(key=lambda x: x.get('approach_number', 0))

    approaches = [s['approach'] for s in scorecards]
    losses = [s['loss'] for s in scorecards]
    runtimes = [s['runtime_ms'] for s in scorecards]

    # Create progress plot for loss
    plt.figure(figsize=(10, 6))
    plt.plot(approaches, losses, marker='o', linestyle='-')
    plt.xlabel('Approach')
    plt.ylabel(f'{benchmark_name.capitalize()} Loss (RMS Relative Error)')
    plt.title(f'Progress of {benchmark_name.capitalize()} Benchmark Loss')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'progress_loss.png'))
    plt.savefig(os.path.join(COMPARISON_DIR, 'progress_loss.pdf'))
    plt.close()

    # Create progress plot for runtime
    plt.figure(figsize=(10, 6))
    plt.plot(approaches, runtimes, marker='o', linestyle='-')
    plt.xlabel('Approach')
    plt.ylabel('Runtime (ms)')
    plt.title(f'Progress of {benchmark_name.capitalize()} Benchmark Runtime')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'progress_runtime.png'))
    plt.savefig(os.path.join(COMPARISON_DIR, 'progress_runtime.pdf'))
    plt.close()

    print("Progress plots generated: progress_loss.png/pdf, progress_runtime.png/pdf")

if __name__ == "__main__":
    plot_progress()
