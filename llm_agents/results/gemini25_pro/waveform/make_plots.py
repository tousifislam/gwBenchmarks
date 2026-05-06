
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from gwbenchmarks.plot_settings import apply as setup_plot_style

def get_model_dirs(base_dir="models"):
    return [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def load_scorecards(model_dirs):
    scorecards = []
    for model_dir in model_dirs:
        scorecard_path = os.path.join(model_dir, "scorecard.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                scorecards.append(json.load(f))
    return scorecards

def plot_progress(scorecards, output_dir="comparison"):
    setup_plot_style()
    plt.figure(figsize=(10, 6))
    
    scorecards.sort(key=lambda x: x["approach_number"])
    
    losses = [s["loss"] for s in scorecards]
    labels = [f'{s["approach_number"]}: {s["approach"]}' for s in scorecards]
    
    plt.plot(range(len(losses)), losses, 'o-', label="Loss")
    
    plt.xlabel("Approach number")
    plt.ylabel("Loss (mismatch)")
    plt.xticks(range(len(losses)), labels, rotation=45, ha="right")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(os.path.join(output_dir, "progress.png"))
    plt.savefig(os.path.join(output_dir, "progress.pdf"))
    plt.close()

def main():
    model_dirs = get_model_dirs()
    scorecards = load_scorecards(model_dirs)
    
    plot_progress(scorecards)
    
    # The other plots will be implemented later
    # plot_error_histograms(scorecards)
    # plot_pareto(scorecards)
    # plot_loss_comparison(scorecards)

if __name__ == "__main__":
    main()
