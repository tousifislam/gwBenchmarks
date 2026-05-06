import os
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add project root to import gwbenchmarks
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))
from gwbenchmarks import plot_settings

def update_comparison_plots():
    """
    Generates and updates all comparison plots and data files.
    """
    
    models_dir = "llm_agents/results/gemini25_pro/dynamics/models"
    comparison_dir = "llm_agents/results/gemini25_pro/dynamics/comparison"
    
    # --- Load data from all models ---
    all_scorecards = []
    all_errors = {}
    
    model_folders = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]
    
    for model_name in sorted(model_folders):
        scorecard_path = os.path.join(models_dir, model_name, "scorecard.json")
        errors_path = os.path.join(models_dir, model_name, "errors.json")
        
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                scorecard = json.load(f)
                all_scorecards.append(scorecard)
                
            if os.path.exists(errors_path):
                with open(errors_path, "r") as f:
                    errors = json.load(f)
                    all_errors[model_name] = errors["validation_errors"]

    # --- Save raw error data ---
    error_data_path = os.path.join(comparison_dir, "error_data.json")
    with open(error_data_path, "w") as f:
        json.dump(all_errors, f, indent=4)
    print(f"Saved error data to {error_data_path}")

    # Apply plot style
    plot_settings.apply()

    # --- Generate Progress Plot (Loss vs. Approach) ---
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1.5, aspect=0.6))
    
    approach_numbers = [s["approach_number"] for s in all_scorecards]
    losses = [s["loss"] for s in all_scorecards]
    labels = [s["approach"] for s in all_scorecards]

    ax.plot(approach_numbers, losses, 'o-', label="Validation Loss")
    
    for i, txt in enumerate(labels):
        ax.annotate(txt, (approach_numbers[i], losses[i]), textcoords="offset points", xytext=(0,10), ha='center')

    ax.set_xlabel("Approach Number")
    ax.set_ylabel("RMS Relative Error")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--")
    
    progress_plot_path_png = os.path.join(comparison_dir, "progress.png")
    progress_plot_path_pdf = os.path.join(comparison_dir, "progress.pdf")
    plt.savefig(progress_plot_path_png)
    plt.savefig(progress_plot_path_pdf)
    print(f"Saved progress plot to {progress_plot_path_png} and .pdf")
    plt.close()

    # --- Generate Error Histograms ---
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1.5, aspect=0.6))
    
    for model_name, errors in all_errors.items():
        ax.hist(errors, bins=50, alpha=0.7, label=model_name, density=True)
        
    ax.set_xlabel("Per-Sample RMS Relative Error")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls="--")
    
    hist_plot_path_png = os.path.join(comparison_dir, "error_histograms.png")
    hist_plot_path_pdf = os.path.join(comparison_dir, "error_histograms.pdf")
    plt.savefig(hist_plot_path_png)
    plt.savefig(hist_plot_path_pdf)
    print(f"Saved error histograms to {hist_plot_path_png} and .pdf")
    plt.close()

if __name__ == "__main__":
    update_comparison_plots()
