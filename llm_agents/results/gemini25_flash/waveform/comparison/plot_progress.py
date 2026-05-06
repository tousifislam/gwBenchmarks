import matplotlib.pyplot as plt
import os
import json
import numpy as np
import gwbenchmarks.plot_settings as plot_settings

def plot_progress():
    plot_settings.apply()
    
    base_dir = "llm_agents/results/gemini25_flash/waveform/models/"
    comparison_dir = "llm_agents/results/gemini25_flash/waveform/comparison/"
    
    model_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    approach_numbers = []
    losses = []
    runtimes = []
    approach_names = []

    for model_dir_name in model_dirs:
        scorecard_path = os.path.join(base_dir, model_dir_name, "scorecard.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                scorecard = json.load(f)
            
            approach_numbers.append(scorecard["approach_number"])
            losses.append(scorecard["loss"])
            runtimes.append(scorecard["runtime_ms"])
            approach_names.append(model_dir_name.replace("NN_", "")) # Assuming "NN_" prefix if any

    # Plot Loss vs. Approach Number
    plt.figure(figsize=plot_settings.figsize())
    plt.plot(approach_numbers, losses, marker='o', linestyle='-')
    plt.xlabel("Approach Number")
    plt.ylabel("Loss (Mean FD Mismatch)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for i, txt in enumerate(approach_names):
        plt.annotate(txt, (approach_numbers[i], losses[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "progress_loss.png"))
    plt.savefig(os.path.join(comparison_dir, "progress_loss.pdf"))
    plt.close()

    # Plot Runtime vs. Approach Number
    plt.figure(figsize=plot_settings.figsize())
    plt.plot(approach_numbers, runtimes, marker='o', linestyle='-')
    plt.xlabel("Approach Number")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, linestyle='--', alpha=0.6)

    for i, txt in enumerate(approach_names):
        plt.annotate(txt, (approach_numbers[i], runtimes[i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "progress_runtime.png"))
    plt.savefig(os.path.join(comparison_dir, "progress_runtime.pdf"))
    plt.close()

    print("Progress plots generated.")

if __name__ == "__main__":
    plot_progress()
