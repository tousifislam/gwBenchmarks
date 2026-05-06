
import os
import json
import matplotlib.pyplot as plt
from gwbenchmarks import plot_settings

def main():
    MODELS_DIR = "llm_agents/results/gemini25_pro/waveform/models"
    COMPARISON_DIR = "llm_agents/results/gemini25_pro/waveform/comparison"
    
    scorecards = []
    for model_dir in os.listdir(MODELS_DIR):
        scorecard_path = os.path.join(MODELS_DIR, model_dir, "scorecard.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                scorecard = json.load(f)
                scorecards.append(scorecard)
                
    if not scorecards:
        print("No scorecards found.")
        return
        
    # Sort by approach number
    scorecards.sort(key=lambda x: x["approach_number"])
    
    approach_numbers = [s["approach_number"] for s in scorecards]
    losses = [s["loss"] for s in scorecards]
    
    # Apply plot settings
    plot_settings.apply()
    
    # Create plot
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=1.5))
    
    ax.plot(approach_numbers, losses, marker='o', linestyle='-', color=plot_settings.COLORS["blue"])
    
    ax.set_xlabel("Approach Number")
    ax.set_ylabel("Mismatch")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    
    plt.savefig(os.path.join(COMPARISON_DIR, "progress.png"))
    plt.savefig(os.path.join(COMPARISON_DIR, "progress.pdf"))
    
    print("Progress plot updated.")

if __name__ == "__main__":
    main()
