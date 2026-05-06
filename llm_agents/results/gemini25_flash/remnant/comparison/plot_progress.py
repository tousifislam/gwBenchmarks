import matplotlib.pyplot as plt
import os
import json
import numpy as np
import gwbenchmarks.plot_settings as plot_settings
import pandas as pd

def plot_progress():
    plot_settings.apply()
    
    base_dir = "llm_agents/results/gemini25_flash/remnant/models/"
    comparison_dir = "llm_agents/results/gemini25_flash/remnant/comparison/"
    
    model_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    all_results = []
    for model_dir_name in model_dirs:
        scorecard_path = os.path.join(base_dir, model_dir_name, "scorecard.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                scorecard = json.load(f)
            
            # Skip blocked approaches
            if scorecard.get("loss") is None or scorecard.get("runtime_ms") is None:
                continue

            all_results.append({
                "model_name": model_dir_name,
                "approach_number": scorecard["approach_number"],
                "loss": scorecard["loss"],
                "runtime_ms": scorecard["runtime_ms"],
                "approach_category": scorecard["approach"],
                "parameterization": scorecard["parameterization"],
                "loss_components": scorecard.get("loss_components", {})
            })

    results_df = pd.DataFrame(all_results)
    if results_df.empty:
        print("No valid results to process.")
        return

    # Sort by approach number for consistent plotting
    results_df = results_df.sort_values(by="approach_number").reset_index(drop=True)

    # --- Generate progress plots ---
    plt.figure(figsize=plot_settings.figsize())
    plt.plot(results_df["approach_number"], results_df["loss"], marker='o', linestyle='-')
    plt.xlabel("Approach Number")
    plt.ylabel("Loss (NRMSE vf_mag)")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for i, txt in enumerate(results_df["model_name"].apply(lambda x: x.replace("GPR_","").replace("_Raw",""))):
        plt.annotate(txt, (results_df["approach_number"][i], results_df["loss"][i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "progress_loss.png"))
    plt.savefig(os.path.join(comparison_dir, "progress_loss.pdf"))
    plt.close()

    plt.figure(figsize=plot_settings.figsize())
    plt.plot(results_df["approach_number"], results_df["runtime_ms"], marker='o', linestyle='-')
    plt.xlabel("Approach Number")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, linestyle='--', alpha=0.6)
    for i, txt in enumerate(results_df["model_name"].apply(lambda x: x.replace("GPR_","").replace("_Raw",""))):
        plt.annotate(txt, (results_df["approach_number"][i], results_df["runtime_ms"][i]), textcoords="offset points", xytext=(0,5), ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "progress_runtime.png"))
    plt.savefig(os.path.join(comparison_dir, "progress_runtime.pdf"))
    plt.close()

    print("Progress plots generated.")

if __name__ == "__main__":
    plot_progress()
