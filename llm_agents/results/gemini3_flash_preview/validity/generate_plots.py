import json
import os
import numpy as np
import matplotlib.pyplot as plt
from gwbenchmarks.plot_settings import apply

apply()

def generate_plots():
    results_dir = "llm_agents/results/gemini3_flash_preview/validity/"
    models_dir = os.path.join(results_dir, "models")
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    approaches = []
    for d in sorted(os.listdir(models_dir)):
        scorecard_path = os.path.join(models_dir, d, "scorecard.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                approaches.append(json.load(f))
    
    # Summary table
    approaches.sort(key=lambda x: x["loss"])
    with open(os.path.join(comparison_dir, "summary_table.json"), "w") as f:
        json.dump(approaches, f, indent=4)
    
    if approaches:
        with open(os.path.join(comparison_dir, "best_model.json"), "w") as f:
            json.dump(approaches[0], f, indent=4)
    
    # Pareto Plot
    plt.figure(figsize=(6, 5))
    for app in approaches:
        plt.scatter(app["runtime_ms"], app["loss"], label=app["approach"])
        plt.text(app["runtime_ms"], app["loss"], app["approach"], fontsize=8)
    plt.xlabel("Evaluation Time (ms)")
    plt.ylabel("RMSE (log10 mismatch)")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(comparison_dir, "pareto_accuracy_speed.png"))
    plt.savefig(os.path.join(comparison_dir, "pareto_accuracy_speed.pdf"))
    plt.close()
    
    # Loss only comparison
    plt.figure(figsize=(10, 6))
    names = [app["approach"] for app in approaches]
    losses = [app["loss"] for app in approaches]
    plt.bar(names, losses)
    plt.ylabel("RMSE (log10 mismatch)")
    plt.xticks(rotation=45, ha='right')
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "loss_only_comparison.png"))
    plt.savefig(os.path.join(comparison_dir, "loss_only_comparison.pdf"))
    plt.close()

    # Progress plot
    plt.figure(figsize=(8, 5))
    apps_sorted_by_num = sorted(approaches, key=lambda x: x["approach_number"])
    nums = [app["approach_number"] for app in apps_sorted_by_num]
    losses_prog = [app["loss"] for app in apps_sorted_by_num]
    plt.plot(nums, losses_prog, marker='o')
    plt.xlabel("Approach Number")
    plt.ylabel("RMSE (log10 mismatch)")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(os.path.join(comparison_dir, "progress.png"))
    plt.savefig(os.path.join(comparison_dir, "progress.pdf"))
    plt.close()
    
    # Error Histograms
    error_data_path = os.path.join(comparison_dir, "error_data.json")
    if os.path.exists(error_data_path):
        with open(error_data_path, "r") as f:
            error_data = json.load(f)
        
        plt.figure(figsize=(15, 12))
        for i, (name, errors) in enumerate(error_data.items()):
            plt.subplot(5, 5, i+1)
            train_err = np.array(errors["train"])
            val_err = np.array(errors["validation"])
            plt.hist(train_err, bins=20, alpha=0.5, label='Train', density=True)
            plt.hist(val_err, bins=20, alpha=0.5, label='Val', density=True, hatch='//')
            plt.title(name, fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "error_histograms.png"))
        plt.savefig(os.path.join(comparison_dir, "error_histograms.pdf"))
        plt.close()

if __name__ == "__main__":
    generate_plots()
