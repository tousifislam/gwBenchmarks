import json
import os
import numpy as np
import matplotlib.pyplot as plt
from gwbenchmarks.plot_settings import apply

apply()

def generate_plots():
    results_dir = "llm_agents/results/gemini3_flash_preview/analytic/"
    models_dir = os.path.join(results_dir, "models")
    comparison_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    approaches = []
    expressions = []
    for d in sorted(os.listdir(models_dir)):
        scorecard_path = os.path.join(models_dir, d, "scorecard.json")
        if os.path.exists(scorecard_path):
            with open(scorecard_path, "r") as f:
                data = json.load(f)
                approaches.append(data)
            
            expr_path = os.path.join(models_dir, d, "expression.txt")
            if os.path.exists(expr_path):
                with open(expr_path, "r") as f:
                    expressions.append({
                        "approach": d,
                        "expression": f.read(),
                        "loss": data["loss"]
                    })
    
    with open(os.path.join(comparison_dir, "all_expressions.json"), "w") as f:
        json.dump(expressions, f, indent=4)
        
    # Summary table
    approaches.sort(key=lambda x: x["loss"] if np.isfinite(x["loss"]) else 1e10)
    with open(os.path.join(comparison_dir, "summary_table.json"), "w") as f:
        json.dump(approaches, f, indent=4)
    
    if approaches:
        with open(os.path.join(comparison_dir, "best_model.json"), "w") as f:
            json.dump(approaches[0], f, indent=4)
    
    # Pareto Plot
    plt.figure(figsize=(6, 5))
    for app in approaches:
        if np.isfinite(app["loss"]):
            plt.scatter(app["runtime_ms"] + 1e-3, app["loss"], label=app["approach"])
            plt.text(app["runtime_ms"] + 1e-3, app["loss"], app["approach"], fontsize=8)
    plt.xlabel("Evaluation Time (ms)")
    plt.ylabel("Mean FD Mismatch")
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(comparison_dir, "pareto_accuracy_speed.png"))
    plt.savefig(os.path.join(comparison_dir, "pareto_accuracy_speed.pdf"))
    plt.close()
    
    # Loss only comparison
    plt.figure(figsize=(10, 6))
    valid_apps = [app for app in approaches if np.isfinite(app["loss"])]
    names = [app["approach"] for app in valid_apps]
    losses = [app["loss"] for app in valid_apps]
    plt.bar(names, losses)
    plt.ylabel("Mean FD Mismatch")
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
    valid_mask = np.isfinite(losses_prog)
    plt.plot(np.array(nums)[valid_mask], np.array(losses_prog)[valid_mask], marker='o')
    plt.xlabel("Approach Number")
    plt.ylabel("Mean FD Mismatch")
    plt.yscale("log")
    plt.grid(True)
    plt.savefig(os.path.join(comparison_dir, "progress.png"))
    plt.savefig(os.path.join(comparison_dir, "progress.pdf"))
    plt.close()

if __name__ == "__main__":
    generate_plots()
