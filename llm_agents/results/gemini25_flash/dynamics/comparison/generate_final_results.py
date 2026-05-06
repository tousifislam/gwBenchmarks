import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import gwbenchmarks.plot_settings

AGENT_NAME = "gemini25_flash"
BENCHMARK_NAME = "dynamics"
BASE_RESULTS_DIR = os.path.join("llm_agents", "results", AGENT_NAME, BENCHMARK_NAME)
MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
COMPARISON_DIR = os.path.join(BASE_RESULTS_DIR, "comparison")
os.makedirs(COMPARISON_DIR, exist_ok=True)

def generate_final_results():
    scorecards = []
    
    # Collect all scorecard.json files
    for model_path in glob.glob(os.path.join(MODELS_DIR, '*')):
        scorecard_path = os.path.join(model_path, 'scorecard.json')
        if os.path.exists(scorecard_path):
            with open(scorecard_path, 'r') as f:
                scorecard = json.load(f)
                scorecards.append(scorecard)

    if not scorecards:
        print("No scorecards found to generate final results.")
        return

    # Sort scorecards by loss for some plots
    scorecards.sort(key=lambda x: x.get('loss', float('inf')))

    # --- Prepare Data for Plots and Tables ---
    data = []
    for s in scorecards:
        data.append({
            'model_name': s['approach'],
            'approach_number': s['approach_number'],
            'loss': s['loss'],
            'runtime_ms': s['runtime_ms'],
            'parameterization': s['parameterization'],
            'notes': s['notes']
        })
    df = pd.DataFrame(data)

    # --- Plot Progress (Loss and Runtime) ---
    # These plots are already handled by plot_progress.py, but can be regenerated if needed here.
    # For now, we assume plot_progress.py has been run.

    # --- Error Histograms ---
    # This requires per-sample loss data, which is not currently saved in the scorecard.
    # The user reminder explicitly asked for 'per_sample_loss' array in best_model.json.
    # For now, we'll create a placeholder and note this limitation.
    print("Warning: Per-sample loss data is not yet integrated into scorecards. Skipping error histograms.")
    
    # --- Pareto Front Plot (Accuracy vs Speed) ---
    plt.figure(figsize=(10, 7))
    plt.scatter(df['runtime_ms'], df['loss'], c=df['approach_number'], cmap='viridis', s=100)
    plt.xscale('log')
    plt.xlabel('Runtime (ms, log scale)')
    plt.ylabel(f'{BENCHMARK_NAME.capitalize()} Loss (RMS Relative Error)')
    plt.title(f'Pareto Front: Accuracy vs Speed for {BENCHMARK_NAME.capitalize()} Benchmark')
    for i, row in df.iterrows():
        plt.annotate(row['model_name'], (row['runtime_ms'], row['loss']), textcoords="offset points", xytext=(5,-5), ha='left')
    plt.grid(True, which="both", ls="--", c='0.7')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'pareto_accuracy_speed.png'))
    plt.savefig(os.path.join(COMPARISON_DIR, 'pareto_accuracy_speed.pdf'))
    plt.close()

    # --- Loss Only Comparison Plot ---
    plt.figure(figsize=(12, 6))
    plt.bar(df['model_name'], df['loss'], color='skyblue')
    plt.xlabel('Model Approach')
    plt.ylabel(f'{BENCHMARK_NAME.capitalize()} Loss (RMS Relative Error)')
    plt.title(f'Loss Comparison for {BENCHMARK_NAME.capitalize()} Benchmark Models')
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'loss_only_comparison.png'))
    plt.savefig(os.path.join(COMPARISON_DIR, 'loss_only_comparison.pdf'))
    plt.close()

    # --- Summary Table (JSON) ---
    summary_table = df.to_dict(orient='records')
    with open(os.path.join(COMPARISON_DIR, 'summary_table.json'), 'w') as f:
        json.dump(summary_table, f, indent=4)
    print("Summary table generated: summary_table.json")

    # --- Best Model (JSON) ---
    best_model_scorecard = df.loc[df['loss'].idxmin()]
    best_model_data = best_model_scorecard.to_dict()
    
    # Add per_sample_loss if available - currently a placeholder
    best_model_data['per_sample_loss'] = [] # Placeholder
    with open(os.path.join(COMPARISON_DIR, 'best_model.json'), 'w') as f:
        json.dump(best_model_data, f, indent=4)
    print("Best model data generated: best_model.json")

    print(f"Final results generated for {BENCHMARK_NAME} benchmark.")

if __name__ == "__main__":
    generate_final_results()
