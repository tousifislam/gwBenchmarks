import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
import pickle # Added this line

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import gwbenchmarks.plot_settings
from llm_agents.run_validity_model import load_validity_data, reparameterize_params, calculate_validity_loss # Import necessary functions

AGENT_NAME = "gemini25_flash"
BENCHMARK_NAME = "validity"
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
    # These plots are already handled by plot_progress function in run_validity_model,
    # but we can regenerate them explicitly here or ensure that plot_progress was called.
    # For simplicity, we assume plot_progress was called during individual model runs.

    # --- Error Histograms ---
    # To generate error histograms, we need to re-run the prediction for the best model
    # and calculate per-sample losses.
    # For simplicity, we will calculate and store per-sample errors for all models for now.

    error_data = {}
    
    val_data_path = os.path.join("datasets", BENCHMARK_NAME, f"{BENCHMARK_NAME}_validation.h5")
    val_params_raw, val_log_mm_true = load_validity_data(val_data_path)

    for s in scorecards:
        model_name = s['approach']
        parameterization_type = s['parameterization']
        model_dir = os.path.join(MODELS_DIR, model_name)
        saved_model_dir = os.path.join(model_dir, "saved_model")

        # Load scaler
        param_scaler = pickle.load(open(os.path.join(saved_model_dir, "param_scaler.pkl"), "rb"))

        # Reparameterize and scale validation parameters
        val_params_reparam = reparameterize_params(val_params_raw, parameterization_type)
        scaled_val_params = param_scaler.transform(val_params_reparam)

        # Load model and predict
        pred_log_mm = None
        if model_name.startswith("Polynomial_Fit") or model_name.startswith("PhenomFit_Poly"):
            poly_features = pickle.load(open(os.path.join(saved_model_dir, "poly_features.pkl"), "rb"))
            val_poly_params = poly_features.transform(scaled_val_params)
            model = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
            pred_log_mm = model.predict(val_poly_params)
        elif model_name.startswith("RBFInterp"):
            model = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
            pred_log_mm = model(scaled_val_params)
        else: # GPR, KRR, SVR, MLP, RandomForest, XGBoost, LightGBM, NNInterp
            model = pickle.load(open(os.path.join(saved_model_dir, "model.pkl"), "rb"))
            pred_log_mm = model.predict(scaled_val_params)
        
        # Calculate per-sample errors
        per_sample_errors = np.abs(pred_log_mm - val_log_mm_true)
        error_data[model_name] = per_sample_errors.tolist()
    
    # Save error data
    with open(os.path.join(COMPARISON_DIR, 'error_data.json'), 'w') as f:
        json.dump(error_data, f, indent=4)
    print("Error data generated: error_data.json")

    # Plot error histograms
    plt.figure(figsize=(12, 8))
    for model_name, errors in error_data.items():
        plt.hist(errors, bins=50, alpha=0.5, label=model_name, density=True)
    plt.xlabel('Absolute Error (log10(mismatch))')
    plt.ylabel('Density')
    plt.title(f'Error Histograms for {BENCHMARK_NAME.capitalize()} Benchmark Models')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(COMPARISON_DIR, 'error_histograms.png'))
    plt.savefig(os.path.join(COMPARISON_DIR, 'error_histograms.pdf'))
    plt.close()

    # --- Pareto Front Plot (Accuracy vs Speed) ---
    plt.figure(figsize=(10, 7))
    plt.scatter(df['runtime_ms'], df['loss'], c=df['approach_number'], cmap='viridis', s=100)
    plt.xscale('log')
    plt.xlabel('Runtime (ms, log scale)')
    plt.ylabel(f'{BENCHMARK_NAME.capitalize()} Loss (RMSE on log10(mismatch))')
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
    plt.ylabel(f'{BENCHMARK_NAME.capitalize()} Loss (RMSE on log10(mismatch))')
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
    
    # Add per_sample_loss to best_model.json
    best_model_data['per_sample_loss'] = error_data[best_model_scorecard['model_name']]
    with open(os.path.join(COMPARISON_DIR, 'best_model.json'), 'w') as f:
        json.dump(best_model_data, f, indent=4)
    print("Best model data generated: best_model.json")

    print(f"Final results generated for {BENCHMARK_NAME} benchmark.")

if __name__ == "__main__":
    generate_final_results()
