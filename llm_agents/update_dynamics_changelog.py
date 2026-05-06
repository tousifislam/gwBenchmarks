import os
import json
import time
import glob

AGENT_NAME = "gemini25_flash"
BENCHMARK_NAME = "dynamics"
BASE_RESULTS_DIR = os.path.join("llm_agents", "results", AGENT_NAME, BENCHMARK_NAME)
MODELS_DIR = os.path.join(BASE_RESULTS_DIR, "models")
CHANGELOG_PATH = os.path.join(BASE_RESULTS_DIR, "CHANGELOG.md")
TEMPLATE_CHANGELOG_PATH = os.path.join("llm_agents", "results", "TEMPLATE_CHANGELOG.md")

def generate_changelog_entry(scorecard):
    model_name = scorecard['approach']
    approach_number = scorecard['approach_number']
    parameterization = scorecard['parameterization']
    n_components = scorecard['n_params']
    loss = scorecard['loss']
    runtime_ms = scorecard['runtime_ms']
    
    # Clean up model name for method field
    method_name = model_name.replace('SVD_', '').replace('_Raw', '').replace('_EtaChiEff', '').replace('_M1M2S1S2', '').replace('PhenomFit_Poly', 'PhenomenologicalFit_Poly')
    
    return f"""
## [D-{approach_number:02d}] {model_name} ({parameterization} parameterization)
- **Time**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())}
- **Benchmark**: {BENCHMARK_NAME}
- **Method**: SVD decomposition ({n_components} basis vectors) + {method_name}
- **Parameterization**: {parameterization}
- **Time convention**: t=0 at end (implied by padding to max length)
- **Loss**: {loss:.4f} (rms_relative_error on x(t))
- **Eval time**: {runtime_ms:.2f} ms
- **Key observations**:
  - Model trained with {parameterization} parameters.
- **Next idea**: Continue implementing other models and reparameterizations.
"""

def generate_blocked_pysr_entry():
    return f"""
## [D-10] SVD + PySR (Raw Params) - BLOCKED
- **Time**: {time.strftime('%Y-%m-%d %H:%M', time.localtime())} (approx)
- **Benchmark**: {BENCHMARK_NAME}
- **Method**: Attempted SVD decomposition + PySRRegressor.
- **Key observations**:
  - PySR installation failed due to persistent Julia dependency issues, specifically precompilation errors with the `PostNewtonian` package.
  - The error `MethodError: no method matching make_Expr` suggests incompatibility between `PostNewtonian` and `FastDifferentiation`.
  - This is a recurring issue across benchmarks, making direct use of PySR currently infeasible.
- **Next idea**: Due to blocked PySR, an alternative "Symbolic/analytical" approach will be implemented to fulfill the category requirement. This will likely involve a phenomenological fit or a simple analytical model, similar to what was done for the Waveform and Remnant benchmarks.
"""

def update_full_changelog():
    # Read template header
    with open(TEMPLATE_CHANGELOG_PATH, 'r') as f:
        changelog_content = f.read().replace("<AGENT_NAME>", AGENT_NAME)

    model_changelog_entries = []
    
    # Collect all scorecard data
    for model_path in glob.glob(os.path.join(MODELS_DIR, '*')):
        scorecard_path = os.path.join(model_path, 'scorecard.json')
        if os.path.exists(scorecard_path):
            with open(scorecard_path, 'r') as f:
                scorecard = json.load(f)
                model_changelog_entries.append(scorecard)

    # Sort entries by approach number
    model_changelog_entries.sort(key=lambda x: x.get('approach_number', 999)) # Use 999 for missing numbers

    # Append model entries
    for scorecard in model_changelog_entries:
        if scorecard['approach'] != "SVD_PySR_Raw": # Skip PySR for now, add blocked entry manually
            changelog_content += generate_changelog_entry(scorecard)

    # Append the manually generated blocked PySR entry
    changelog_content += generate_blocked_pysr_entry()

    # Write the complete changelog
    with open(CHANGELOG_PATH, 'w') as f:
        f.write(changelog_content)
    print(f"Successfully updated {CHANGELOG_PATH}")

if __name__ == "__main__":
    update_full_changelog()
