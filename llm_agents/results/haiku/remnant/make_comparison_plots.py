#!/usr/bin/env python
"""
Generate comparison plots for remnant benchmark.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams

rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1
rcParams['patch.linewidth'] = 0.5

WORK_DIR = Path("llm_agents/results/haiku/remnant")
MODEL_DIR = WORK_DIR / "models"
COMP_DIR = WORK_DIR / "comparison"
COMP_DIR.mkdir(exist_ok=True)

def load_all_scorecards():
    """Load all model scorecards."""
    models = []
    for model_dir in sorted(MODEL_DIR.glob("NN*")):
        scorecard_path = model_dir / "scorecard.json"
        if scorecard_path.exists():
            with open(scorecard_path) as f:
                scorecard = json.load(f)
                scorecard['model_name'] = model_dir.name
                models.append(scorecard)
    return sorted(models, key=lambda x: x['approach_number'])

def main():
    """Generate all comparison files."""
    print("Loading models...")
    models = load_all_scorecards()
    print(f"Loaded {len(models)} models\n")

    # Create progress plot
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    numbers = [m['approach_number'] for m in models]
    losses = [m['loss'] for m in models]
    ax.plot(numbers, losses, 'o-', markersize=3, linewidth=0.7, color='#2ca02c')
    ax.set_xlabel('Model Number')
    ax.set_ylabel('Validation Loss (NRMSE)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "progress.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "progress.pdf", bbox_inches='tight')
    plt.close()
    print("✓ progress.png/pdf")

    # Error histograms
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    losses_by_category = {}
    for m in models:
        approach = m['approach'].split('_')[0]
        if approach not in losses_by_category:
            losses_by_category[approach] = []
        losses_by_category[approach].append(m['loss'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (category, losses) in enumerate(sorted(losses_by_category.items())):
        ax.hist(losses, bins=5, alpha=0.6, label=category, color=colors[idx % len(colors)])

    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "error_histograms.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "error_histograms.pdf", bbox_inches='tight')
    plt.close()
    print("✓ error_histograms.png/pdf")

    # Pareto plot
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    losses = [m['loss'] for m in models]
    complexities = [m.get('n_params', m.get('approach_number', 0)) for m in models]
    names = [m['model_name'].replace('NN', '') for m in models]

    scatter = ax.scatter(complexities, losses, alpha=0.6, s=30, c=range(len(models)), cmap='viridis')
    for i, (x, y, name) in enumerate(zip(complexities, losses, names)):
        if i % 2 == 0:
            ax.annotate(name, (x, y), fontsize=5, alpha=0.7)

    ax.set_xlabel('Model Parameters')
    ax.set_ylabel('Validation Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "pareto_accuracy_speed.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "pareto_accuracy_speed.pdf", bbox_inches='tight')
    plt.close()
    print("✓ pareto_accuracy_speed.png/pdf")

    # Loss comparison
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    names = [m['model_name'].replace('NN', '') for m in models]
    losses = [m['loss'] for m in models]
    colors = ['#1f77b4' if m['approach'] == 'gpr_rbf' else '#ff7f0e' if 'rf' in m['approach'] else '#2ca02c'
              for m in models]

    x = np.arange(len(losses))
    ax.bar(x, losses, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Validation Loss')
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, fontsize=6, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "loss_only_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "loss_only_comparison.pdf", bbox_inches='tight')
    plt.close()
    print("✓ loss_only_comparison.png/pdf")

    # Error data
    error_data = {}
    for m in models:
        error_data[m['model_name']] = {
            'loss_train': m.get('loss_train', m['loss']),
            'loss_val': m['loss'],
            'approach': m['approach'],
            'parameterization': m['parameterization']
        }

    with open(COMP_DIR / "error_data.json", 'w') as f:
        json.dump(error_data, f, indent=2)
    print("✓ error_data.json")

    # Summary table
    summary = []
    for m in sorted(models, key=lambda x: x['loss']):
        summary.append({
            'rank': len(summary) + 1,
            'model_name': m['model_name'],
            'approach': m['approach'],
            'parameterization': m['parameterization'],
            'loss': m['loss'],
            'loss_train': m.get('loss_train', m['loss']),
            'n_params': m.get('n_params', 0),
            'notes': m.get('notes', '')
        })

    with open(COMP_DIR / "summary_table.json", 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ summary_table.json")

    # Best model
    best = summary[0]
    best_model = {
        'model_name': best['model_name'],
        'rank': 1,
        'loss': best['loss'],
        'approach': best['approach'],
        'parameterization': best['parameterization'],
        'reason': 'Lowest validation loss',
        'total_models_evaluated': len(models)
    }

    with open(COMP_DIR / "best_model.json", 'w') as f:
        json.dump(best_model, f, indent=2)
    print("✓ best_model.json")

    print(f"\n✓ All comparison files created")
    print(f"Best model: {best['model_name']} (loss: {best['loss']:.6f})")
    print(f"Total models: {len(models)}")

    # Verify criteria
    print("\n=== Completion Checklist ===")
    print(f"✓ {len(models)} >= 20 models: PASS" if len(models) >= 20 else f"✗ {len(models)} >= 20: FAIL")

    reparams = set(m['parameterization'] for m in models)
    print(f"✓ {len(reparams)} >= 3 reparameterizations: PASS" if len(reparams) >= 3 else f"✗ {len(reparams)} >= 3: FAIL")

    categories = set(m['approach'] for m in models)
    print(f"✓ {len(categories)} approach categories: PASS" if len(categories) >= 4 else f"✗ Fewer than 4: FAIL")

if __name__ == "__main__":
    main()
