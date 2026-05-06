#!/usr/bin/env python
"""Generic comparison plots for any benchmark."""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
import sys

rcParams['font.size'] = 8
rcParams['axes.linewidth'] = 0.5
rcParams['lines.linewidth'] = 1

def create_plots(benchmark_name):
    """Create all comparison plots for a benchmark."""
    WORK_DIR = Path(f"llm_agents/results/haiku/{benchmark_name}")
    MODEL_DIR = WORK_DIR / "models"
    COMP_DIR = WORK_DIR / "comparison"
    COMP_DIR.mkdir(exist_ok=True)

    # Load models
    models = []
    for model_dir in sorted(MODEL_DIR.glob("NN*")):
        scorecard_path = model_dir / "scorecard.json"
        if scorecard_path.exists():
            with open(scorecard_path) as f:
                scorecard = json.load(f)
                scorecard['model_name'] = model_dir.name
                models.append(scorecard)

    models = sorted(models, key=lambda x: x['approach_number'])
    print(f"Creating plots for {len(models)} models...")

    # Progress plot
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    ax.plot([m['approach_number'] for m in models],
            [m['loss'] for m in models], 'o-', markersize=3, linewidth=0.7, color='#2ca02c')
    ax.set_xlabel('Model Number')
    ax.set_ylabel('Validation Loss')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(COMP_DIR / "progress.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "progress.pdf", bbox_inches='tight')
    plt.close()

    # Error histograms
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    losses_by_cat = {}
    for m in models:
        cat = m['approach'].split('_')[0]
        if cat not in losses_by_cat:
            losses_by_cat[cat] = []
        losses_by_cat[cat].append(m['loss'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (cat, losses) in enumerate(sorted(losses_by_cat.items())):
        ax.hist(losses, bins=5, alpha=0.6, label=cat, color=colors[idx % len(colors)])
    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "error_histograms.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "error_histograms.pdf", bbox_inches='tight')
    plt.close()

    # Pareto plot
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    losses = [m['loss'] for m in models]
    complexities = [m.get('n_params', m['approach_number']) for m in models]
    names = [m['model_name'].replace('NN', '') for m in models]

    ax.scatter(complexities, losses, alpha=0.6, s=30, c=range(len(models)), cmap='viridis')
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

    # Loss comparison
    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    x = np.arange(len(losses))
    ax.bar(x, losses, alpha=0.7, edgecolor='black', linewidth=0.5, color='#1f77b4')
    ax.set_ylabel('Validation Loss')
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, fontsize=6, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(COMP_DIR / "loss_only_comparison.png", dpi=150, bbox_inches='tight')
    plt.savefig(COMP_DIR / "loss_only_comparison.pdf", bbox_inches='tight')
    plt.close()

    # Error data
    error_data = {m['model_name']: {'loss_train': m.get('loss_train', m['loss']),
                                     'loss_val': m['loss'],
                                     'approach': m['approach'],
                                     'parameterization': m['parameterization']}
                  for m in models}
    with open(COMP_DIR / "error_data.json", 'w') as f:
        json.dump(error_data, f, indent=2)

    # Summary table
    summary = []
    for m in sorted(models, key=lambda x: x['loss']):
        summary.append({'rank': len(summary) + 1,
                       'model_name': m['model_name'],
                       'loss': m['loss'],
                       'approach': m['approach'],
                       'parameterization': m['parameterization']})
    with open(COMP_DIR / "summary_table.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Best model
    best = summary[0]
    with open(COMP_DIR / "best_model.json", 'w') as f:
        json.dump({'model_name': best['model_name'],
                  'loss': best['loss'],
                  'approach': best['approach'],
                  'rank': 1,
                  'total_models': len(models)}, f)

    print(f"✓ All plots created")
    print(f"✓ Best model: {best['model_name']} (loss: {best['loss']:.6f})")
    print(f"✓ Total models: {len(models)}")

if __name__ == "__main__":
    benchmark = sys.argv[1] if len(sys.argv) > 1 else "dynamics"
    create_plots(benchmark)
