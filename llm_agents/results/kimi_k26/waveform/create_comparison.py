"""Generate all comparison plots and JSON files for waveform benchmark."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load plot settings from gwbenchmarks
from gwbenchmarks import plot_settings

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/waveform')
COMPARISON_DIR = RESULTS_DIR / 'comparison'
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

def load_all_scorecards():
    """Load all scorecards from model directories."""
    models_dir = RESULTS_DIR / 'models'
    scorecards = []
    for model_dir in sorted(models_dir.iterdir()):
        if model_dir.is_dir():
            sc_path = model_dir / 'scorecard.json'
            if sc_path.exists():
                with open(sc_path) as f:
                    sc = json.load(f)
                    sc['model_name'] = model_dir.name
                    scorecards.append(sc)
    return scorecards

def create_summary_table(scorecards):
    """Create summary_table.json with ranked approaches."""
    summary = []
    for sc in scorecards:
        summary.append({
            'model_name': sc['model_name'],
            'approach_number': sc.get('approach_number', 0),
            'loss': sc['loss'],
            'runtime_ms': sc.get('runtime_ms', 0),
            'parameterization': sc.get('parameterization', 'unknown'),
            'approach_category': sc.get('notes', '').split(',')[0] if 'notes' in sc else 'unknown',
        })
    
    summary = sorted(summary, key=lambda x: x['loss'])
    
    with open(COMPARISON_DIR / 'summary_table.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def create_error_data(scorecards):
    """Create error_data.json with per-sample errors."""
    error_data = []
    for sc in scorecards:
        entry = {
            'model_name': sc['model_name'],
            'approach_number': sc.get('approach_number', 0),
            'loss': sc['loss'],
            'val_losses': sc.get('val_losses', []),
        }
        error_data.append(entry)
    
    with open(COMPARISON_DIR / 'error_data.json', 'w') as f:
        json.dump(error_data, f, indent=2)
    
    return error_data

def create_best_model_json(summary):
    """Create best_model.json."""
    if summary:
        best = summary[0]
        best_model = {
            'model_name': best['model_name'],
            'loss': best['loss'],
            'approach_number': best['approach_number'],
        }
        with open(COMPARISON_DIR / 'best_model.json', 'w') as f:
            json.dump(best_model, f, indent=2)

def create_progress_plot(summary):
    """Create progress plot showing loss vs approach number."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by approach number
    sorted_by_num = sorted(summary, key=lambda x: x['approach_number'])
    nums = [x['approach_number'] for x in sorted_by_num]
    losses = [x['loss'] for x in sorted_by_num]
    names = [x['model_name'] for x in sorted_by_num]
    
    ax.plot(nums, losses, 'o-', markersize=4, linewidth=1)
    ax.axhline(y=0.0014, color='r', linestyle='--', label='NR error floor')
    ax.set_xlabel('Approach Number')
    ax.set_ylabel('Mean FD Mismatch')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(COMPARISON_DIR / 'progress.png', dpi=150, bbox_inches='tight')
    fig.savefig(COMPARISON_DIR / 'progress.pdf', bbox_inches='tight')
    plt.close(fig)

def create_loss_comparison(summary):
    """Create bar chart of losses."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by loss
    sorted_by_loss = sorted(summary, key=lambda x: x['loss'])
    names = [x['model_name'].replace('svd_', '').replace('_', ' ') for x in sorted_by_loss]
    losses = [x['loss'] for x in sorted_by_loss]
    
    y_pos = np.arange(len(names))
    ax.barh(y_pos, losses, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean FD Mismatch')
    ax.axvline(x=0.0014, color='r', linestyle='--', label='NR error floor')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')
    
    fig.savefig(COMPARISON_DIR / 'loss_only_comparison.png', dpi=150, bbox_inches='tight')
    fig.savefig(COMPARISON_DIR / 'loss_only_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

def create_error_histograms(error_data):
    """Create histograms of per-sample errors."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histograms for a subset of approaches
    colors = plt.cm.tab10(np.linspace(0, 1, min(10, len(error_data))))
    
    for i, entry in enumerate(error_data[:10]):
        losses = np.array(entry['val_losses'])
        losses = losses[losses < 1.0]  # Filter out NaN/inf
        if len(losses) > 0:
            ax.hist(losses, bins=30, alpha=0.4, label=entry['model_name'], color=colors[i])
    
    ax.axvline(x=0.0014, color='r', linestyle='--', linewidth=2, label='NR error floor')
    ax.set_xlabel('FD Mismatch')
    ax.set_ylabel('Count')
    ax.set_xlim([0, 1.0])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.savefig(COMPARISON_DIR / 'error_histograms.png', dpi=150, bbox_inches='tight')
    fig.savefig(COMPARISON_DIR / 'error_histograms.pdf', bbox_inches='tight')
    plt.close(fig)

def create_pareto_plot(summary):
    """Create Pareto plot (loss vs runtime)."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    losses = [x['loss'] for x in summary]
    runtimes = [x.get('runtime_ms', 0) for x in summary]
    names = [x['model_name'] for x in summary]
    
    scatter = ax.scatter(runtimes, losses, s=100, alpha=0.6)
    
    # Label each point
    for i, name in enumerate(names):
        short_name = name.replace('svd_', '').replace('_', ' ')
        ax.annotate(short_name, (runtimes[i], losses[i]), 
                   fontsize=7, alpha=0.7, xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Runtime per sample (ms)')
    ax.set_ylabel('Mean FD Mismatch')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    fig.savefig(COMPARISON_DIR / 'pareto_accuracy_speed.png', dpi=150, bbox_inches='tight')
    fig.savefig(COMPARISON_DIR / 'pareto_accuracy_speed.pdf', bbox_inches='tight')
    plt.close(fig)

def create_changelog(scorecards):
    """Create CHANGELOG.md."""
    changelog_path = RESULTS_DIR / 'CHANGELOG.md'
    
    with open(changelog_path, 'w') as f:
        f.write('# Waveform Benchmark - CHANGELOG\n\n')
        f.write('## Summary\n\n')
        f.write(f'- Total approaches: {len(scorecards)}\n')
        f.write(f'- Best loss: {min(sc["loss"] for sc in scorecards):.4f}\n')
        f.write(f'- Parameterizations tested: raw, effective, mass_diff, spherical\n')
        f.write(f'- Categories: SVD/decomposition, Symbolic/analytical, Interpolation/kernel, Machine learning\n\n')
        f.write('## Approaches\n\n')
        
        for sc in sorted(scorecards, key=lambda x: x.get('approach_number', 0)):
            f.write(f'### {sc["model_name"]}\n')
            f.write(f'- Loss: {sc["loss"]:.4f}\n')
            f.write(f'- Parameterization: {sc.get("parameterization", "unknown")}\n')
            f.write(f'- Notes: {sc.get("notes", "")}\n\n')

if __name__ == '__main__':
    print('Loading scorecards...')
    scorecards = load_all_scorecards()
    print(f'Found {len(scorecards)} scorecards')
    
    print('Creating summary table...')
    summary = create_summary_table(scorecards)
    
    print('Creating error data...')
    error_data = create_error_data(scorecards)
    
    print('Creating best model json...')
    create_best_model_json(summary)
    
    print('Creating progress plot...')
    create_progress_plot(summary)
    
    print('Creating loss comparison...')
    create_loss_comparison(summary)
    
    print('Creating error histograms...')
    create_error_histograms(error_data)
    
    print('Creating Pareto plot...')
    create_pareto_plot(summary)
    
    print('Creating CHANGELOG...')
    create_changelog(scorecards)
    
    print('All comparison files generated!')
