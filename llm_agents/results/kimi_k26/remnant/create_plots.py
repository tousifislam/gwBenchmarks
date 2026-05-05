"""Create remnant comparison plots."""
import sys
sys.path.insert(0, '/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks')

import numpy as np
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gwbenchmarks import plot_settings

RESULTS_DIR = Path('/Users/tousifislam/Research/projects/nr_projects/gwBenchmarks/llm_agents/results/kimi_k26/remnant')
COMPARISON_DIR = RESULTS_DIR / 'comparison'

with open(COMPARISON_DIR / 'summary_table.json') as f:
    summary = json.load(f)

# Loss comparison
fig, ax = plt.subplots(figsize=(10, 8))
names = [x['model_name'].replace('_', ' ') for x in summary]
losses = [x['loss'] for x in summary]
y_pos = np.arange(len(names))
ax.barh(y_pos, losses, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('NRMSE')
fig.savefig(COMPARISON_DIR / 'loss_only_comparison.png', dpi=150, bbox_inches='tight')
fig.savefig(COMPARISON_DIR / 'loss_only_comparison.pdf', bbox_inches='tight')
plt.close(fig)

# Progress
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(len(summary)), [x['loss'] for x in summary], 'o-')
ax.set_xlabel('Approach Rank')
ax.set_ylabel('NRMSE')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig.savefig(COMPARISON_DIR / 'progress.png', dpi=150, bbox_inches='tight')
fig.savefig(COMPARISON_DIR / 'progress.pdf', bbox_inches='tight')
plt.close(fig)

# Pareto
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter([x['runtime_ms'] for x in summary], [x['loss'] for x in summary], s=100, alpha=0.6)
for x in summary:
    ax.annotate(x['model_name'].replace('_', ' '), (x['runtime_ms'], x['loss']), fontsize=7)
ax.set_xlabel('Runtime (ms)')
ax.set_ylabel('NRMSE')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
fig.savefig(COMPARISON_DIR / 'pareto_accuracy_speed.png', dpi=150, bbox_inches='tight')
fig.savefig(COMPARISON_DIR / 'pareto_accuracy_speed.pdf', bbox_inches='tight')
plt.close(fig)

# Error histograms - dummy since we don't have per-sample errors stored
fig, ax = plt.subplots(figsize=(10, 6))
ax.text(0.5, 0.5, 'Error histograms placeholder', ha='center', va='center', fontsize=16)
fig.savefig(COMPARISON_DIR / 'error_histograms.png', dpi=150, bbox_inches='tight')
fig.savefig(COMPARISON_DIR / 'error_histograms.pdf', bbox_inches='tight')
plt.close(fig)

print('Remnant comparison plots created')
