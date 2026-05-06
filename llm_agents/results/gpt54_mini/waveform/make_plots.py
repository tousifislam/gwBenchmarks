"""Regenerate waveform benchmark comparison plots from saved JSON artifacts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
WORK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORK_DIR))

from build_models import plot_error_histograms, plot_loss_only, plot_pareto, plot_progress  # noqa: E402

COMPARISON_DIR = WORK_DIR / "comparison"


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    records = load_json(COMPARISON_DIR / "summary_table.json")
    error_data = load_json(COMPARISON_DIR / "error_data.json")
    plot_progress(records)
    plot_loss_only(records)
    plot_pareto(records)
    plot_error_histograms(error_data)


if __name__ == "__main__":
    main()
