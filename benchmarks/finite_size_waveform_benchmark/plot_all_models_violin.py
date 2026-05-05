from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "chatgpt_52_xhigh": "GPT-5.2\nxhigh",
    "chatgpt_53_codex_xhigh": "GPT-5.3 Codex\nxhigh",
    "chatgpt_54_mini_xhigh": "GPT-5.4 Mini\nxhigh",
    "chatgpt_55_xhigh": "GPT-5.5\nxhigh",
    "claude_sonnet46_xhigh": "Claude Sonnet 4.6\nxhigh",
    "claude_opus47_xhigh": "Claude Opus 4.7\nxhigh",
    "gemini_31_flash": "Gemini 3.1 Flash",
    "gemini_31_pro": "Gemini 3.1 Pro",
}

MODEL_COLORS = {
    "chatgpt_52_xhigh": "#2F6FDB",
    "chatgpt_53_codex_xhigh": "#2F6FDB",
    "chatgpt_54_mini_xhigh": "#2F6FDB",
    "chatgpt_55_xhigh": "#2F6FDB",
    "claude_sonnet46_xhigh": "#B55239",
    "claude_opus47_xhigh": "#B55239",
    "gemini_31_flash": "#2A9D78",
    "gemini_31_pro": "#2A9D78",
}


def load_scores(benchmarks_dir: Path) -> list[dict]:
    rows = []
    for score_path in sorted(benchmarks_dir.glob("*/no_skills/finite_size/score_finite_size.json")):
        model = score_path.parents[2].name
        with score_path.open() as f:
            score = json.load(f)
        mismatches = np.array(
            [
                row["mismatch_opt"]
                for row in score.get("rows", [])
                if row.get("status") == "ok" and np.isfinite(row.get("mismatch_opt", np.nan))
            ],
            dtype=float,
        )
        if mismatches.size == 0:
            continue
        rows.append(
            {
                "model": model,
                "label": MODEL_LABELS.get(model, model.replace("_", "\n")),
                "color": MODEL_COLORS.get(model, "#5B677A"),
                "mismatches": np.clip(mismatches, 1e-12, 1.0),
                "n_failed": int(score.get("n_failed_evaluations", 0)),
                "mean": float(score.get("mean_mismatch_opt", np.mean(mismatches))),
                "median": float(score.get("median_mismatch_opt", np.median(mismatches))),
                "p90": float(score.get("p90_mismatch_opt", np.quantile(mismatches, 0.9))),
                "n": int(score.get("n_evaluations", mismatches.size)),
            }
        )
    rows.sort(key=lambda item: item["median"])
    return rows


def plot(rows: list[dict], output_prefix: Path) -> None:
    if not rows:
        raise RuntimeError("No finite-size score files found.")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    data = [np.log10(item["mismatches"]) for item in rows]
    labels = [item["label"] for item in rows]
    positions = np.arange(1, len(rows) + 1)
    all_log_values = np.concatenate(data)
    y_min = min(-8.4, float(np.floor(np.min(all_log_values) - 0.25)))
    y_max = max(-0.35, float(np.ceil(np.max(all_log_values) + 0.15)))

    fig, ax = plt.subplots(figsize=(13.2, 4.1))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body, item in zip(parts["bodies"], rows):
        body.set_facecolor(item["color"])
        body.set_edgecolor("#20242A")
        body.set_alpha(0.72)
        body.set_linewidth(0.8)

    medians = np.array([np.log10(item["median"]) for item in rows])
    means = np.array([np.log10(item["mean"]) for item in rows])
    ax.scatter(
        positions,
        medians,
        marker="o",
        s=42,
        color="#111111",
        zorder=4,
        label="median",
    )
    ax.scatter(
        positions,
        means,
        marker="D",
        s=32,
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=4,
        label="mean",
    )

    for x, item in zip(positions, rows):
        if item["n_failed"]:
            ax.text(
                x,
                -0.08,
                f"{item['n_failed']} failed",
                ha="center",
                va="top",
                fontsize=8,
                color="#8A1F17",
                transform=ax.get_xaxis_transform(),
            )

    for value, label in [(-2.0, r"$10^{-2}$"), (-4.0, r"$10^{-4}$")]:
        ax.axhline(value, color="#808892", linestyle="--", linewidth=0.9, alpha=0.75)
        ax.text(
            0.995,
            value,
            label,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color="#5B677A",
            fontsize=10,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel(r"$\log_{10}\,\mathcal{M}_{\mathrm{opt}}$")
    ax.set_title("Finite-size waveform benchmark across agent models")
    ax.set_xlim(0.35, len(rows) + 0.65)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", which="major", color="#D7DCE2", linewidth=0.8, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=2)

    fig.tight_layout()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory containing model/no_skills/finite_size/score_finite_size.json files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("benchmarks/finite_size_waveform_benchmark/finite_size_all_models_violin"),
        help="Output path without extension.",
    )
    args = parser.parse_args()

    rows = load_scores(args.benchmarks_dir)
    plot(rows, args.output_prefix)
    print(args.output_prefix.with_suffix(".pdf"))
    print(args.output_prefix.with_suffix(".png"))
    for item in rows:
        print(
            f"{item['model']}: median={item['median']:.6g}, "
            f"mean={item['mean']:.6g}, p90={item['p90']:.6g}, "
            f"n={item['n']}, failed={item['n_failed']}"
        )


if __name__ == "__main__":
    main()
