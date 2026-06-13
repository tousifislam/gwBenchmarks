"""Driver for the Dynamics Bench."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import dapproaches as A
import dmodels as M
import dreport as R

HERE = Path(__file__).resolve().parent
CHANGELOG = HERE / "CHANGELOG.md"


def reasoning(sc):
    gap = sc["loss"] / max(sc["train_loss"], 1e-9)
    if gap > 4:
        d = f"train/val gap {gap:.1f}x: coeff regression extrapolates poorly."
    elif sc["train_loss"] > 0.03:
        d = "high train loss: underfits the eccentric oscillation modes."
    else:
        d = "generalises; residual set by unmodelled eccentric oscillation phase."
    return (f"- **Observed**: RMS-rel-err val {sc['loss']:.4f} (median {sc['val_median']:.4f}), "
            f"train {sc['train_loss']:.4f}. {d}\n"
            f"- **Hypothesis/Change**: {sc['notes']}\n"
            f"- **Result**: rank {sc['rank']}, {sc['parameterization']} reparam, "
            f"{sc['runtime_ms']:.2f} ms/evolution.\n")


def write_changelog(results):
    L = ["# Dynamics Bench - opus48 CHANGELOG\n",
         "Target: PN frequency parameter x(t) for eccentric spinning BBH. "
         "Loss = pointwise RMS relative error. Representation: duration-normalised "
         "tau in [0,1], SVD/EIM of log(x)(tau), param->coeff regression; eval grid "
         "given so endpoints need no modelling.\n",
         "## Key findings\n",
         "- x(t) = smooth inspiral growth + eccentric oscillations (~30-50 cycles).\n",
         "- The secular trend dominates the relative error and regresses well "
         "(oracle rank-20 ~0.013); the eccentric oscillation modes have varying "
         "phase in tau and do not regress, so adding high-rank modes plateaus.\n",
         "- cos/sin(zeta0) embedding (trig/full reparam) captures initial anomaly.\n",
         f"\n## Approaches ({len(results)})\n"]
    for r in sorted(results, key=lambda x: x["scorecard"]["approach_number"]):
        sc = r["scorecard"]
        L.append(f"\n### {sc['approach_number']}. {sc['approach']} [{sc['category']}]\n")
        L.append(reasoning(sc))
    L.append("\n## Ranking (by RMS rel err)\n\n| rank | approach | category | loss | median | train | ms |\n|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(sorted(results, key=lambda x: x["scorecard"]["loss"])):
        sc = r["scorecard"]
        L.append(f"| {i+1} | {sc['approach']} | {sc['category']} | {sc['loss']:.4f} | "
                 f"{sc['val_median']:.4f} | {sc['train_loss']:.4f} | {sc['runtime_ms']:.2f} |\n")
    CHANGELOG.write_text("".join(L))


def main():
    t0 = time.time(); results = []
    for spec in A.SPECS:
        try:
            r = M.run_approach(spec, write=True)
            results.append(r); R.update_progress(results); write_changelog(results)
        except Exception as e:
            import traceback
            print(f"[{spec['number']}] {spec['name']} FAILED: {e}"); traceback.print_exc()
    R.final_plots(results); write_changelog(results)
    best = min(results, key=lambda x: x["scorecard"]["loss"])["scorecard"]
    print(f"\nDONE {len(results)} approaches in {time.time()-t0:.0f}s. "
          f"Best: {best['approach']} loss={best['loss']:.4f}")


if __name__ == "__main__":
    main()
