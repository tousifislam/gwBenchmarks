"""Driver for the Validity Bench."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import vapproaches as A
import vmodels as M
import vreport as R

HERE = Path(__file__).resolve().parent
CHANGELOG = HERE / "CHANGELOG.md"


def reasoning(sc):
    gap = sc["loss"] - sc["train_loss"]
    if gap > 0.5:
        d = f"val-train log-RMSE gap {gap:.2f}: overfits; the extrapolation region (q>8, |chi|>0.8) is undersampled."
    elif sc["train_loss"] > 1.5:
        d = "high train loss: mismatch spans orders of magnitude and the model underfits the saturated (mm~1) region."
    else:
        d = "generalises; residual set by intrinsic scatter of NR-vs-surrogate mismatch."
    return (f"- **Observed**: log RMSE val {sc['loss']:.4f}, train {sc['train_loss']:.4f}. {d}\n"
            f"- **Hypothesis/Change**: {sc['notes']}\n"
            f"- **Result**: {sc['parameterization']} reparam, {sc['runtime_ms']:.3f} ms/sample.\n")


def write_changelog(results):
    L = ["# Validity Bench - opus48 CHANGELOG\n",
         "Target: log(mismatch) between SXS NR and NRHybSur3dq8 (aligned spin). "
         "Loss = natural-log RMSE. NRHybSur3dq8 is valid for q<=8, |chi|<=0.8; "
         "beyond that the mismatch saturates toward 1. Boundary-distance features "
         "(max(0,q-8), max(0,|chi|-0.8)) make the validity edge explicit.\n",
         f"\n## Approaches ({len(results)})\n"]
    for r in sorted(results, key=lambda x: x["scorecard"]["approach_number"]):
        sc = r["scorecard"]
        L.append(f"\n### {sc['approach_number']}. {sc['approach']} [{sc['category']}]\n")
        L.append(reasoning(sc))
    L.append("\n## Ranking (by log RMSE)\n\n| rank | approach | category | log RMSE | train | ms |\n|---|---|---|---|---|---|\n")
    for i, r in enumerate(sorted(results, key=lambda x: x["scorecard"]["loss"])):
        sc = r["scorecard"]
        L.append(f"| {i+1} | {sc['approach']} | {sc['category']} | {sc['loss']:.4f} | "
                 f"{sc['train_loss']:.4f} | {sc['runtime_ms']:.3f} |\n")
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
          f"Best: {best['approach']} log_rmse={best['loss']:.4f}")


if __name__ == "__main__":
    main()
