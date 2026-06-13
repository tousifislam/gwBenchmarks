"""Driver for the Ringdown Bench."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import rdapproaches as A
import rdmodels as M
import rdreport as R

HERE = Path(__file__).resolve().parent
CHANGELOG = HERE / "CHANGELOG.md"


def reasoning(sc):
    c = sc["loss_components"]
    harder = "omega_I" if c["rel_error_omega_imag"] > c["rel_error_omega_real"] else "omega_R"
    gap = sc["loss"] / max(sc["train_loss"], 1e-12)
    d = (f"{harder} is harder (rel err wR={c['rel_error_omega_real']:.2e}, "
         f"wI={c['rel_error_omega_imag']:.2e}); train/val gap {gap:.1f}x.")
    return (f"- **Observed**: mean rel err val {sc['loss']:.3e}, train {sc['train_loss']:.3e}. {d}\n"
            f"- **Hypothesis/Change**: {sc['notes']}\n"
            f"- **Result**: {sc['parameterization']} reparam, {sc['runtime_ms']:.4f} ms/spin.\n")


def write_changelog(results):
    L = ["# Ringdown Bench - opus48 CHANGELOG\n",
         "Target: Kerr (l=2,m=2,n=0) QNM (omega_R, omega_I) vs spin a. "
         "Loss = 0.5*(mean|dwR/wR| + mean|dwI/wI|). Smooth 1D functions; the "
         "a->1 (near-extremal) region and omega_I->0 drive relative error. "
         "Log-compactification -log(1-a) resolves the near-extremal regime.\n",
         f"\n## Approaches ({len(results)})\n"]
    for r in sorted(results, key=lambda x: x["scorecard"]["approach_number"]):
        sc = r["scorecard"]
        L.append(f"\n### {sc['approach_number']}. {sc['approach']} [{sc['category']}]\n")
        if sc.get("expression_omega_r"):
            L.append(f"- omega_R ~ `{sc['expression_omega_r'][:90]}`\n")
        L.append(reasoning(sc))
    L.append("\n## Ranking (by mean rel err)\n\n| rank | approach | category | loss | wR | wI | train |\n|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(sorted(results, key=lambda x: x["scorecard"]["loss"])):
        sc = r["scorecard"]; c = sc["loss_components"]
        L.append(f"| {i+1} | {sc['approach']} | {sc['category']} | {sc['loss']:.2e} | "
                 f"{c['rel_error_omega_real']:.2e} | {c['rel_error_omega_imag']:.2e} | {sc['train_loss']:.2e} |\n")
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
          f"Best: {best['approach']} loss={best['loss']:.3e}")


if __name__ == "__main__":
    main()
