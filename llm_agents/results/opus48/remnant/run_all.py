"""Driver for the Remnant Bench: run all approaches, score, persist, report."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import approaches_r as A
import rmodels as M
import rreport as R

HERE = Path(__file__).resolve().parent
CHANGELOG = HERE / "CHANGELOG.md"


def reasoning(sc):
    gap = sc["loss"] / max(sc["train_loss"], 1e-9)
    if gap > 5:
        d = f"train/val gap {gap:.1f}x -> overfitting in the sparse kick landscape."
    elif sc["train_loss"] > 0.2:
        d = "high train loss -> underfits the sharp superkick features."
    else:
        d = "generalises well; error set by intrinsic kick scatter."
    return (f"- **Observed**: NRMSE(v_k) val {sc['loss']:.4f}, train {sc['train_loss']:.4f}. {d}\n"
            f"- **Hypothesis/Change**: {sc['notes']}\n"
            f"- **Result**: {sc['parameterization']} reparam, target {sc['target']}, "
            f"{sc['runtime_ms']:.3f} ms/sample.\n")


def write_changelog(results):
    L = ["# Remnant Bench - opus48 CHANGELOG\n",
         "Target: remnant kick magnitude v_k. Loss = NRMSE(v_k). Kicks are "
         "notoriously hard: superkick configurations (q~1, anti-aligned in-plane "
         "spins) produce sharp peaks. The in-plane spin difference and PN product "
         "features (eta, chi_eff, delta*chi_a) are physically motivated.\n",
         f"\n## Approaches\n"]
    for r in sorted(results, key=lambda x: x["scorecard"]["approach_number"]):
        sc = r["scorecard"]
        L.append(f"\n### {sc['approach_number']}. {sc['approach']} [{sc['category']}] "
                 f"(target {sc['target']})\n")
        L.append(reasoning(sc))
    L.append("\n## Ranking (v_k approaches, by NRMSE)\n\n| rank | approach | category | NRMSE | train | ms |\n|---|---|---|---|---|---|\n")
    vf = [r for r in results if r["scorecard"]["target"] == "vf_mag"]
    for i, r in enumerate(sorted(vf, key=lambda x: x["scorecard"]["loss"])):
        sc = r["scorecard"]
        L.append(f"| {i+1} | {sc['approach']} | {sc['category']} | {sc['loss']:.4f} | "
                 f"{sc['train_loss']:.4f} | {sc['runtime_ms']:.3f} |\n")
    CHANGELOG.write_text("".join(L))


def main():
    t0 = time.time(); results = []
    for spec in A.SPECS:
        try:
            r = M.run_approach(spec, write=True)
            results.append(r)
            R.update_progress(results); write_changelog(results)
        except Exception as e:
            import traceback
            print(f"[{spec['number']}] {spec['name']} FAILED: {e}")
            traceback.print_exc()
    R.final_plots(results); write_changelog(results)
    vf = [r for r in results if r["scorecard"]["target"] == "vf_mag"]
    best = min(vf, key=lambda x: x["scorecard"]["loss"])["scorecard"]
    print(f"\nDONE {len(results)} approaches ({len(vf)} v_k) in {time.time()-t0:.0f}s. "
          f"Best: {best['approach']} NRMSE={best['loss']:.4f}")


if __name__ == "__main__":
    main()
