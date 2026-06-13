"""Driver for the Analytic Bench."""
import time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import aapproaches as A
import arunner as RUN
import areport as R

HERE = Path(__file__).resolve().parent
CHANGELOG = HERE / "CHANGELOG.md"


def reasoning(sc):
    gap = sc["loss"] - sc["train_loss"]
    if gap > 0.05:
        d = f"val-train gap {gap:.3f}: the eta-polynomial coefficient model extrapolates imperfectly across q."
    else:
        d = "generalises across q; residual set by the closed-form's intrinsic fidelity to NR."
    return (f"- **Observed**: mean FD mismatch val {sc['loss']:.4f} (median {sc['val_median']:.4f}), "
            f"train {sc['train_loss']:.4f}. {d}\n"
            f"- **Hypothesis/Change**: {sc['notes']}\n"
            f"- **Result**: {sc['parameterization']} mass variable, {sc['runtime_ms']:.2f} ms/waveform, "
            f"closed-form (see expression.txt).\n")


def write_changelog(results):
    L = ["# Analytic Bench - opus48 CHANGELOG\n",
         "Goal: closed-form h22(t;q) for non-spinning quasi-circular BBH. "
         "Model: h22 = A(t;q) exp(-i phi(t;q)), all closed-form (no SVD/PCA, no "
         "stored bases, no ODE solves). Phase = exact integral of an integrable "
         "frequency omega = b0 + b1 (tc-t)^(-3/8) + b2 tanh((t-tm)/wr) (PN chirp + "
         "tanh merger transition), giving phi = b0 t - (8/5) b1 (tc-t)^(5/8) "
         "+ b2 wr log cosh((t-tm)/wr) + c. Coefficients are analytic polynomials "
         "in the mass variable, fitted across the 20 training waveforms. PySR/"
         "gplearn discover closed-form log-amplitude. Loss = mean aLIGO FD mismatch.\n",
         "## Key findings\n",
         "- Per-waveform the closed form reaches ~0.05 mismatch; the integrable "
         "frequency makes the phase an exact analytic integral (no numerics at eval).\n",
         "- Phase sign is canonicalised (metric scores only Re(h)).\n",
         f"\n## Approaches ({len(results)})\n"]
    for r in sorted(results, key=lambda x: x["scorecard"]["approach_number"]):
        sc = r["scorecard"]
        L.append(f"\n### {sc['approach_number']}. {sc['approach']} [{sc['category']}]\n")
        L.append(reasoning(sc))
    L.append("\n## Ranking (by FD mismatch)\n\n| rank | approach | category | loss | median | train | ms |\n|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(sorted(results, key=lambda x: x["scorecard"]["loss"])):
        sc = r["scorecard"]
        L.append(f"| {i+1} | {sc['approach']} | {sc['category']} | {sc['loss']:.4f} | "
                 f"{sc['val_median']:.4f} | {sc['train_loss']:.4f} | {sc['runtime_ms']:.2f} |\n")
    CHANGELOG.write_text("".join(L))


def main():
    t0 = time.time(); results = []
    for spec in A.SPECS:
        try:
            r = RUN.run(spec, write=True)
            results.append(r); R.update_progress(results); write_changelog(results)
        except Exception as e:
            import traceback
            print(f"[{spec['number']}] {spec['name']} FAILED: {e}"); traceback.print_exc()
    R.final_plots(results); write_changelog(results)
    best = min(results, key=lambda x: x["scorecard"]["loss"])["scorecard"]
    print(f"\nDONE {len(results)} approaches in {time.time()-t0:.0f}s. "
          f"Best: {best['approach']} mismatch={best['loss']:.4f}")


if __name__ == "__main__":
    main()
