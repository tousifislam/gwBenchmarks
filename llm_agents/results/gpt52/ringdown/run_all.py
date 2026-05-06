from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import train_and_score  # noqa: E402


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    agent_root = work_dir.parent
    models_dir = work_dir / "models"
    mode = "l2/m+2/n0"

    approaches = [
        # Analytical / classical
        dict(name="Poly-10 (raw)", cat="analytical/classical", par="raw_a", method="poly", hp={"degree": 10}),
        dict(name="Poly-15 (-log(1-a))", cat="analytical/classical", par="log_compact", method="poly", hp={"degree": 15}),
        dict(name="Chebyshev-25 (2a-1)", cat="analytical/classical", par="cheb_mapped", method="chebyshev", hp={"degree": 25}),
        dict(name="Chebyshev-35 (2a-1)", cat="analytical/classical", par="cheb_mapped", method="chebyshev", hp={"degree": 35}),
        dict(name="Rational [5,5] (log)", cat="analytical/classical", par="log_compact", method="rational", hp={"m": 5, "n": 5}),
        dict(name="Rational [7,7] (raw)", cat="analytical/classical", par="raw_a", method="rational", hp={"m": 7, "n": 7}),
        # Interpolation
        dict(name="CubicSpline (raw)", cat="interpolation", par="raw_a", method="spline", hp={}),
        dict(name="CubicSpline (sqrt(1-a^2))", cat="interpolation", par="sqrt_irreducible", method="spline", hp={}),
        dict(name="RBFInterp (TPS, raw)", cat="interpolation", par="raw_a", method="rbf_interp", hp={"kernel": "thin_plate_spline", "neighbors": 250, "smoothing": 0.0}),
        dict(name="RBFInterp (cubic, log)", cat="interpolation", par="log_compact", method="rbf_interp", hp={"kernel": "cubic", "neighbors": 250, "smoothing": 1e-12}),
        # Machine learning / kernels
        dict(name="GPR-RBF (raw)", cat="machine_learning", par="raw_a", method="gpr_rbf", hp={"length_scale": 0.5, "n_restarts": 0}),
        dict(name="KRR-RBF (raw)", cat="machine_learning", par="raw_a", method="krr_rbf", hp={"alpha": 1e-10}),
        dict(name="SVR-RBF (log)", cat="machine_learning", par="log_compact", method="svr_rbf", hp={"C": 300.0, "epsilon": 1e-12}),
        dict(name="MLP (log)", cat="machine_learning", par="log_compact", method="mlp", hp={"hidden_layer_sizes": (64, 64), "max_iter": 4000}),
        dict(name="RandomForest (raw)", cat="machine_learning", par="raw_a", method="rf", hp={"n_estimators": 800}),
        dict(name="HistGB (raw)", cat="machine_learning", par="raw_a", method="hgb", hp={"max_iter": 1200, "learning_rate": 0.03}),
        # Extra variants to reach 18
        dict(name="Poly-12 (compactified)", cat="analytical/classical", par="compactified", method="poly", hp={"degree": 12}),
        dict(name="Rational [9,9] (cheb)", cat="analytical/classical", par="cheb_mapped", method="rational", hp={"m": 9, "n": 9}),
    ]

    for i, cfg in enumerate(approaches, start=1):
        model_dir = models_dir / (
            f"{i:02d}_"
            + cfg["name"]
            .lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(",", "")
            .replace("+", "_")
            .replace("-", "_")
        )
        if (model_dir / "scorecard.json").exists():
            continue
        train_and_score(
            work_dir=work_dir,
            agent_root=agent_root,
            model_dir=model_dir,
            approach_number=i,
            display_name=cfg["name"],
            category=cfg["cat"],
            parameterization=cfg["par"],
            mode=mode,
            method=cfg["method"],
            method_hyperparams=cfg["hp"],
        )


if __name__ == "__main__":
    main()

