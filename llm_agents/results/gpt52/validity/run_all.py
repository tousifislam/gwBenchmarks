from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import train_and_score  # noqa: E402


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    models_dir = work_dir / "models"

    approaches = [
        # Kernel / GP
        dict(name="GPR-RBF (raw)", cat="kernel/gp", par="raw_4d", kind="gpr_rbf", hp={"length_scale": 1.0, "n_restarts": 0}),
        dict(name="GPR-Matern-5/2 (eff)", cat="kernel/gp", par="effective_4d", kind="gpr_matern", hp={"length_scale": 1.0, "nu": 2.5, "n_restarts": 0}),
        dict(name="KRR-RBF (log)", cat="kernel/gp", par="log_4d", kind="krr_rbf", hp={"alpha": 1e-6}),
        dict(name="SVR-RBF (raw)", cat="kernel/gp", par="raw_4d", kind="svr_rbf", hp={"C": 30.0, "epsilon": 5e-3}),
        # Interpolation
        dict(name="RBFInterp (TPS, raw)", cat="interpolation", par="raw_4d", kind="rbf_interp", hp={"kernel": "thin_plate_spline", "neighbors": 120, "smoothing": 0.0}),
        dict(name="kNN (eff)", cat="interpolation", par="effective_4d", kind="knn", hp={"n_neighbors": 9}),
        # Symbolic / analytical (non-symbolic baselines)
        dict(name="Ridge (eff)", cat="symbolic/analytical", par="effective_4d", kind="ridge", hp={"alpha": 1e-2}),
        dict(name="Poly-3 Ridge (raw)", cat="symbolic/analytical", par="raw_4d", kind="poly_ridge", hp={"degree": 3, "alpha": 1e-3}),
        dict(name="Lasso (raw)", cat="symbolic/analytical", par="raw_4d", kind="lasso", hp={"alpha": 1e-3}),
        dict(name="ElasticNet (raw)", cat="symbolic/analytical", par="raw_4d", kind="elasticnet", hp={"alpha": 1e-3, "l1_ratio": 0.5}),
        # Machine learning
        dict(name="RandomForest (raw)", cat="machine_learning", par="raw_4d", kind="rf", hp={"n_estimators": 1200}),
        dict(name="ExtraTrees (raw)", cat="machine_learning", par="raw_4d", kind="extratrees", hp={"n_estimators": 1600}),
        dict(name="HistGB (raw)", cat="machine_learning", par="raw_4d", kind="hgb", hp={"max_iter": 1200, "learning_rate": 0.03}),
        dict(name="MLP (log)", cat="machine_learning", par="log_4d", kind="mlp", hp={"hidden_layer_sizes": (64, 64), "max_iter": 5000}),
        # Extra variants / feature engineering
        dict(name="KRR-RBF (interaction)", cat="kernel/gp", par="interaction_6d", kind="krr_rbf", hp={"alpha": 1e-6}),
        dict(name="SVR-RBF (log)", cat="kernel/gp", par="log_4d", kind="svr_rbf", hp={"C": 50.0, "epsilon": 3e-3}),
        dict(name="Ridge (boundary-distance)", cat="symbolic/analytical", par="boundary_distance_6d", kind="ridge", hp={"alpha": 1e-2}),
        dict(name="HistGB (boundary-distance)", cat="machine_learning", par="boundary_distance_6d", kind="hgb", hp={"max_iter": 1200, "learning_rate": 0.03}),
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
            model_dir=model_dir,
            approach_number=i,
            display_name=cfg["name"],
            category=cfg["cat"],
            regressor_kind=cfg["kind"],
            regressor_hyperparams=cfg["hp"],
            parameterization=cfg["par"],
        )


if __name__ == "__main__":
    main()

