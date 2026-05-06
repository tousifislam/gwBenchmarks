from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import train_and_score  # noqa: E402


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    agent_root = work_dir.parent
    models_dir = work_dir / "models"

    approaches = [
        # Kernel/GP
        dict(display_name="GPR (RBF, raw)", category="kernel/gp", kind="gpr_rbf", hp={"length_scale": 1.0}, par="raw_7d"),
        dict(display_name="GPR (Matern-5/2, eff)", category="kernel/gp", kind="gpr_matern", hp={"length_scale": 1.0, "nu": 2.5}, par="effective_spins_7d"),
        dict(display_name="KRR (RBF, pn)", category="kernel/gp", kind="krr_rbf", hp={"alpha": 1e-3}, par="pn_products_5d"),
        dict(display_name="SVR (RBF, raw)", category="kernel/gp", kind="svr_rbf", hp={"C": 20.0, "epsilon": 1e-4}, par="raw_7d"),
        # Interpolation
        dict(display_name="RBFInterpolator (eff)", category="interpolation", kind="rbf_interp", hp={"neighbors": 80, "smoothing": 1e-8}, par="effective_spins_7d"),
        dict(display_name="kNN (eff)", category="interpolation", kind="knn", hp={"n_neighbors": 9, "weights": "distance"}, par="effective_spins_7d"),
        # ML
        dict(display_name="MLP (eff)", category="machine_learning", kind="mlp", hp={"hidden_layer_sizes": (128, 128)}, par="effective_spins_7d"),
        dict(display_name="RandomForest (raw)", category="machine_learning", kind="rf", hp={"n_estimators": 700}, par="raw_7d"),
        dict(display_name="ExtraTrees (raw)", category="machine_learning", kind="extratrees", hp={"n_estimators": 900}, par="raw_7d"),
        dict(display_name="HistGB (pn)", category="machine_learning", kind="hgb", hp={"max_iter": 600}, par="pn_products_5d"),
        # Analytical-ish
        dict(display_name="Ridge (massdiff+chia)", category="symbolic/analytical", kind="ridge", hp={"alpha": 1e-2}, par="massdiff_chia_5d"),
        dict(display_name="Poly-3 Ridge (raw)", category="symbolic/analytical", kind="poly_ridge", hp={"degree": 3, "alpha": 1e-3}, par="raw_7d"),
        dict(display_name="Poly-5 Ridge (pn)", category="symbolic/analytical", kind="poly_ridge", hp={"degree": 5, "alpha": 3e-3}, par="pn_products_5d"),
        dict(display_name="Ridge (spherical)", category="symbolic/analytical", kind="ridge", hp={"alpha": 1e-2}, par="spherical_spins_7d"),
        # More variants to reach 18 (before symbolic)
        dict(display_name="KRR (RBF, raw, alpha=1e-2)", category="kernel/gp", kind="krr_rbf", hp={"alpha": 1e-2}, par="raw_7d"),
        dict(display_name="SVR (RBF, pn)", category="kernel/gp", kind="svr_rbf", hp={"C": 50.0, "epsilon": 2e-4}, par="pn_products_5d"),
        dict(display_name="MLP (raw)", category="machine_learning", kind="mlp", hp={"hidden_layer_sizes": (256, 128)}, par="raw_7d"),
        dict(display_name="HistGB (eff)", category="machine_learning", kind="hgb", hp={"max_iter": 800, "learning_rate": 0.03}, par="effective_spins_7d"),
    ]

    for i, cfg in enumerate(approaches, start=1):
        model_dir = models_dir / (
            f"{i:02d}_"
            + cfg["display_name"]
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
            display_name=cfg["display_name"],
            category=cfg["category"],
            regressor_kind=cfg["kind"],
            regressor_hyperparams=cfg["hp"],
            parameterization=cfg["par"],
        )


if __name__ == "__main__":
    main()
