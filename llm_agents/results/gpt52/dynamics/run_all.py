from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import train_and_score_svd_regressor  # noqa: E402


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    agent_root = work_dir.parent
    models_dir = work_dir / "models"

    approaches = [
        dict(display_name="SVD+Ridge (raw, tau)", category="svd/decomposition", kind="ridge", hp={"alpha": 1e-2}, par="raw_6d", tconv="normalized_time", n_svd=30, eim=False),
        dict(display_name="SVD+GPR (RBF, raw, tau)", category="svd/decomposition", kind="gpr_rbf", hp={"length_scale": 1.0}, par="raw_6d", tconv="normalized_time", n_svd=10, eim=False),
        dict(display_name="SVD+GPR (Matern, eff, tau)", category="svd/decomposition", kind="gpr_matern", hp={"length_scale": 1.0, "nu": 2.5}, par="eff_loge0_6d", tconv="normalized_time", n_svd=10, eim=False),
        dict(display_name="SVD+Poly-3 Ridge (raw, tau)", category="svd/decomposition", kind="poly_ridge", hp={"degree": 3, "alpha": 1e-3}, par="raw_6d", tconv="normalized_time", n_svd=30, eim=False),
        dict(display_name="SVD+KRR (trig, tau)", category="interpolation/kernel", kind="krr_rbf", hp={"alpha": 1e-3}, par="trig_anomaly_7d", tconv="normalized_time", n_svd=20, eim=False),
        dict(display_name="SVD+RBFInterpolator (eff, tau)", category="interpolation/kernel", kind="rbf_interp", hp={"neighbors": 80, "smoothing": 1e-8}, par="eff_loge0_6d", tconv="normalized_time", n_svd=14, eim=False),
        dict(display_name="SVD+kNN (eff, tau)", category="interpolation/kernel", kind="knn", hp={"n_neighbors": 9, "weights": "distance"}, par="eff_loge0_6d", tconv="normalized_time", n_svd=20, eim=False),
        dict(display_name="EIM+KRR (raw, tau)", category="svd/decomposition", kind="krr_rbf", hp={"alpha": 1e-3}, par="raw_6d", tconv="normalized_time", n_svd=18, eim=True),
        dict(display_name="SVD+MLP (eff, tau)", category="machine_learning", kind="mlp", hp={"hidden_layer_sizes": (256, 256)}, par="eff_loge0_6d", tconv="normalized_time", n_svd=30, eim=False),
        dict(display_name="SVD+RandomForest (raw, tau)", category="machine_learning", kind="rf", hp={"n_estimators": 500}, par="raw_6d", tconv="normalized_time", n_svd=20, eim=False),
        dict(display_name="SVD+ExtraTrees (raw, tau)", category="machine_learning", kind="extratrees", hp={"n_estimators": 800}, par="raw_6d", tconv="normalized_time", n_svd=20, eim=False),
        dict(display_name="SVD+HistGB (eff, tau)", category="machine_learning", kind="hgb", hp={"max_iter": 700, "learning_rate": 0.04}, par="eff_loge0_6d", tconv="normalized_time", n_svd=20, eim=False),
        dict(display_name="SVD+SVR (raw, tau)", category="interpolation/kernel", kind="svr_rbf", hp={"C": 20.0, "epsilon": 1e-3}, par="raw_6d", tconv="normalized_time", n_svd=10, eim=False),
        dict(display_name="SVD+Lasso (raw, tau)", category="svd/decomposition", kind="lasso", hp={"alpha": 1e-4}, par="raw_6d", tconv="normalized_time", n_svd=30, eim=False),
        dict(display_name="SVD+ElasticNet (raw, tau)", category="svd/decomposition", kind="elasticnet", hp={"alpha": 2e-4, "l1_ratio": 0.3}, par="raw_6d", tconv="normalized_time", n_svd=30, eim=False),
        dict(display_name="SVD+Ridge (fully, tau)", category="svd/decomposition", kind="ridge", hp={"alpha": 2e-2}, par="fully_transformed_7d", tconv="normalized_time", n_svd=30, eim=False),
        dict(display_name="SVD+Ridge (raw, t_end)", category="svd/decomposition", kind="ridge", hp={"alpha": 1e-2}, par="raw_6d", tconv="t0_at_end", n_svd=30, eim=False),
        dict(display_name="SVD+GPR (RBF, raw, t_end)", category="svd/decomposition", kind="gpr_rbf", hp={"length_scale": 1.0}, par="raw_6d", tconv="t0_at_end", n_svd=10, eim=False),
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
        train_and_score_svd_regressor(
            work_dir=work_dir,
            agent_root=agent_root,
            model_dir=model_dir,
            approach_number=i,
            display_name=cfg["display_name"],
            category=cfg["category"],
            regressor_kind=cfg["kind"],
            regressor_hyperparams=cfg["hp"],
            parameterization=cfg["par"],
            time_convention=cfg["tconv"],
            n_svd=int(cfg["n_svd"]),
            eim=bool(cfg["eim"]),
        )


if __name__ == "__main__":
    main()

