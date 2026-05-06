from __future__ import annotations

from pathlib import Path
import sys

# Keep this runnable from repo root without needing llm_agents as an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib import train_and_score_svd_regressor  # noqa: E402


def main() -> None:
    work_dir = Path(__file__).resolve().parent
    agent_root = work_dir.parent
    models_dir = work_dir / "models"

    approaches = [
        # SVD / decomposition-based
        dict(
            display_name="SVD+Ridge (raw)",
            category="svd/decomposition",
            regressor_kind="ridge",
            regressor_hyperparams={"alpha": 1e-2},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        dict(
            display_name="SVD+GPR (RBF, raw)",
            category="svd/decomposition",
            regressor_kind="gpr_rbf",
            regressor_hyperparams={"length_scale": 1.0, "n_restarts": 0},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=12,
            eim=False,
        ),
        dict(
            display_name="SVD+GPR (Matern-5/2, eff)",
            category="svd/decomposition",
            regressor_kind="gpr_matern",
            regressor_hyperparams={"length_scale": 1.2, "nu": 2.5, "n_restarts": 0},
            parameterization="effective_spins_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=12,
            eim=False,
        ),
        dict(
            display_name="SVD+Poly-3 Ridge (raw)",
            category="svd/decomposition",
            regressor_kind="poly_ridge",
            regressor_hyperparams={"degree": 3, "alpha": 1e-3},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        # Interpolation / kernel
        dict(
            display_name="SVD+KRR (RBF, spherical)",
            category="interpolation/kernel",
            regressor_kind="krr_rbf",
            regressor_hyperparams={"alpha": 1e-3, "gamma": None},
            parameterization="spherical_spins_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=30,
            eim=False,
        ),
        dict(
            display_name="SVD+RBFInterpolator (eff)",
            category="interpolation/kernel",
            regressor_kind="rbf_interp",
            regressor_hyperparams={"kernel": "thin_plate_spline", "neighbors": 60, "smoothing": 1e-6},
            parameterization="effective_spins_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=16,
            eim=False,
        ),
        dict(
            display_name="SVD+kNN (eff)",
            category="interpolation/kernel",
            regressor_kind="knn",
            regressor_hyperparams={"n_neighbors": 9, "weights": "distance"},
            parameterization="effective_spins_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=30,
            eim=False,
        ),
        dict(
            display_name="EIM+KRR (raw)",
            category="svd/decomposition",
            regressor_kind="krr_rbf",
            regressor_hyperparams={"alpha": 1e-3, "gamma": None},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=24,
            eim=True,
        ),
        # Machine learning
        dict(
            display_name="SVD+MLP (sklearn, eff)",
            category="machine_learning",
            regressor_kind="mlp_sklearn",
            regressor_hyperparams={"hidden_layer_sizes": (256, 256), "alpha": 1e-5, "learning_rate_init": 2e-3, "max_iter": 1200},
            parameterization="effective_spins_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        dict(
            display_name="SVD+RandomForest (raw)",
            category="machine_learning",
            regressor_kind="rf",
            regressor_hyperparams={"n_estimators": 500, "max_depth": None},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=30,
            eim=False,
        ),
        dict(
            display_name="SVD+ExtraTrees (raw)",
            category="machine_learning",
            regressor_kind="extratrees",
            regressor_hyperparams={"n_estimators": 800, "max_depth": None},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=30,
            eim=False,
        ),
        dict(
            display_name="SVD+HistGB (raw)",
            category="machine_learning",
            regressor_kind="hgb",
            regressor_hyperparams={"max_depth": 6, "learning_rate": 0.05, "max_iter": 500},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=30,
            eim=False,
        ),
        # Additional variety
        dict(
            display_name="SVD+SVR (RBF, raw)",
            category="interpolation/kernel",
            regressor_kind="svr_rbf",
            regressor_hyperparams={"C": 20.0, "epsilon": 1e-3, "gamma": "scale"},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=12,
            eim=False,
        ),
        dict(
            display_name="SVD+Lasso (raw)",
            category="svd/decomposition",
            regressor_kind="lasso",
            regressor_hyperparams={"alpha": 1e-4},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        dict(
            display_name="SVD+ElasticNet (raw)",
            category="svd/decomposition",
            regressor_kind="elasticnet",
            regressor_hyperparams={"alpha": 2e-4, "l1_ratio": 0.3},
            parameterization="raw_7d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        dict(
            display_name="SVD+Ridge (raw+omega0)",
            category="svd/decomposition",
            regressor_kind="ridge",
            regressor_hyperparams={"alpha": 1e-2},
            parameterization="raw_plus_omega0_8d",
            time_convention="t0_at_peak",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        dict(
            display_name="Amp/Phase SVD+Ridge (eff)",
            category="svd/decomposition",
            regressor_kind="ridge",
            regressor_hyperparams={"alpha": 5e-2},
            parameterization="effective_spins_7d",
            time_convention="t0_at_peak",
            representation="amp_phase",
            n_svd=40,
            eim=False,
        ),
        dict(
            display_name="Reversed-time SVD+Ridge (raw)",
            category="svd/decomposition",
            regressor_kind="ridge",
            regressor_hyperparams={"alpha": 1e-2},
            parameterization="raw_7d",
            time_convention="reversed_time",
            representation="real_imag",
            n_svd=40,
            eim=False,
        ),
        # NOTE: Symbolic approaches (PySR, gplearn) are added by a separate runner once their
        # environment is bootstrapped. This file trains the first 18 approaches.
    ]

    for i, cfg in enumerate(approaches, start=1):
        model_dir = models_dir / (
            f"{i:02d}_"
            + cfg["display_name"]
            .lower()
            .replace("+", "_")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace("-", "_")
            .replace(",", "")
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
            regressor_kind=cfg["regressor_kind"],
            regressor_hyperparams=cfg["regressor_hyperparams"],
            parameterization=cfg["parameterization"],
            time_convention=cfg["time_convention"],
            representation=cfg["representation"],
            n_svd=int(cfg["n_svd"]),
            seed=0,
            eim=bool(cfg["eim"]),
        )


if __name__ == "__main__":
    main()
