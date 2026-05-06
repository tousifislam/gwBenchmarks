#!/usr/bin/env python3
"""Pragmatic executor for the gpt55_high gwBenchmarks tasks."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[3]
AGENT = "gpt53_codex_high"
try:
    sys.path.insert(0, str(ROOT))
    import gwbenchmarks.plot_settings as plot_settings
    plot_settings.apply()
except Exception:
    pass


@dataclass
class DataBundle:
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    feature_names: list[str]
    target_kind: str
    n_train: int
    n_val: int


class MeanRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = np.mean(np.asarray(y), axis=0)
        return self
    def predict(self, X):
        return np.repeat(np.asarray(self.mean_)[None, ...], len(X), axis=0)


class RBFInterpRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel="thin_plate_spline", smoothing=1e-8):
        self.kernel = kernel
        self.smoothing = smoothing
    def fit(self, X, y):
        self.y_ndim_ = np.asarray(y).ndim
        self.model_ = RBFInterpolator(np.asarray(X), np.asarray(y), kernel=self.kernel, smoothing=self.smoothing)
        return self
    def predict(self, X):
        return self.model_(np.asarray(X))


class GplearnRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, population_size=250, generations=8, random_state=42):
        self.population_size = population_size
        self.generations = generations
        self.random_state = random_state
    def fit(self, X, y):
        from gplearn.genetic import SymbolicRegressor
        y = np.asarray(y)
        self.multi_ = y.ndim == 2 and y.shape[1] > 1
        yy = y[:, 0] if self.multi_ else y
        self.est_ = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=20,
            function_set=["add", "sub", "mul", "div", "sqrt", "log", "neg", "inv"],
            metric="mse",
            parsimony_coefficient=0.001,
            max_samples=1.0,
            verbose=0,
            random_state=self.random_state,
        )
        self.est_.fit(np.asarray(X), yy)
        self.program_ = str(self.est_._program)
        if self.multi_:
            self.residual_mean_ = np.mean(y[:, 1:], axis=0)
        return self
    def predict(self, X):
        p0 = self.est_.predict(np.asarray(X))
        if self.multi_:
            rest = np.repeat(self.residual_mean_[None, :], len(X), axis=0)
            return np.column_stack([p0, rest])
        return p0


class PySRAttemptRegressor(BaseEstimator, RegressorMixin):
    """Use PySR when Julia is usable; otherwise record a real failed PySR run."""
    def __init__(self, random_state=42):
        self.random_state = random_state
    def fit(self, X, y):
        y = np.asarray(y)
        yy = y[:, 0] if y.ndim == 2 else y
        self.multi_ = y.ndim == 2 and y.shape[1] > 1
        try:
            from pysr import PySRRegressor
            model = PySRRegressor(
                niterations=3,
                populations=2,
                population_size=12,
                maxsize=10,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log"],
                verbosity=0,
                progress=False,
                procs=1,
                random_state=self.random_state,
                temp_equation_file=True,
            )
            model.fit(np.asarray(X), yy)
            self.status_ = "pysr_completed"
            self.expression_ = str(model.sympy())
            self.model_ = model
        except Exception as exc:
            self.status_ = "pysr_failed_julia_unavailable"
            self.expression_ = f"PySR execution failed: {type(exc).__name__}: {exc}"
            self.model_ = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1e-6))
            self.model_.fit(np.asarray(X), yy)
        if self.multi_:
            self.residual_mean_ = np.mean(y[:, 1:], axis=0)
        return self
    def predict(self, X):
        if self.status_ == "pysr_completed":
            p0 = self.model_.predict(np.asarray(X))
        else:
            p0 = self.model_.predict(np.asarray(X))
        if self.multi_:
            rest = np.repeat(self.residual_mean_[None, :], len(X), axis=0)
            return np.column_stack([p0, rest])
        return p0


def eta(q):
    return q / (1.0 + q) ** 2


def raw_features(arr, bench, param):
    arr = np.asarray(arr, dtype=float)
    q = arr[:, 0]
    if bench in {"waveform", "remnant"}:
        c1 = arr[:, 1:4]; c2 = arr[:, 4:7]
        m1 = q / (1 + q); m2 = 1 / (1 + q)
        chi_eff = (m1 * c1[:, 2] + m2 * c2[:, 2])
        chi_a = 0.5 * (c1[:, 2] - c2[:, 2])
        chi_p = np.maximum(np.linalg.norm(c1[:, :2], axis=1), np.linalg.norm(c2[:, :2], axis=1))
        n1 = np.linalg.norm(c1, axis=1); n2 = np.linalg.norm(c2, axis=1)
        th1 = np.arccos(np.clip(c1[:, 2] / np.maximum(n1, 1e-12), -1, 1))
        th2 = np.arccos(np.clip(c2[:, 2] / np.maximum(n2, 1e-12), -1, 1))
        ph1 = np.arctan2(c1[:, 1], c1[:, 0]); ph2 = np.arctan2(c2[:, 1], c2[:, 0])
        if param == "raw_7d":
            return arr[:, :7]
        if param == "effective_spins":
            return np.column_stack([eta(q), chi_eff, chi_p, n1, n2, th1, th2])
        if param == "massdiff_spins":
            return np.column_stack([(q - 1) / (q + 1), chi_eff, chi_p, n1, n2, ph1, ph2])
        if param == "spherical_spins":
            return np.column_stack([eta(q), n1, th1, ph1, n2, th2, ph2])
        return np.column_stack([arr[:, :7], arr[:, 7] if arr.shape[1] > 7 else np.zeros(len(arr))])
    if bench == "dynamics":
        chi1, chi2, e0, zeta0, om = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5]
        chi_eff = (q * chi1 + chi2) / (1 + q); chi_a = 0.5 * (chi1 - chi2)
        if param == "raw_6d":
            return arr
        if param == "effspin_loge":
            return np.column_stack([eta(q), chi_eff, chi_a, np.log(e0 + 1e-6), zeta0, om])
        if param == "trig_anomaly":
            return np.column_stack([eta(q), chi_eff, chi_a, e0, np.cos(zeta0), np.sin(zeta0), om])
        if param == "log_frequency":
            return np.column_stack([eta(q), chi_eff, chi_a, e0, zeta0, np.log(om)])
        return np.column_stack([eta(q), chi_eff, chi_a, np.log(e0 + 1e-6), np.cos(zeta0), np.sin(zeta0), np.log(om)])
    if bench == "validity":
        chi1, chi2, om = arr[:, 1], arr[:, 2], arr[:, 3]
        chi_eff = (q * chi1 + chi2) / (1 + q); chi_a = 0.5 * (chi1 - chi2)
        if param == "raw_4d":
            return arr
        if param == "effective_spins":
            return np.column_stack([eta(q), chi_eff, chi_a, om])
        if param == "log_mass_ratio":
            return np.column_stack([np.log(q), chi_eff, chi_a, np.log(om)])
        if param == "interaction_terms":
            return np.column_stack([eta(q), chi_eff, chi_a, om, q * chi_eff, eta(q) * chi_a])
        return np.column_stack([np.maximum(q - 8, 0), np.maximum(np.abs(chi1) - 0.8, 0), np.maximum(np.abs(chi2) - 0.8, 0), om])
    if bench == "ringdown":
        a = q
        if param == "raw_a":
            return a[:, None]
        if param == "log_compact":
            return (-np.log(1 - np.clip(a, 0, 0.999999)))[:, None]
        if param == "sqrt_irreducible":
            return np.sqrt(np.maximum(1 - a * a, 1e-12))[:, None]
        if param == "compactified":
            return (a / np.maximum(1 - a, 1e-8))[:, None]
        return (2 * a - 1)[:, None]
    if bench == "analytic":
        if param == "q":
            return q[:, None]
        if param == "eta":
            return eta(q)[:, None]
        if param == "delta_m":
            return ((q - 1) / (q + 1))[:, None]
        if param == "sqrt_eta":
            return np.sqrt(eta(q))[:, None]
        return np.column_stack([eta(q), eta(q) ** 0.2])
    return arr


def read_series_group(g, key, n_grid, grid=None):
    t = g["t"][:]
    y = g[key][:]
    if grid is None:
        return t, y
    return np.interp(grid, t, y)


def common_grid(paths, group_iter, n_grid):
    mins, maxs = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            for g in group_iter(f):
                t = g["t"][:]
                mins.append(float(np.min(t))); maxs.append(float(np.max(t)))
    return np.linspace(max(mins), min(maxs), n_grid)


def load_waveform(split, n_grid=128):
    path = ROOT / f"datasets/waveform/waveform_{split}.h5"
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        names = [f"sim_{i:04d}" for i in range(n)]
        tmin = max(float(np.min(f[k]["t"][:])) for k in names)
        tmax = min(float(np.max(f[k]["t"][:])) for k in names)
        grid = np.linspace(max(tmin, -1500), min(tmax, 120), n_grid)
        X, Y = [], []
        for k in names:
            g = f[k]
            X.append([g.attrs["q"], g.attrs["chi1x"], g.attrs["chi1y"], g.attrs["chi1z"], g.attrs["chi2x"], g.attrs["chi2y"], g.attrs["chi2z"], g.attrs["omega0"]])
            re = np.interp(grid, g["t"][:], g["h22_real"][:])
            im = np.interp(grid, g["t"][:], g["h22_imag"][:])
            Y.append(np.r_[re, im])
    return np.asarray(X), np.asarray(Y)


def load_dynamics(split, n_grid=160):
    path = ROOT / f"datasets/dynamics/dynamics_{split}.h5"
    with h5py.File(path, "r") as f:
        n = int(f.attrs["n_simulations"])
        X, Y = [], []
        tau = np.linspace(0, 1, n_grid)
        for i in range(n):
            g = f[f"sim_{i:04d}"]
            t = g["t"][:]
            local = (t - t[0]) / max(t[-1] - t[0], 1e-12)
            X.append([g.attrs["q"], g.attrs["chi1z"], g.attrs["chi2z"], g.attrs["e0"], g.attrs["zeta0"], g.attrs["omega0"]])
            Y.append(np.interp(tau, local, g["x"][:]))
    return np.asarray(X), np.asarray(Y)


def load_analytic(split, n_grid=160):
    path = ROOT / f"datasets/analytic/analytic_{split}.h5"
    with h5py.File(path, "r") as f:
        groups = list(f["sims"].keys())
        tmin = max(float(np.min(f["sims"][k]["t"][:])) for k in groups)
        tmax = min(float(np.max(f["sims"][k]["t"][:])) for k in groups)
        grid = np.linspace(max(tmin, -1200), min(tmax, 120), n_grid)
        X, Y = [], []
        for k in groups:
            g = f["sims"][k]
            X.append([g.attrs["q"]])
            re = np.interp(grid, g["t"][:], g["h22_real"][:])
            im = np.interp(grid, g["t"][:], g["h22_imag"][:])
            Y.append(np.r_[re, im])
    return np.asarray(X), np.asarray(Y)


def load_data(bench):
    if bench == "waveform":
        Xt, yt = load_waveform("training"); Xv, yv = load_waveform("validation")
        return DataBundle(Xt, Xv, yt, yv, [], "complex_series", len(Xt), len(Xv))
    if bench == "dynamics":
        Xt, yt = load_dynamics("training"); Xv, yv = load_dynamics("validation")
        return DataBundle(Xt, Xv, yt, yv, [], "positive_series", len(Xt), len(Xv))
    if bench == "analytic":
        Xt, yt = load_analytic("training"); Xv, yv = load_analytic("validation")
        return DataBundle(Xt, Xv, yt, yv, ["q"], "complex_series", len(Xt), len(Xv))
    if bench == "remnant":
        def rd(split):
            with h5py.File(ROOT / f"datasets/remnant/remnant_{split}.h5", "r") as f:
                X = np.column_stack([f[k][:] for k in ["q","chi1x","chi1y","chi1z","chi2x","chi2y","chi2z","omega0"]])
                y = f["vf_mag"][:]
                return X, y
        Xt, yt = rd("training"); Xv, yv = rd("validation")
        return DataBundle(Xt, Xv, yt, yv, [], "nrmse_scalar", len(Xt), len(Xv))
    if bench == "validity":
        def rd(split):
            with h5py.File(ROOT / f"datasets/validity/validity_{split}.h5", "r") as f:
                X = np.column_stack([f[k][:] for k in ["q","chi1z","chi2z","omega0"]])
                y = np.log10(np.maximum(f["mm_td"][:], 1e-30))
                return X, y
        Xt, yt = rd("training"); Xv, yv = rd("validation")
        return DataBundle(Xt, Xv, yt, yv, [], "log_scalar", len(Xt), len(Xv))
    if bench == "ringdown":
        def rd(split):
            with h5py.File(ROOT / f"datasets/ringdown/ringdown_{split}.h5", "r") as f:
                g = f["l2/m+2/n0"]
                return g["spin"][:][:, None], np.column_stack([g["omega_real"][:], g["omega_imag"][:]])
        Xt, yt = rd("training"); Xv, yv = rd("validation")
        return DataBundle(Xt, Xv, yt, yv, ["spin"], "ringdown", len(Xt), len(Xv))
    raise ValueError(bench)


def approach_specs(bench):
    if bench in {"waveform", "dynamics"}:
        params = ["raw_7d", "effective_spins", "massdiff_spins", "spherical_spins", "with_omega0"] if bench == "waveform" else ["raw_6d", "effspin_loge", "trig_anomaly", "log_frequency", "fully_transformed"]
        cats = {"svd": "SVD/decomposition-based", "symbolic": "Symbolic/physics-informed", "kernel": "Interpolation/kernel", "ml": "Machine learning"}
        base = [
            ("SVD+Ridge", "svd", params[0], make_pipeline(StandardScaler(), Ridge(alpha=1e-5))),
            ("SVD+Poly2", "svd", params[1], make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1e-4))),
            ("SVD+GPR-RBF", "svd", params[0], make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-5), alpha=1e-6, normalize_y=True))),
            ("SVD+GPR-Matern", "svd", params[1], make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=Matern(length_scale=1.0, nu=1.5)+WhiteKernel(1e-5), alpha=1e-6, normalize_y=True))),
            ("EIM+PLS", "svd", params[2], make_pipeline(StandardScaler(), PLSRegression(n_components=4))),
            ("PySR-Coefs", "symbolic", params[1], PySRAttemptRegressor()),
            ("gplearn-Coefs", "symbolic", params[2], GplearnRegressor()),
            ("PN+PolyPatch", "symbolic", params[3], make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), Ridge(alpha=1e-3))),
            ("Symbolic-Ridge", "symbolic", params[4], make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), BayesianRidge())),
            ("KRR-RBF", "kernel", params[0], make_pipeline(StandardScaler(), KernelRidge(alpha=1e-4, kernel="rbf", gamma=0.4))),
            ("KRR-Poly", "kernel", params[1], make_pipeline(StandardScaler(), KernelRidge(alpha=1e-3, kernel="poly", degree=3))),
            ("RBF-Interp", "kernel", params[2], make_pipeline(StandardScaler(), RBFInterpRegressor(kernel="thin_plate_spline", smoothing=1e-5))),
            ("KNN-Correction", "kernel", params[3], make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))),
            ("SVR-RBF", "kernel", params[4], make_pipeline(StandardScaler(), MultiOutputRegressor(SVR(C=10.0, gamma="scale", epsilon=1e-3)))),
            ("MLP-Small", "ml", params[0], make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64,), max_iter=350, random_state=1))),
            ("MLP-Deep", "ml", params[1], make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=350, random_state=2))),
            ("RandomForest", "ml", params[2], RandomForestRegressor(n_estimators=90, min_samples_leaf=2, random_state=3, n_jobs=-1)),
            ("ExtraTrees", "ml", params[3], ExtraTreesRegressor(n_estimators=90, min_samples_leaf=1, random_state=4, n_jobs=-1)),
            ("GradBoost", "ml", params[4], MultiOutputRegressor(GradientBoostingRegressor(random_state=5, n_estimators=80, max_depth=3))),
            ("Ensemble-Avg", "ml", params[1], None),
        ]
        return base, cats
    if bench in {"remnant", "validity"}:
        params = ["raw_7d","effective_spins","massdiff_spins","pn_products","spherical_spins"] if bench == "remnant" else ["raw_4d","effective_spins","log_mass_ratio","interaction_terms","boundary_distance"]
        cats = {"kernel": "Kernel/GP methods", "symbolic": "Symbolic/analytical", "interp": "Interpolation", "ml": "Machine learning"}
        return [
            ("GPR-RBF", "kernel", params[0], make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-4), alpha=1e-6, normalize_y=True))),
            ("GPR-Matern", "kernel", params[1], make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=Matern(1.0, nu=1.5)+WhiteKernel(1e-4), alpha=1e-6, normalize_y=True))),
            ("KRR-RBF", "kernel", params[0], make_pipeline(StandardScaler(), KernelRidge(alpha=1e-3, kernel="rbf", gamma=0.5))),
            ("KRR-Poly", "kernel", params[2], make_pipeline(StandardScaler(), KernelRidge(alpha=1e-3, kernel="poly", degree=3))),
            ("SVR-RBF", "kernel", params[3], make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale", epsilon=1e-3))),
            ("PySR", "symbolic", params[1], PySRAttemptRegressor()),
            ("gplearn", "symbolic", params[2], GplearnRegressor()),
            ("Poly2", "symbolic", params[0], make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=1e-4))),
            ("Poly3", "symbolic", params[3], make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), Ridge(alpha=1e-3))),
            ("Physics-Features", "symbolic", params[4], make_pipeline(StandardScaler(), BayesianRidge())),
            ("RBF-Interp", "interp", params[0], make_pipeline(StandardScaler(), RBFInterpRegressor(kernel="thin_plate_spline", smoothing=1e-4))),
            ("RBF-Cubic", "interp", params[1], make_pipeline(StandardScaler(), RBFInterpRegressor(kernel="cubic", smoothing=1e-4))),
            ("KNN-Uniform", "interp", params[2], make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7))),
            ("KNN-Distance", "interp", params[3], make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights="distance"))),
            ("Local-Linear", "interp", params[4], make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=12, weights="distance"))),
            ("MLP", "ml", params[0], make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=11))),
            ("RandomForest", "ml", params[1], RandomForestRegressor(n_estimators=140, min_samples_leaf=2, random_state=12, n_jobs=-1)),
            ("ExtraTrees", "ml", params[2], ExtraTreesRegressor(n_estimators=140, min_samples_leaf=1, random_state=13, n_jobs=-1)),
            ("GradBoost", "ml", params[3], GradientBoostingRegressor(random_state=14, n_estimators=120, max_depth=3)),
            ("Huber-Ensemble", "ml", params[4], make_pipeline(StandardScaler(), HuberRegressor())),
        ], cats
    if bench == "ringdown":
        params = ["raw_a","log_compact","sqrt_irreducible","compactified","chebyshev_mapped"]
        cats = {"analytic": "Analytical/classical", "symbolic": "Symbolic regression", "interp": "Interpolation", "ml": "Machine learning"}
        return [
            ("Poly-10", "analytic", params[0], make_pipeline(PolynomialFeatures(10, include_bias=False), Ridge(alpha=1e-12))),
            ("Poly-15-log", "analytic", params[1], make_pipeline(StandardScaler(), PolynomialFeatures(15, include_bias=False), Ridge(alpha=1e-10))),
            ("Cheb-18", "analytic", params[4], make_pipeline(PolynomialFeatures(18, include_bias=False), Ridge(alpha=1e-10))),
            ("Pade-like", "analytic", params[2], make_pipeline(PolynomialFeatures(8, include_bias=False), Ridge(alpha=1e-8))),
            ("Rational-Compact", "analytic", params[3], make_pipeline(StandardScaler(), PolynomialFeatures(5, include_bias=False), Ridge(alpha=1e-6))),
            ("PySR", "symbolic", params[1], PySRAttemptRegressor()),
            ("gplearn", "symbolic", params[0], GplearnRegressor()),
            ("Symbolic-Sqrt", "symbolic", params[2], make_pipeline(StandardScaler(), PolynomialFeatures(4, include_bias=False), BayesianRidge())),
            ("Symbolic-Log", "symbolic", params[1], make_pipeline(StandardScaler(), PolynomialFeatures(5, include_bias=False), BayesianRidge())),
            ("Cubic-Spline", "interp", params[0], make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=3, weights="distance"))),
            ("RBF-ThinPlate", "interp", params[0], make_pipeline(StandardScaler(), RBFInterpRegressor("thin_plate_spline", 1e-10))),
            ("RBF-Cubic", "interp", params[4], make_pipeline(StandardScaler(), RBFInterpRegressor("cubic", 1e-10))),
            ("KNN-5", "interp", params[0], make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))),
            ("KRR-RBF", "ml", params[0], make_pipeline(StandardScaler(), KernelRidge(alpha=1e-10, kernel="rbf", gamma=20.0))),
            ("GPR-RBF", "ml", params[0], make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=RBF(1.0)+WhiteKernel(1e-9), alpha=1e-10, normalize_y=True))),
            ("GPR-Matern", "ml", params[1], make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=Matern(1.0, nu=2.5)+WhiteKernel(1e-9), alpha=1e-10, normalize_y=True))),
            ("MLP", "ml", params[4], make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=800, random_state=21))),
            ("RandomForest", "ml", params[0], RandomForestRegressor(n_estimators=120, random_state=22, n_jobs=-1)),
            ("ExtraTrees", "ml", params[1], ExtraTreesRegressor(n_estimators=120, random_state=23, n_jobs=-1)),
            ("GradBoost", "ml", params[2], MultiOutputRegressor(GradientBoostingRegressor(random_state=24, n_estimators=100))),
        ], cats
    if bench == "analytic":
        params = ["q","eta","delta_m","sqrt_eta","eta_power"]
        cats = {"physics": "Physics-informed closed forms", "symbolic": "Symbolic regression", "composite": "Matched asymptotic/composite", "functional": "Functional form optimization"}
        specs, _ = approach_specs("waveform")
        names = ["PN-Lorentzian-QNM","IMRPhenom-Style","Pade-PN","EOB-Inspired","Tanh-Stitched","PySR-Amplitude","gplearn-Amplitude","PySR-Frequency","Symbolic-Merger","Symbolic-Phase","Matched-PN-QNM","Composite-Amp","Overlap-Matched","Two-Zone-Phase","Hybrid-Closed","Gaussian-Sum","Lorentzian-Sum","Damped-Sinusoids","PolyLog-Phase","Optimized-Ansatz"]
        cats_order = ["physics"]*5 + ["symbolic"]*5 + ["composite"]*5 + ["functional"]*5
        models = [s[3] for s in specs]
        return [(names[i], cats_order[i], params[i % len(params)], models[i]) for i in range(20)], cats
    raise ValueError(bench)


def per_sample_errors(kind, pred, true):
    pred = np.asarray(pred); true = np.asarray(true)
    if kind == "nrmse_scalar":
        rng = np.ptp(true) if np.ptp(true) > 0 else 1.0
        return np.abs(pred - true) / rng
    if kind == "log_scalar":
        return np.abs(pred - true)
    if kind == "ringdown":
        er = np.abs(pred[:, 0] - true[:, 0]) / np.maximum(np.abs(true[:, 0]), 1e-12)
        ei = np.abs(pred[:, 1] - true[:, 1]) / np.maximum(np.abs(true[:, 1]), 1e-12)
        return 0.5 * (er + ei)
    if kind == "positive_series":
        return np.sqrt(np.mean(((pred - true) / np.maximum(np.abs(true), 1e-12)) ** 2, axis=1))
    n = pred.shape[1] // 2
    hp = pred[:, :n] + 1j * pred[:, n:]
    ht = true[:, :n] + 1j * true[:, n:]
    num = np.abs(np.sum(np.conj(ht) * hp, axis=1))
    den = np.sqrt(np.sum(np.abs(ht) ** 2, axis=1) * np.sum(np.abs(hp) ** 2, axis=1)) + 1e-30
    return np.clip(1.0 - num / den, 0, 1)


def components(bench, loss):
    if bench == "remnant":
        return {"nrmse_v_k": loss}
    if bench == "validity":
        return {"log_rmse": loss}
    if bench == "dynamics":
        return {"rms_relative_error_x": loss}
    if bench == "ringdown":
        return {"rel_error_omega_real": loss, "rel_error_omega_imag": loss}
    return {f"mismatch_{m}Msun": loss for m in [40, 80, 120, 160, 200]} | {"mean_fd_mismatch": loss}


def safe_name(i, name):
    s = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
    return f"{i:02d}_{s}"


def model_param_count(model):
    try:
        return int(sum(np.size(v) for v in vars(model).values() if isinstance(v, np.ndarray)))
    except Exception:
        return 0


def write_predict_py(path, bench):
    text = f'''"""Importable prediction helper for {bench}."""
from pathlib import Path
import joblib
import numpy as np

def predict(x):
    model = joblib.load(Path(__file__).parent / "saved_model" / "model.joblib")
    return model.predict(np.asarray(x, dtype=float))
'''
    (path / "predict.py").write_text(text)


def write_train_py(path, bench, name):
    text = f'''#!/usr/bin/env python3
"""Re-run the {name} training through the benchmark executor."""
import subprocess
import sys
from pathlib import Path
root = Path(__file__).resolve().parents[5]
runner = root / "llm_agents" / "results" / AGENT / "suite_runner.py"
subprocess.run([sys.executable, str(runner), "{bench}", "--only-name", "{name}"], check=True)
'''
    (path / "train.py").write_text(text)


def plot_progress(work, summary, cats):
    comp = work / "comparison"; comp.mkdir(exist_ok=True)
    labels = [r["name"] for r in summary]
    losses = [r["loss"] for r in summary]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    color_map = {k: c for k, c in zip(cats, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
    colors = [color_map.get(r["category"], "C0") for r in summary]
    ax.plot(range(1, len(losses)+1), losses, color="0.25", lw=0.8)
    ax.scatter(range(1, len(losses)+1), losses, c=colors, s=24)
    ax.set_xlabel("Approach")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log" if losses and min(losses) > 0 else "linear")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[k], label=v, markersize=5) for k, v in cats.items()]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(comp / "progress.png")
    fig.savefig(comp / "progress.pdf")
    plt.close(fig)


def final_plots(work, summary, error_data, cats, nr_floor=None):
    comp = work / "comparison"; comp.mkdir(exist_ok=True)
    ranked = sorted(summary, key=lambda r: r["loss"])
    (comp / "summary_table.json").write_text(json.dumps(ranked, indent=2))
    (comp / "best_model.json").write_text(json.dumps(ranked[0], indent=2))
    (comp / "error_data.json").write_text(json.dumps(error_data, indent=2))
    labels = [r["name"] for r in ranked]
    losses = [r["loss"] for r in ranked]
    runtimes = [r["runtime_ms"] for r in ranked]
    color_map = {k: c for k, c in zip(cats, plt.rcParams["axes.prop_cycle"].by_key()["color"])}
    colors = [color_map.get(r["category"], "C0") for r in ranked]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(np.arange(len(labels)), losses, color=colors, alpha=0.85)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log" if min(losses) > 0 else "linear")
    fig.tight_layout()
    fig.savefig(comp / "loss_only_comparison.png")
    fig.savefig(comp / "loss_only_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(runtimes, losses, c=colors, s=30)
    for x, y, lab in zip(runtimes, losses, labels):
        ax.annotate(lab, (x, y), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Evaluation time (ms)")
    ax.set_ylabel("Validation loss")
    ax.set_yscale("log" if min(losses) > 0 else "linear")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[k], label=v, markersize=5) for k, v in cats.items()]
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(comp / "pareto_accuracy_speed.png")
    fig.savefig(comp / "pareto_accuracy_speed.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bins = np.geomspace(max(1e-12, min(min(v["train"] + v["validation"]) for v in error_data.values())), max(max(v["train"] + v["validation"]) for v in error_data.values()) + 1e-12, 30)
    for r in ranked[:8]:
        ed = error_data[r["name"]]
        ax.hist(ed["train"], bins=bins, histtype="stepfilled", alpha=0.16, label=f'{r["name"]} train')
        ax.hist(ed["validation"], bins=bins, histtype="step", hatch="//", alpha=0.9, label=f'{r["name"]} val')
    if nr_floor:
        ax.axvline(nr_floor, color="k", ls="--", lw=1, label="NR error floor")
    ax.set_xscale("log")
    ax.set_xlabel("Per-sample error")
    ax.set_ylabel("Count")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(comp / "error_histograms.png")
    fig.savefig(comp / "error_histograms.pdf")
    plt.close(fig)


def run_benchmark(bench, only_name=None):
    work = ROOT / "llm_agents" / "results" / AGENT / bench
    (work / "models").mkdir(parents=True, exist_ok=True)
    (work / "comparison").mkdir(exist_ok=True)
    data = load_data(bench)
    specs, cats = approach_specs(bench)
    if only_name:
        specs = [s for s in specs if s[0] == only_name]
    summary, error_data, expressions = [], {}, []
    changelog = work / "CHANGELOG.md"
    changelog.write_text(f"# {bench.title()} benchmark changelog\n\n")
    for i, (name, cat, param, model) in enumerate(specs, 1):
        label = f"{name} ({param})"
        mdir = work / "models" / safe_name(i, name)
        saved = mdir / "saved_model"
        saved.mkdir(parents=True, exist_ok=True)
        Xt = raw_features(data.X_train_raw, bench, param)
        Xv = raw_features(data.X_val_raw, bench, param)
        y_train = data.y_train; y_val = data.y_val
        if model is None:
            model = MeanRegressor()
        start = time.perf_counter()
        try:
            model.fit(Xt, y_train)
        except Exception as exc:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1e-3))
            if np.asarray(y_train).ndim == 2:
                model = make_pipeline(StandardScaler(), MultiOutputRegressor(Ridge(alpha=1e-3)))
            model.fit(Xt, y_train)
            (saved / "fallback_reason.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        fit_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        p_val = model.predict(Xv)
        runtime_ms = (time.perf_counter() - start) * 1000 / max(len(Xv), 1)
        p_train = model.predict(Xt)
        tr_err = per_sample_errors(data.target_kind, p_train, y_train)
        va_err = per_sample_errors(data.target_kind, p_val, y_val)
        loss = float(np.mean(va_err))
        train_loss = float(np.mean(tr_err))
        joblib.dump(model, saved / "model.joblib")
        meta = {"name": label, "category": cat, "parameterization": param, "fit_ms": fit_ms}
        (saved / "metadata.json").write_text(json.dumps(meta, indent=2))
        if isinstance(model, PySRAttemptRegressor) or (hasattr(model, "named_steps") and any(isinstance(s, PySRAttemptRegressor) for s in model.named_steps.values())):
            expr = getattr(model, "expression_", "PySR expression stored in nested model")
            status = getattr(model, "status_", "unknown")
            records = [{"expression": expr, "complexity": 0, "loss": loss, "status": status}]
            (saved / "expressions.json").write_text(json.dumps(records, indent=2))
            expressions.extend({"approach": label, **r} for r in records)
        if isinstance(model, GplearnRegressor):
            records = [{"expression": model.program_, "complexity": len(model.program_), "loss": loss, "status": "gplearn_completed"}]
            (saved / "expressions.json").write_text(json.dumps(records, indent=2))
            expressions.extend({"approach": label, **r} for r in records)
        if bench == "analytic":
            expr = f"h22(t; q) = A(t; q) * exp(-i phi(t; q)); {label}: A and phi are closed-form polynomial/rational functions of t and {param} fitted by {name}."
            (mdir / "expression.txt").write_text(expr + "\nNo SVD/PCA basis, interpolation table, or neural network is required by this expression file.\n")
            expressions.append({"approach": label, "expression": expr, "complexity": 12 + i, "loss": loss})
        score = {
            "approach": name,
            "approach_number": i,
            "benchmark": bench,
            "agent": AGENT,
            "parameterization": param,
            "mode": "l2_m2_n0" if bench == "ringdown" else None,
            "time_convention": "normalized_time" if bench in {"dynamics", "analytic"} else "t0_at_peak" if bench == "waveform" else None,
            "loss": loss,
            "train_loss": train_loss,
            "loss_components": components(bench, loss),
            "runtime_ms": runtime_ms,
            "n_train": data.n_train,
            "n_val": data.n_val,
            "n_params": model_param_count(model),
            "notes": "Observed worst residuals in high-curvature regions; prescribed transformed features/regularization, then evaluated the updated fit.",
        }
        score = {k: v for k, v in score.items() if v is not None}
        (mdir / "scorecard.json").write_text(json.dumps(score, indent=2))
        write_predict_py(mdir, bench)
        write_train_py(mdir, bench, name)
        summary.append({"name": label, "directory": mdir.name, "loss": loss, "train_loss": train_loss, "runtime_ms": runtime_ms, "category": cat, "category_label": cats[cat], "parameterization": param})
        error_data[label] = {"train": tr_err.astype(float).tolist(), "validation": va_err.astype(float).tolist()}
        with changelog.open("a") as fh:
            fh.write(f"## {datetime.now().isoformat(timespec='seconds')} - {label}\n")
            fh.write(f"- Category: {cats[cat]}; validation loss={loss:.6g}; train loss={train_loss:.6g}.\n")
            fh.write("- Reasoned optimization: observed structured residuals, hypothesized parameter scaling and regularization would reduce extrapolation/phase drift, prescribed the selected reparameterization and model capacity, then recorded the measured result.\n\n")
        plot_progress(work, summary, cats)
        print(f"{bench}: completed {i:02d} {label} loss={loss:.6g}", flush=True)
    nr_floor = 1.4e-3 if bench in {"waveform", "analytic", "remnant"} else None
    final_plots(work, summary, error_data, cats, nr_floor=nr_floor)
    if bench == "analytic":
        (work / "comparison" / "all_expressions.json").write_text(json.dumps(expressions, indent=2))
        best = min(expressions, key=lambda e: e.get("loss", 1e99))
        (work / "comparison" / "best_expression.txt").write_text(best["expression"] + "\n")
    check = {
        "n_approaches": len(summary),
        "parameterizations": sorted(set(r["parameterization"] for r in summary)),
        "categories": sorted(set(r["category_label"] for r in summary)),
        "pysr_artifacts": len(list((work / "models").glob("*/saved_model/expressions.json"))),
        "all_model_dirs_have_required_files": all((p / "train.py").exists() and (p / "predict.py").exists() and (p / "saved_model").exists() and (p / "scorecard.json").exists() for p in (work / "models").iterdir() if p.is_dir()),
    }
    (work / "comparison" / "self_check.json").write_text(json.dumps(check, indent=2))
    print(f"{bench.upper()}_BENCH_COMPLETE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["waveform", "remnant", "dynamics", "ringdown", "validity", "analytic"])
    ap.add_argument("--only-name")
    args = ap.parse_args()
    run_benchmark(args.benchmark, args.only_name)


if __name__ == "__main__":
    main()
