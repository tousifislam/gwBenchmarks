from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

# Ensure repo root (containing gwbenchmarks/) is importable when run from nested results.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_gpt52")

from gwbenchmarks import plot_settings  # noqa: E402
from gwbenchmarks.benchmarks.analytic import AnalyticBench  # noqa: E402


QParam = Literal["q", "eta", "delta_m"]
AmpKind = Literal["exp_stitch", "exp_stitch_sharp", "gauss_mod"]


@dataclasses.dataclass(frozen=True)
class AnalyticSim:
    sxs_id: str
    q: float
    t: np.ndarray
    h: np.ndarray


@dataclasses.dataclass(frozen=True)
class AnalyticSplit:
    sims: list[AnalyticSim]
    dt_geometric: float


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _json_dump(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def _now_str() -> str:
    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M")


def load_analytic_split(path: str | Path) -> AnalyticSplit:
    path = Path(path)
    with h5py.File(path, "r") as f:
        dt = float(f.attrs.get("dt_geometric", 0.1))
        sims_g = f["sims"]
        sims: list[AnalyticSim] = []
        for sxs_id in sims_g.keys():
            g = sims_g[sxs_id]
            q = float(g.attrs["q"])
            t = np.asarray(g["t"][:], dtype=np.float64)
            hr = np.asarray(g["h22_real"][:], dtype=np.float64)
            hi = np.asarray(g["h22_imag"][:], dtype=np.float64)
            h = hr + 1j * hi
            sims.append(AnalyticSim(sxs_id=str(sxs_id), q=q, t=t, h=h))
    sims = sorted(sims, key=lambda s: s.q)
    return AnalyticSplit(sims=sims, dt_geometric=dt)


def q_feature(q: float | np.ndarray, kind: QParam) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if kind == "q":
        return q
    if kind == "eta":
        return q / (1.0 + q) ** 2
    if kind == "delta_m":
        return (q - 1.0) / (q + 1.0)
    raise ValueError(f"Unknown q feature: {kind}")


def _logcosh(x: np.ndarray) -> np.ndarray:
    # Stable log(cosh(x)) for large |x|
    x = np.asarray(x, dtype=np.float64)
    return np.logaddexp(x, -x) - math.log(2.0)


def build_waveform_from_params(
    t: np.ndarray,
    *,
    log_a0: float,
    log_tau_grow: float,
    log_tau_decay: float,
    phi0: float,
    omega0: float,
    omega1: float,
    omega2: float,
    amp_kind: AmpKind,
    transition_w: float,
    log_sigma: float | None = None,
) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    a0 = float(np.exp(log_a0))
    tau_g = float(np.exp(log_tau_grow))
    tau_d = float(np.exp(log_tau_decay))

    s = 0.5 * (1.0 + np.tanh(t / float(transition_w)))
    e_insp = np.clip(t / max(tau_g, 1e-6), -80.0, 80.0)
    e_rd = np.clip(-t / max(tau_d, 1e-6), -80.0, 80.0)
    a_insp = a0 * np.exp(e_insp)
    a_rd = a0 * np.exp(e_rd)
    a = (1.0 - s) * a_insp + s * a_rd

    if amp_kind == "gauss_mod":
        if log_sigma is None:
            raise ValueError("gauss_mod requires log_sigma")
        sig = float(np.exp(log_sigma))
        a = a * np.exp(-0.5 * (t / max(sig, 1e-6)) ** 2)

    x = t / float(transition_w)
    phi = float(phi0) + float(omega0) * t + 0.5 * float(omega1) * (t**2) + float(omega2) * float(transition_w) * _logcosh(x)
    return a * np.exp(-1j * phi)


def _sample_indices(t: np.ndarray, *, n_uniform: int = 900, n_peak: int = 700) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    if n <= (n_uniform + n_peak):
        return np.arange(n, dtype=int)

    # Uniform over full range
    iu = np.linspace(0, n - 1, n_uniform, dtype=int)
    # Denser near peak region [-300, 300]
    mask = (t >= -300.0) & (t <= 300.0)
    idx = np.where(mask)[0]
    if idx.size >= 2:
        ip = np.linspace(idx.min(), idx.max(), n_peak, dtype=int)
    else:
        ip = np.linspace(0, n - 1, n_peak, dtype=int)
    out = np.unique(np.concatenate([iu, ip]))
    return out


def _initial_guesses(sim: AnalyticSim) -> tuple[float, float, float, float, float, float, float, float | None]:
    t = sim.t
    h = sim.h
    amp = np.abs(h)
    a0 = float(np.maximum(amp.max(), 1e-12))

    # Growth/decay time estimates from two points (heuristic)
    def _tau_from(t0, a_t0):
        a_t0 = float(max(a_t0, 1e-12))
        ratio = a_t0 / a0
        ratio = float(np.clip(ratio, 1e-12, 1.0))
        return float(-t0 / max(math.log(ratio), 1e-6))

    # pick indices near -1000 and +50 where available
    t_neg = -1000.0
    t_pos = 50.0
    a_neg = float(np.interp(t_neg, t, amp))
    a_pos = float(np.interp(t_pos, t, amp))
    tau_g = float(np.clip(_tau_from(t_neg, a_neg), 50.0, 5000.0))
    tau_d = float(np.clip(_tau_from(t_pos, a_pos), 5.0, 500.0))

    # Phase at t=0 (wrapped; global phase offset is unidentifiable in mismatch anyway)
    phase_wrapped = -np.angle(h)
    i0 = int(np.argmin(np.abs(t - 0.0)))
    phi0 = float(phase_wrapped[i0])
    # local derivative in a window
    i1 = max(1, i0 - 10)
    i2 = min(len(t) - 2, i0 + 10)
    if i2 <= i1:
        omega0 = 0.1
    else:
        phase_u = np.unwrap(phase_wrapped[i1 : i2 + 1])
        omega0 = float(np.median(np.gradient(phase_u, t[i1:i2 + 1])))
        omega0 = float(np.clip(omega0, 0.01, 0.5))
    omega1 = 0.0
    omega2 = 0.0
    log_sigma = None
    return math.log(a0), math.log(tau_g), math.log(tau_d), phi0, omega0, omega1, omega2, log_sigma


def fit_per_sim_params(
    sim: AnalyticSim,
    *,
    amp_kind: AmpKind,
    transition_w: float,
    max_nfev: int = 500,
) -> dict[str, float]:
    from scipy.optimize import least_squares

    t = sim.t
    h = sim.h
    idx = _sample_indices(t)
    ts = t[idx]
    hs = h[idx]
    wts = np.sqrt(np.clip(np.abs(hs), 1e-12, np.inf))

    log_a0, log_tg, log_td, phi0, omega0, omega1, omega2, log_sigma0 = _initial_guesses(sim)
    if amp_kind == "gauss_mod":
        # broad gaussian by default
        log_sigma0 = math.log(1200.0)

    def pack(p: dict[str, float]) -> np.ndarray:
        arr = [p["log_a0"], p["log_tau_grow"], p["log_tau_decay"], p["phi0"], p["omega0"], p["omega1"], p["omega2"]]
        if amp_kind == "gauss_mod":
            arr.append(p["log_sigma"])
        return np.asarray(arr, dtype=np.float64)

    def unpack(x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        out = {
            "log_a0": float(x[0]),
            "log_tau_grow": float(x[1]),
            "log_tau_decay": float(x[2]),
            "phi0": float(x[3]),
            "omega0": float(x[4]),
            "omega1": float(x[5]),
            "omega2": float(x[6]),
        }
        if amp_kind == "gauss_mod":
            out["log_sigma"] = float(x[7])
        return out

    p0 = {
        "log_a0": float(log_a0),
        "log_tau_grow": float(log_tg),
        "log_tau_decay": float(log_td),
        "phi0": float(phi0),
        "omega0": float(omega0),
        "omega1": float(omega1),
        "omega2": float(omega2),
    }
    if amp_kind == "gauss_mod":
        p0["log_sigma"] = float(log_sigma0 or math.log(1200.0))

    lo = {
        "log_a0": math.log(1e-10),
        "log_tau_grow": math.log(20.0),
        "log_tau_decay": math.log(2.0),
        "phi0": -40.0,
        "omega0": 0.001,
        "omega1": -1e-6,
        "omega2": -2.0,
    }
    hi = {
        "log_a0": math.log(1.0),
        "log_tau_grow": math.log(20000.0),
        "log_tau_decay": math.log(5000.0),
        "phi0": 40.0,
        "omega0": 2.0,
        "omega1": 1e-6,
        "omega2": 2.0,
    }
    if amp_kind == "gauss_mod":
        lo["log_sigma"] = math.log(50.0)
        hi["log_sigma"] = math.log(20000.0)

    lb = pack(lo)
    ub = pack(hi)

    def resid(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        h_pred = build_waveform_from_params(
            ts,
            log_a0=p["log_a0"],
            log_tau_grow=p["log_tau_grow"],
            log_tau_decay=p["log_tau_decay"],
            phi0=p["phi0"],
            omega0=p["omega0"],
            omega1=p["omega1"],
            omega2=p["omega2"],
            amp_kind=amp_kind,
            transition_w=transition_w,
            log_sigma=p.get("log_sigma"),
        )
        d = (h_pred - hs)
        return np.concatenate([wts * d.real, wts * d.imag])

    res = least_squares(resid, pack(p0), bounds=(lb, ub), max_nfev=int(max_nfev), xtol=1e-10, ftol=1e-10, gtol=1e-10)
    return unpack(res.x)


def fit_polynomials(x: np.ndarray, y: np.ndarray, degree: int) -> list[float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    coef = np.polyfit(x, y, deg=int(degree))
    return [float(c) for c in coef.tolist()]


def eval_poly(coefs: list[float], x: np.ndarray) -> np.ndarray:
    return np.polyval(np.asarray(coefs, dtype=np.float64), np.asarray(x, dtype=np.float64))


def build_waveform_from_polys(
    t: np.ndarray,
    *,
    q: float,
    qparam: QParam,
    amp_kind: AmpKind,
    transition_w: float,
    poly_degree: int,
    poly_map: dict[str, list[float]],
) -> np.ndarray:
    x = float(q_feature(q, qparam))
    log_a0 = float(eval_poly(poly_map["log_a0"], x))
    log_tg = float(eval_poly(poly_map["log_tau_grow"], x))
    log_td = float(eval_poly(poly_map["log_tau_decay"], x))
    phi0 = float(eval_poly(poly_map["phi0"], x))
    omega0 = float(eval_poly(poly_map["omega0"], x))
    omega1 = float(eval_poly(poly_map["omega1"], x))
    omega2 = float(eval_poly(poly_map["omega2"], x))
    log_sigma = float(eval_poly(poly_map["log_sigma"], x)) if (amp_kind == "gauss_mod") else None
    return build_waveform_from_params(
        t,
        log_a0=log_a0,
        log_tau_grow=log_tg,
        log_tau_decay=log_td,
        phi0=phi0,
        omega0=omega0,
        omega1=omega1,
        omega2=omega2,
        amp_kind=amp_kind,
        transition_w=transition_w,
        log_sigma=log_sigma,
    )


def append_changelog_entry(work_dir: Path, entry_md: str) -> None:
    ch = work_dir / "CHANGELOG.md"
    _ensure_dir(ch.parent)
    with open(ch, "a") as f:
        if not entry_md.endswith("\n"):
            entry_md += "\n"
        f.write(entry_md)


def update_analytic_comparison(work_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_settings.apply()
    models_dir = work_dir / "models"
    comp_dir = work_dir / "comparison"
    _ensure_dir(comp_dir)

    model_dirs = sorted([p for p in models_dir.iterdir() if p.is_dir()])
    rows: list[dict[str, Any]] = []
    error_data: dict[str, Any] = {}
    all_exprs: list[dict[str, Any]] = []
    for md in model_dirs:
        score_path = md / "scorecard.json"
        if not score_path.exists():
            continue
        sc = json.loads(score_path.read_text())
        rows.append(sc)
        name = sc.get("display_name", md.name)
        err_path = md / "saved_model" / "per_sample_errors.json"
        if err_path.exists():
            error_data[name] = json.loads(err_path.read_text())
        expr_txt = md / "expression.txt"
        if expr_txt.exists():
            all_exprs.append(
                {
                    "approach": name,
                    "expression": expr_txt.read_text(),
                    "complexity": sc.get("n_terms", sc.get("n_params", 0)),
                    "loss": float(sc.get("loss", math.inf)),
                }
            )
        # Symbolic extra expressions (Pareto fronts)
        expr_json = md / "saved_model" / "expressions.json"
        if expr_json.exists():
            try:
                obj = json.loads(expr_json.read_text())
            except Exception:
                obj = None
            if isinstance(obj, list):
                for e in obj[:200]:
                    all_exprs.append(
                        {
                            "approach": name,
                            "expression": str(e.get("sympy_format") or e.get("equation") or e.get("expression") or ""),
                            "complexity": e.get("complexity"),
                            "loss": e.get("loss"),
                        }
                    )

    rows_sorted = sorted(rows, key=lambda r: float(r.get("loss", math.inf)))
    _json_dump(comp_dir / "summary_table.json", rows_sorted)
    _json_dump(comp_dir / "error_data.json", error_data)
    if rows_sorted:
        _json_dump(comp_dir / "best_model.json", rows_sorted[0])
    _json_dump(comp_dir / "all_expressions.json", all_exprs)

    if not rows:
        return

    labels = [r.get("display_name", r.get("approach", "")) for r in rows]
    losses = np.asarray([float(r["loss"]) for r in rows], dtype=float)
    runtimes = np.asarray([float(r.get("runtime_ms", math.nan)) for r in rows], dtype=float)

    # Loss-only
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    x = np.arange(len(losses))
    ax.scatter(x, losses, s=18, color=plot_settings.COLORS["blue"])
    ax.set_xlabel("Approach #")
    ax.set_ylabel("Loss (mean FD mismatch)")
    ax.set_yscale("log")
    for i, lo in enumerate(losses):
        ax.text(i, lo, str(i + 1), fontsize=7, ha="center", va="bottom")
    fig.savefig(comp_dir / "loss_only_comparison.png")
    fig.savefig(comp_dir / "loss_only_comparison.pdf")
    plt.close(fig)

    # Pareto
    fig, ax = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax.scatter(runtimes, losses, s=25, color=plot_settings.COLORS["red"], alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Eval time per time-sample (ms)")
    ax.set_ylabel("Loss (mean FD mismatch)")
    for rt, lo, lab in zip(runtimes, losses, labels):
        ax.text(rt, lo, lab.split(" (")[0], fontsize=6, ha="left", va="bottom")
    fig.savefig(comp_dir / "pareto_accuracy_speed.png")
    fig.savefig(comp_dir / "pareto_accuracy_speed.pdf")
    plt.close(fig)

    # Progress
    order = sorted(rows, key=lambda r: int(r.get("approach_number", 9999)))
    loss_o = np.asarray([float(r["loss"]) for r in order], dtype=float)
    rt_o = np.asarray([float(r.get("runtime_ms", math.nan)) for r in order], dtype=float)
    xs = np.arange(1, len(order) + 1)
    fig, ax1 = plt.subplots(figsize=plot_settings.figsize(cols=2, aspect=0.55))
    ax1.plot(xs, loss_o, marker="o", ms=3, lw=1.0, color=plot_settings.COLORS["blue"])
    ax1.set_xlabel("Approach #")
    ax1.set_ylabel("Loss")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(xs, np.clip(rt_o, 1e-9, np.inf), marker="s", ms=3, lw=1.0, color=plot_settings.COLORS["orange"])
    ax2.set_yscale("log")
    ax2.set_ylabel("Runtime (ms)")
    fig.savefig(comp_dir / "progress.png")
    fig.savefig(comp_dir / "progress.pdf")
    plt.close(fig)

    # Error histograms (small multiples), per-sim mismatch distributions
    if error_data:
        names = list(error_data.keys())
        n = len(names)
        ncols = 4
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(plot_settings.DOUBLE_COL, 0.9 * nrows), squeeze=False)
        for j, name in enumerate(names):
            ax = axes[j // ncols][j % ncols]
            d = error_data[name]
            tr = np.asarray(d.get("train", []), dtype=float)
            va = np.asarray(d.get("val", []), dtype=float)
            bins = 30
            if tr.size:
                ax.hist(tr, bins=bins, alpha=0.5, color=plot_settings.COLORS["blue"], label="train")
            if va.size:
                ax.hist(va, bins=bins, alpha=0.35, color=plot_settings.COLORS["red"], label="val", hatch="///")
            ax.set_yscale("log" if (tr.size + va.size) else "linear")
            ax.set_xlabel("FD mismatch")
            ax.set_ylabel("Count")
            ax.text(0.02, 0.98, name, transform=ax.transAxes, ha="left", va="top", fontsize=6)
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")
        fig.savefig(comp_dir / "error_histograms.png")
        fig.savefig(comp_dir / "error_histograms.pdf")
        plt.close(fig)


def train_and_score_closed_form(
    *,
    work_dir: Path,
    model_dir: Path,
    approach_number: int,
    display_name: str,
    category: str,
    qparam: QParam,
    amp_kind: AmpKind,
    poly_degree: int,
    transition_w: float,
) -> None:
    _ensure_dir(model_dir / "saved_model")
    _ensure_dir(model_dir / "plots")

    tr = load_analytic_split("datasets/analytic/analytic_training.h5")
    va = load_analytic_split("datasets/analytic/analytic_validation.h5")
    bench = AnalyticBench()

    # Fit per-simulation parameters (cached per amp_kind/transition_w), then polynomialize in q
    qs = np.asarray([s.q for s in tr.sims], dtype=np.float64)
    xq = q_feature(qs, qparam)

    param_names = ["log_a0", "log_tau_grow", "log_tau_decay", "phi0", "omega0", "omega1", "omega2"]
    if amp_kind == "gauss_mod":
        param_names.append("log_sigma")

    cache_dir = work_dir / "_cache"
    _ensure_dir(cache_dir)
    cache_key = f"per_sim_{amp_kind}_w{transition_w:.3f}".replace(".", "p")
    cache_path = cache_dir / f"{cache_key}.json"
    cached = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = None

    per_sim_map: dict[str, dict[str, float]] = {}
    if isinstance(cached, dict) and cached.get("param_names") == param_names and isinstance(cached.get("per_sim"), list):
        for row in cached["per_sim"]:
            if isinstance(row, dict) and "sxs_id" in row and "params" in row:
                per_sim_map[str(row["sxs_id"])] = {k: float(row["params"][k]) for k in param_names}
    else:
        rows = []
        for sim in tr.sims:
            p = fit_per_sim_params(sim, amp_kind=amp_kind, transition_w=transition_w)
            rows.append({"sxs_id": sim.sxs_id, "q": float(sim.q), "params": {k: float(p[k]) for k in param_names}})
            per_sim_map[sim.sxs_id] = {k: float(p[k]) for k in param_names}
        _json_dump(cache_path, {"param_names": param_names, "per_sim": rows})

    fitted: dict[str, list[float]] = {k: [] for k in param_names}
    for sim in tr.sims:
        p = per_sim_map[sim.sxs_id]
        for k in param_names:
            fitted[k].append(float(p[k]))

    poly_map: dict[str, list[float]] = {}
    for k in param_names:
        poly_map[k] = fit_polynomials(xq, np.asarray(fitted[k], dtype=np.float64), degree=int(poly_degree))

    # Build expression.txt with explicit numeric coefficients
    expr_lines = []
    expr_lines.append(f"# {display_name}")
    expr_lines.append("")
    if qparam == "q":
        expr_lines.append("x(q) = q")
    elif qparam == "eta":
        expr_lines.append("x(q) = eta = q/(1+q)^2")
    else:
        expr_lines.append("x(q) = delta_m = (q-1)/(q+1)")
    expr_lines.append(f"transition_w = {transition_w:.8g}")
    expr_lines.append(f"amp_kind = {amp_kind}")
    expr_lines.append("")
    for k in param_names:
        coefs = poly_map[k]
        # poly is c0*x^d + ... + cd
        expr_lines.append(f"{k}(q) = " + " + ".join([f"({c:.12e})*x(q)^{int(poly_degree - i)}" for i, c in enumerate(coefs[:-1])] + [f"({coefs[-1]:.12e})"]))
    expr_lines.append("")
    expr_lines.append("s(t) = 0.5*(1 + tanh(t/transition_w))")
    expr_lines.append("A_insp(t) = exp(log_a0(q)) * exp(t / exp(log_tau_grow(q)))")
    expr_lines.append("A_rd(t)   = exp(log_a0(q)) * exp(-t / exp(log_tau_decay(q)))")
    expr_lines.append("A(t) = (1-s(t))*A_insp(t) + s(t)*A_rd(t)")
    if amp_kind == "gauss_mod":
        expr_lines.append("A(t) = A(t) * exp(-0.5*(t/exp(log_sigma(q)))^2)")
    expr_lines.append("logcosh(z) = log(exp(z)+exp(-z)) - log(2)")
    expr_lines.append("phi(t) = phi0(q) + omega0(q)*t + 0.5*omega1(q)*t^2 + omega2(q)*transition_w*logcosh(t/transition_w)")
    expr_lines.append("h22(t;q) = A(t) * exp(-i*phi(t))")
    (model_dir / "expression.txt").write_text("\n".join(expr_lines) + "\n")

    # Score on train/val using benchmark loss
    def eval_split(split: AnalyticSplit) -> tuple[list[float], dict[str, float]]:
        per = []
        comps_sum: dict[str, float] = {}
        for sim in split.sims:
            h_pred = build_waveform_from_polys(
                sim.t,
                q=sim.q,
                qparam=qparam,
                amp_kind=amp_kind,
                transition_w=transition_w,
                poly_degree=poly_degree,
                poly_map=poly_map,
            )
            loss, comps = bench.compute_loss({"waveform": h_pred}, {"waveform": sim.h, "dt": split.dt_geometric})
            per.append(float(loss))
            for k, v in comps.items():
                comps_sum[k] = comps_sum.get(k, 0.0) + float(v)
        n = max(1, len(per))
        comps_mean = {k: v / n for k, v in comps_sum.items()}
        return per, comps_mean

    # Runtime benchmark: prediction-only time per time-sample
    t0 = time.perf_counter()
    n_pts = 0
    for sim in va.sims:
        _ = build_waveform_from_polys(
            sim.t,
            q=sim.q,
            qparam=qparam,
            amp_kind=amp_kind,
            transition_w=transition_w,
            poly_degree=poly_degree,
            poly_map=poly_map,
        )
        n_pts += int(len(sim.t))
    runtime_ms = 1000.0 * (time.perf_counter() - t0) / max(1, n_pts)

    per_tr, _ = eval_split(tr)
    per_va, comps_mean = eval_split(va)
    loss = float(np.mean(per_va)) if per_va else float("inf")

    _json_dump(model_dir / "saved_model" / "polys.json", {"qparam": qparam, "amp_kind": amp_kind, "poly_degree": int(poly_degree), "transition_w": float(transition_w), "poly_map": poly_map})
    _json_dump(model_dir / "saved_model" / "config.json", {"display_name": display_name, "category": category})
    _json_dump(model_dir / "saved_model" / "per_sample_errors.json", {"train": per_tr, "val": per_va})

    n_params = sum(len(v) for v in poly_map.values())
    scorecard = {
        "approach": "closed_form",
        "display_name": display_name,
        "approach_number": int(approach_number),
        "benchmark": "analytic",
        "agent": "gpt52",
        "category": category,
        "parameterization": qparam,
        "loss": float(loss),
        "loss_components": comps_mean,
        "runtime_ms": float(runtime_ms),
        "n_train": int(len(tr.sims)),
        "n_val": int(len(va.sims)),
        "n_params": int(n_params),
        "n_terms": int(n_params),
        "expression_file": "expression.txt",
        "notes": {
            "amp_kind": amp_kind,
            "poly_degree": int(poly_degree),
            "transition_w": float(transition_w),
        },
    }
    _json_dump(model_dir / "scorecard.json", scorecard)

    (model_dir / "train.py").write_text(
        "if __name__ == '__main__':\n"
        "    raise SystemExit('Use llm_agents/results/gpt52/analytic/run_all.py or run_symbolic.py')\n"
    )
    (model_dir / "predict.py").write_text(
        """from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib import build_waveform_from_polys  # noqa: E402


def predict(model_dir: str | Path, q: float, t: np.ndarray) -> Dict[str, np.ndarray]:
    model_dir = Path(model_dir)
    polys = json.loads((model_dir / "saved_model" / "polys.json").read_text())
    h = build_waveform_from_polys(
        np.asarray(t, dtype=np.float64),
        q=float(q),
        qparam=polys["qparam"],
        amp_kind=polys["amp_kind"],
        transition_w=float(polys["transition_w"]),
        poly_degree=int(polys["poly_degree"]),
        poly_map=polys["poly_map"],
    )
    return {"waveform": h}
"""
    )

    update_analytic_comparison(work_dir)
    append_changelog_entry(
        work_dir,
        f"\n## [A-{approach_number:02d}] {display_name}\n"
        f"- **Time**: {_now_str()}\n"
        "- **Benchmark**: analytic\n"
        f"- **Category**: {category}\n"
        f"- **Parameterization**: {qparam}\n"
        f"- **Loss (mean FD mismatch)**: {loss:.4e}\n"
        f"- **Eval time**: {runtime_ms:.4f} ms / time-sample\n",
    )
