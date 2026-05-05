from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


PARAM_NAMES = [
    "ln_Mc",
    "eta",
    "chi1",
    "chi2",
    "tc",
    "phic",
    "ln_dL",
    "kappa_s",
    "H0",
    "Lambda_tilde",
]
FINITE_SIZE_PARAMS = ["kappa_s", "H0", "Lambda_tilde"]
DEFAULT_STEPS = {
    "ln_Mc": 1e-7,
    "eta": 1e-6,
    "chi1": 1e-6,
    "chi2": 1e-6,
    "tc": 1e-3,
    "phic": 1e-5,
    "ln_dL": 1e-7,
    "kappa_s": 1e-5,
    "H0": 1e-5,
    "Lambda_tilde": 1e-3,
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inner(f, h1, h2, Sn):
    df = f[1] - f[0]
    valid = np.isfinite(Sn) & (Sn > 0.0)
    return 4.0 * np.sum(np.where(valid, h1 * np.conj(h2) / Sn, 0.0)) * df


def norm(f, h, Sn):
    return float(np.sqrt(max(np.real(inner(f, h, h, Sn)), 0.0)))


def psd_gwbenchmarks_aligo(f, f_low=15.0):
    try:
        from pycbc.psd import aLIGOZeroDetHighPower
    except ImportError as exc:
        raise ImportError("PyCBC is required for the gwBenchmarks aLIGO PSD.") from exc

    f = np.asarray(f, dtype=float)
    df = float(f[1] - f[0])
    flen = int(np.ceil(np.nanmax(f) / df)) + 1
    psd = np.asarray(aLIGOZeroDetHighPower(flen, df, f_low), dtype=float)
    idx = np.rint(f / df).astype(int)
    Sn = np.full_like(f, np.inf, dtype=float)
    valid = (f >= f_low) & np.isfinite(f) & (idx >= 0) & (idx < len(psd))
    Sn[valid] = psd[idx[valid]]
    return np.where((Sn > 0.0) & np.isfinite(Sn), Sn, np.inf)


def max_overlap(f, h_ref, h_cand, Sn):
    df = f[1] - f[0]
    valid = np.isfinite(Sn) & (Sn > 0.0)
    nrm = norm(f, h_ref, Sn) * norm(f, h_cand, Sn)
    if nrm <= 0.0 or not np.any(valid):
        return 0.0
    a = np.where(valid, h_ref * np.conj(h_cand) / Sn, 0.0)
    n = 1
    while n < 8 * len(a):
        n *= 2
    corr = np.fft.ifft(a, n=n)
    val = 4.0 * df * n * float(np.max(np.abs(corr))) / nrm
    return float(np.clip(val, 0.0, 1.0))


def phase_overlap(f, h_ref, h_cand, Sn):
    nrm = norm(f, h_ref, Sn) * norm(f, h_cand, Sn)
    if nrm <= 0.0:
        return 0.0
    return float(np.clip(abs(inner(f, h_ref, h_cand, Sn)) / nrm, 0.0, 1.0))


def theta_from_case(case):
    return np.array(
        [
            np.log(case["Mc"]),
            case["eta"],
            case.get("chi1", 0.0),
            case.get("chi2", 0.0),
            case.get("tc", 0.0),
            case.get("phic", 0.0),
            np.log(case["dL"]),
            case.get("kappa_s", 1.0),
            case.get("H0", 0.0),
            case.get("Lambda_tilde", 0.0),
        ],
        dtype=float,
    )


def params_from_theta(theta, base_case, f_low, wf_kwargs):
    params = {
        "Mc": float(np.exp(theta[0])),
        "eta": float(theta[1]),
        "chi1": float(theta[2]),
        "chi2": float(theta[3]),
        "tc": float(theta[4]),
        "phic": float(theta[5]),
        "dL": float(np.exp(theta[6])),
        "kappa_s": float(theta[7]),
        "H0": float(theta[8]),
        "Lambda_tilde": float(theta[9]),
        "kappa_a": base_case.get("kappa_a", 0.0),
        "H1E": base_case.get("H1E", 0.0),
        "H1B": base_case.get("H1B", 0.0),
        "H1E_bar": base_case.get("H1E_bar", 0.0),
        "H1B_bar": base_case.get("H1B_bar", 0.0),
        "f_low": f_low,
    }
    params.update(wf_kwargs)
    return params


def h_from_theta(module, f, theta, base_case, f_low, wf_kwargs):
    return module.h_of_f(f, **params_from_theta(theta, base_case, f_low, wf_kwargs))


def finite_derivs(module, f, theta, base_case, f_low, wf_kwargs):
    derivs = np.zeros((len(PARAM_NAMES), len(f)), dtype=complex)
    for i, name in enumerate(PARAM_NAMES):
        eps = DEFAULT_STEPS[name]
        tp = theta.copy()
        tm = theta.copy()
        tp[i] += eps
        tm[i] -= eps
        hp = h_from_theta(module, f, tp, base_case, f_low, wf_kwargs)
        hm = h_from_theta(module, f, tm, base_case, f_low, wf_kwargs)
        derivs[i] = (hp - hm) / (2.0 * eps)
    return derivs


def fisher(f, derivs, Sn):
    n = derivs.shape[0]
    G = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            val = float(np.real(inner(f, derivs[i], derivs[j], Sn)))
            G[i, j] = val
            G[j, i] = val
    return G


def inv_fisher(G):
    if not np.all(np.isfinite(G)):
        return np.full_like(G, np.nan), float("inf"), True
    try:
        cond = float(np.linalg.cond(G))
    except np.linalg.LinAlgError:
        return np.full_like(G, np.nan), float("inf"), True
    try:
        if cond < 1e15:
            return np.linalg.inv(G), cond, False
        return np.linalg.pinv(G, rcond=1e-15), cond, True
    except np.linalg.LinAlgError:
        return np.linalg.pinv(G, rcond=1e-15), cond, True


def bias_score(module, f, h_ref, h_cand, Sn, case, f_low, wf_kwargs):
    try:
        theta = theta_from_case(case)
        derivs = finite_derivs(module, f, theta, case, f_low, wf_kwargs)
        if not np.all(np.isfinite(derivs)):
            raise FloatingPointError("non-finite candidate derivatives")
        G = fisher(f, derivs, Sn)
        C, cond, used_pinv = inv_fisher(G)
        if not np.all(np.isfinite(C)):
            raise FloatingPointError("non-finite covariance")
        delta_h = h_ref - h_cand
        rhs = np.array(
            [float(np.real(inner(f, derivs[i], delta_h, Sn))) for i in range(len(PARAM_NAMES))]
        )
        dtheta = C @ rhs
        out = {
            "bias_status": "ok",
            "fisher_condition_number": cond,
            "fisher_used_pinv": used_pinv,
            "delta_sys": {name: float(dtheta[i]) for i, name in enumerate(PARAM_NAMES)},
        }
        for name in FINITE_SIZE_PARAMS:
            idx = PARAM_NAMES.index(name)
            sigma2 = float(C[idx, idx])
            sigma = float(np.sqrt(sigma2)) if sigma2 > 0.0 else np.nan
            delta = float(dtheta[idx])
            out[f"sigma_{name}_cand"] = sigma
            out[f"delta_sys_{name}"] = delta
            out[f"abs_bias_over_sigma_{name}"] = (
                float(abs(delta) / sigma) if np.isfinite(sigma) and sigma > 0.0 else None
            )
        return out
    except Exception as exc:
        out = {
            "bias_status": "failed",
            "bias_error": f"{type(exc).__name__}: {exc}",
            "fisher_condition_number": None,
            "fisher_used_pinv": None,
            "delta_sys": None,
        }
        for name in FINITE_SIZE_PARAMS:
            out[f"sigma_{name}_cand"] = None
            out[f"delta_sys_{name}"] = None
            out[f"abs_bias_over_sigma_{name}"] = None
        return out


def _sample_cases(rng, n, prefix, ranges):
    cases = []
    for i in range(n):
        case = {"id": f"{prefix}_{i:03d}"}
        for key, (lo, hi) in ranges.items():
            case[key] = float(rng.uniform(lo, hi))
        cases.append(case)
    return cases


def make_cases(smoke=False, n_cases=200):
    if smoke:
        return [
            {
                "id": "bns_gw170817_like",
                "Mc": 1.186,
                "eta": 0.249,
                "dL": 40.0,
                "chi1": 0.02,
                "chi2": 0.01,
                "kappa_s": 4.0,
                "kappa_a": 0.0,
                "H0": 0.0,
                "Lambda_tilde": 456.0,
            }
        ]

    rng = np.random.default_rng(241000294)
    n_bns = n_cases // 2
    n_bbh = (3 * n_cases) // 10
    n_intermediate = n_cases - n_bns - n_bbh
    cases = []
    cases.extend(
        _sample_cases(
            rng,
            n_bns,
            "bns",
            {
                "Mc": (1.05, 1.55),
                "eta": (0.22, 0.2495),
                "dL": (40.0, 350.0),
                "chi1": (-0.05, 0.05),
                "chi2": (-0.05, 0.05),
                "kappa_s": (2.0, 12.0),
                "kappa_a": (-2.0, 2.0),
                "H0": (0.0, 0.5),
                "Lambda_tilde": (50.0, 900.0),
            },
        )
    )
    cases.extend(
        _sample_cases(
            rng,
            n_bbh,
            "bbh",
            {
                "Mc": (8.0, 45.0),
                "eta": (0.12, 0.249),
                "dL": (200.0, 2000.0),
                "chi1": (-0.8, 0.8),
                "chi2": (-0.8, 0.8),
                "kappa_s": (0.8, 1.3),
                "kappa_a": (-0.15, 0.15),
                "H0": (0.5, 1.5),
                "Lambda_tilde": (0.0, 20.0),
            },
        )
    )
    cases.extend(
        _sample_cases(
            rng,
            n_intermediate,
            "intermediate",
            {
                "Mc": (2.0, 8.0),
                "eta": (0.12, 0.245),
                "dL": (100.0, 1000.0),
                "chi1": (-0.5, 0.8),
                "chi2": (-0.2, 0.4),
                "kappa_s": (1.0, 5.0),
                "kappa_a": (-1.0, 1.0),
                "H0": (0.0, 1.2),
                "Lambda_tilde": (0.0, 250.0),
            },
        )
    )
    return cases


def eval_case(ref, cand, detector, case, wf_kwargs, compute_bias=True):
    f = np.arange(detector["f_low"], detector["f_high"], detector["df"])
    if detector["psd"] == "gwbenchmarks_aligo":
        Sn = psd_gwbenchmarks_aligo(f, f_low=detector["f_low"])
    else:
        Sn = getattr(ref, detector["psd"])(f)
    params = {k: v for k, v in case.items() if k != "id"}
    params.update(wf_kwargs)
    params["f_low"] = detector["f_low"]
    row = {"detector": detector["name"], "case_id": case["id"]}
    try:
        h_ref = ref.h_of_f(f, **params)
        h_cand = cand.h_of_f(f, **params)
        if not (np.all(np.isfinite(h_ref)) and np.all(np.isfinite(h_cand))):
            raise FloatingPointError("non-finite waveform values")
        oo = max_overlap(f, h_ref, h_cand, Sn)
        op = phase_overlap(f, h_ref, h_cand, Sn)
        nr = norm(f, h_ref, Sn)
        nc = norm(f, h_cand, Sn)
        row.update(
            {
                "status": "ok",
                "error": None,
                "overlap_opt": oo,
                "mismatch_opt": 1.0 - oo,
                "overlap_phase": op,
                "mismatch_phase": 1.0 - op,
                "snr_ratio": (nc / nr) if nr > 0.0 else None,
                "n_ref": nr,
                "n_cand": nc,
            }
        )
        if compute_bias:
            row.update(bias_score(cand, f, h_ref, h_cand, Sn, case, detector["f_low"], wf_kwargs))
        return row
    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "overlap_opt": 0.0,
                "mismatch_opt": 1.0,
                "overlap_phase": 0.0,
                "mismatch_phase": 1.0,
                "snr_ratio": None,
                "n_ref": None,
                "n_cand": None,
            }
        )
        if compute_bias:
            row.update(bias_score(cand, f, np.zeros_like(f), np.zeros_like(f), Sn, case, detector["f_low"], wf_kwargs))
            row["bias_status"] = "failed"
            row["bias_error"] = row["error"]
        return row


def _finite_metric(rows, key):
    vals = np.array([r.get(key) for r in rows if r.get(key) is not None], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return None
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.9)),
        "max": float(np.max(vals)),
    }


def summarize(label, candidate, rows, smoke, wf_kwargs, detector_set):
    mm = np.array([r["mismatch_opt"] for r in rows], dtype=float)
    mph = np.array([r["mismatch_phase"] for r in rows], dtype=float)
    snr = np.array([r["snr_ratio"] for r in rows if r.get("snr_ratio") is not None], dtype=float)
    return {
        "label": label,
        "candidate": str(candidate),
        "mode": "smoke" if smoke else "full",
        "detector_set": detector_set,
        "waveform_kwargs": wf_kwargs,
        "n_evaluations": len(rows),
        "n_source_cases": len({r["case_id"] for r in rows}),
        "n_failed_evaluations": sum(r.get("status") != "ok" for r in rows),
        "n_bias_failures": sum(r.get("bias_status") == "failed" for r in rows),
        "mean_mismatch_opt": float(np.mean(mm)),
        "median_mismatch_opt": float(np.median(mm)),
        "p90_mismatch_opt": float(np.quantile(mm, 0.9)),
        "max_mismatch_opt": float(np.max(mm)),
        "mean_mismatch_phase": float(np.mean(mph)),
        "median_mismatch_phase": float(np.median(mph)),
        "median_abs_log10_snr_ratio": float(np.median(np.abs(np.log10(snr)))) if len(snr) else None,
        "bias_over_sigma": {
            name: _finite_metric(rows, f"abs_bias_over_sigma_{name}") for name in FINITE_SIZE_PARAMS
        },
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-bias", action="store_true")
    parser.add_argument("--n-cases", type=int, default=200)
    parser.add_argument(
        "--detector-set",
        choices=["network", "gwbenchmarks_aligo"],
        default="gwbenchmarks_aligo",
        help="Use the original multi-detector setup or the gwBenchmarks aLIGOZeroDetHighPower PSD.",
    )
    args = parser.parse_args()

    ref = load_module(args.repo_root / "src" / "finite_size" / "waveform.py", "finite_size_reference_eval")
    cand = load_module(args.candidate, "finite_size_candidate_eval")
    if args.detector_set == "gwbenchmarks_aligo":
        detectors = [
            {
                "name": "aLIGOZeroDetHighPower",
                "psd": "gwbenchmarks_aligo",
                "f_low": 15.0,
                "f_high": 990.0,
                "df": 0.125,
            }
        ]
    else:
        detectors = [
            {"name": "aLIGO", "psd": "psd_aLIGO", "f_low": 20.0, "f_high": 2048.0, "df": 0.125},
            {"name": "Voyager", "psd": "psd_Voyager", "f_low": 10.0, "f_high": 2048.0, "df": 0.125},
            {"name": "ET-D", "psd": "psd_ET_D", "f_low": 5.0, "f_high": 2048.0, "df": 0.125},
            {"name": "CE_20km", "psd": "psd_CE_20km", "f_low": 5.0, "f_high": 2048.0, "df": 0.125},
            {"name": "CE_40km", "psd": "psd_CE_40km", "f_low": 5.0, "f_high": 2048.0, "df": 0.125},
        ]
    wf_kwargs = {
        "fmax_over_fisco": 1.0,
        "sigma_taper_over_fisco": 0.01,
        "phase_only": False,
    }
    rows = []
    for detector in detectors:
        for case in make_cases(smoke=args.smoke, n_cases=args.n_cases):
            rows.append(eval_case(ref, cand, detector, case, wf_kwargs, compute_bias=not args.skip_bias))
    out = summarize(args.label, args.candidate, rows, args.smoke, wf_kwargs, args.detector_set)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(args.output)
    print(
        json.dumps(
            {
                key: out[key]
                for key in [
                    "n_evaluations",
                    "n_failed_evaluations",
                    "n_bias_failures",
                    "mean_mismatch_opt",
                    "median_mismatch_opt",
                    "bias_over_sigma",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
