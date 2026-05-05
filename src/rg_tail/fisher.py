"""Fisher matrix forecasts for the RG-tail factorized waveform.

Computes Fisher information matrices for the parameter vector:
    theta = [ln_Mc, eta, tc, phic, ln_dL, lambda_RG]

using numerical (centered finite-difference) derivatives of h(f)
from the factorized (2,2) SPA waveform in waveform.py.

The key beyond-GR parameter is lambda_RG (GR value = 1), which scales
the universal anomalous dimension of the (2,2) multipole moment.
sigma(lambda_RG) quantifies how well a given detector can constrain
deviations from the GR prediction for tail resummation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.rg_tail.waveform import (
    h_of_f,
    h_of_f_taylorf2_baseline,
    compute_snr,
    psd_aLIGO,
    psd_Voyager,
    psd_CE_20km,
    psd_ET_D,
    psd_CE_40km,
    MSUN_SEC,
    PI,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
#  Parameter definitions
# ═══════════════════════════════════════════════════════════════════════

PARAM_NAMES = ["ln_Mc", "eta", "tc", "phic", "ln_dL", "lambda_RG"]
N_PARAMS = len(PARAM_NAMES)

# Default finite-difference step sizes (chosen to balance truncation
# error vs numerical noise for double-precision waveforms).
DEFAULT_STEPS = {
    "ln_Mc":     1e-7,
    "eta":       1e-6,
    "tc":        1e-3,   # seconds
    "phic":      1e-5,   # radians
    "ln_dL":     1e-7,
    "lambda_RG": 1e-5,
}


@dataclass
class FiducialParams:
    """Fiducial source parameters for the Fisher matrix."""
    Mc: float = 28.3       # chirp mass [Msun]
    eta: float = 0.247     # symmetric mass ratio
    tc: float = 0.0        # coalescence time [s]
    phic: float = 0.0      # coalescence phase [rad]
    dL: float = 410.0      # luminosity distance [Mpc]
    lambda_RG: float = 1.0 # RG running (GR = 1)

    @property
    def ln_Mc(self) -> float:
        return np.log(self.Mc)

    @property
    def ln_dL(self) -> float:
        return np.log(self.dL)

    def to_theta(self) -> np.ndarray:
        """Convert to the Fisher parameter vector."""
        return np.array([self.ln_Mc, self.eta, self.tc,
                         self.phic, self.ln_dL, self.lambda_RG])

    @staticmethod
    def from_theta(theta: np.ndarray) -> "FiducialParams":
        """Reconstruct from Fisher parameter vector."""
        return FiducialParams(
            Mc=np.exp(theta[0]),
            eta=theta[1],
            tc=theta[2],
            phic=theta[3],
            dL=np.exp(theta[4]),
            lambda_RG=theta[5],
        )


# ═══════════════════════════════════════════════════════════════════════
#  Waveform wrapper: theta -> h(f)
# ═══════════════════════════════════════════════════════════════════════

def _h_from_theta(f: np.ndarray, theta: np.ndarray, f_low: float,
                  fmax_over_fisco: float = 1.3,
                  sigma_taper_over_fisco: float = 0.01,
                  phase_only: bool = False,
                  amplitude_model: str = "balance",
                  waveform_func: Callable | None = None) -> np.ndarray:
    """Evaluate h(f) from the Fisher parameter vector.

    theta = [ln_Mc, eta, tc, phic, ln_dL, lambda_RG]
    """
    Mc = np.exp(theta[0])
    eta = theta[1]
    tc = theta[2]
    phic = theta[3]
    dL = np.exp(theta[4])
    lambda_RG = theta[5]

    func = waveform_func if waveform_func is not None else h_of_f
    kwargs = dict(Mc=Mc, eta=eta, dL=dL, tc=tc, phic=phic,
                  lambda_RG=lambda_RG, f_low=f_low,
                  fmax_over_fisco=fmax_over_fisco,
                  sigma_taper_over_fisco=sigma_taper_over_fisco,
                  phase_only=phase_only)
    if waveform_func is None:
        kwargs["amplitude_model"] = amplitude_model
    return func(f, **kwargs)


# ═══════════════════════════════════════════════════════════════════════
#  Numerical derivatives
# ═══════════════════════════════════════════════════════════════════════

def compute_derivatives(f: np.ndarray, theta: np.ndarray, f_low: float,
                        steps: dict[str, float] | None = None,
                        fmax_over_fisco: float = 1.3,
                        sigma_taper_over_fisco: float = 0.01,
                        phase_only: bool = False,
                        amplitude_model: str = "balance",
                        waveform_func: Callable | None = None,
                        ) -> np.ndarray:
    """Centered finite-difference derivatives dh/d(theta_i).

    Returns array of shape (N_PARAMS, len(f)), complex.
    """
    if steps is None:
        steps = DEFAULT_STEPS

    wf_kw = dict(fmax_over_fisco=fmax_over_fisco,
                 sigma_taper_over_fisco=sigma_taper_over_fisco,
                 phase_only=phase_only,
                 amplitude_model=amplitude_model,
                 waveform_func=waveform_func)

    n_f = len(f)
    derivs = np.zeros((N_PARAMS, n_f), dtype=complex)
    h_fid = _h_from_theta(f, theta, f_low, **wf_kw)

    for i, name in enumerate(PARAM_NAMES):
        if name == "tc":
            derivs[i] = 1j * 2.0 * PI * f * h_fid
            continue
        if name == "phic":
            derivs[i] = -1j * h_fid
            continue
        if name == "ln_dL":
            derivs[i] = -h_fid
            continue

        eps = steps[name]

        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[i] += eps
        theta_minus[i] -= eps

        h_plus = _h_from_theta(f, theta_plus, f_low, **wf_kw)
        h_minus = _h_from_theta(f, theta_minus, f_low, **wf_kw)

        derivs[i] = (h_plus - h_minus) / (2.0 * eps)

    return derivs


# ═══════════════════════════════════════════════════════════════════════
#  Fisher matrix
# ═══════════════════════════════════════════════════════════════════════

def compute_fisher_matrix(f: np.ndarray, derivs: np.ndarray,
                          Sn: np.ndarray) -> np.ndarray:
    """Compute Fisher information matrix.

    Gamma_ij = 4 * Re * integral( dh_i^* * dh_j / S_n(f) ) df

    Parameters
    ----------
    f : frequency array [Hz]
    derivs : (N_PARAMS, len(f)) complex derivatives
    Sn : PSD values at each frequency [1/Hz]

    Returns
    -------
    (N_PARAMS, N_PARAMS) real Fisher matrix.
    """
    df = f[1] - f[0]
    valid = np.isfinite(Sn) & (Sn > 0)

    Gamma = np.zeros((N_PARAMS, N_PARAMS))

    for i in range(N_PARAMS):
        for j in range(i, N_PARAMS):
            integrand = np.where(
                valid,
                np.real(np.conj(derivs[i]) * derivs[j]) / Sn,
                0.0,
            )
            Gamma[i, j] = 4.0 * np.sum(integrand) * df
            Gamma[j, i] = Gamma[i, j]

    return Gamma


def invert_fisher(Gamma: np.ndarray) -> tuple[np.ndarray, float]:
    """Invert Fisher matrix, falling back to pseudo-inverse if ill-conditioned.

    Returns (covariance_matrix, condition_number).
    """
    cond = np.linalg.cond(Gamma)

    if cond < 1e15:
        cov = np.linalg.inv(Gamma)
    else:
        logger.warning("Fisher matrix ill-conditioned (cond=%.2e), using pinv", cond)
        cov = np.linalg.pinv(Gamma, rcond=1e-15)

    return cov, cond


# ═══════════════════════════════════════════════════════════════════════
#  High-level forecast function
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class FisherResult:
    """Results of a Fisher forecast."""
    param_names: list[str]
    fiducial: np.ndarray
    fisher_matrix: np.ndarray
    covariance: np.ndarray
    condition_number: float
    snr: float
    sigmas: dict[str, float] = field(default_factory=dict)
    detector_name: str = ""
    f_low: float = 0.0

    def sigma(self, name: str) -> float:
        """1-sigma uncertainty on a parameter."""
        idx = self.param_names.index(name)
        return np.sqrt(self.covariance[idx, idx])

    def sigma_Mc_over_Mc(self) -> float:
        """Fractional chirp mass uncertainty (sigma(ln_Mc) = sigma(Mc)/Mc)."""
        return self.sigma("ln_Mc")

    def sigma_dL_over_dL(self) -> float:
        """Fractional distance uncertainty."""
        return self.sigma("ln_dL")

    def sigma_lambda_RG(self) -> float:
        """Uncertainty on the RG running parameter."""
        return self.sigma("lambda_RG")


def run_fisher(params: FiducialParams | None = None,
               psd_func: Callable | None = None,
               detector_name: str = "aLIGO",
               f_low: float = 20.0,
               f_high: float = 2048.0,
               df: float = 1.0 / 8.0,
               steps: dict[str, float] | None = None,
               fmax_over_fisco: float = 1.3,
               sigma_taper_over_fisco: float = 0.01,
               phase_only: bool = False,
               amplitude_model: str = "balance",
               waveform_func: Callable | None = None,
               ) -> FisherResult:
    """Run a complete Fisher forecast.

    Parameters
    ----------
    params : fiducial source parameters (default: GW150914-like)
    psd_func : callable psd_func(f) -> Sn(f). If None, chosen by detector_name.
    detector_name : "aLIGO", "Voyager", "ET-D", or "CE_40km" (used if psd_func is None)
    f_low, f_high, df : frequency grid parameters
    steps : finite-difference step sizes (default: DEFAULT_STEPS)
    fmax_over_fisco : hard cutoff as multiple of f_ISCO (default 1.3)
    sigma_taper_over_fisco : Fermi taper width / f_ISCO (default 0.01)
    phase_only : if True, use GR amplitude but full RG phase for the proxy
        amplitude. The default balance-law amplitude has no explicit
        lambda_RG-dependent magnitude after the SPA cancellation.
    amplitude_model : "balance", "proxy", or "newtonian" for the default
        h_of_f waveform. Ignored when waveform_func is supplied.
    waveform_func : callable with same signature as h_of_f (default: h_of_f)
    """
    if params is None:
        params = FiducialParams()

    if psd_func is None:
        psd_func = {
            "aLIGO": psd_aLIGO,
            "Voyager": psd_Voyager,
            "CE_20km": psd_CE_20km,
            "ET-D": psd_ET_D,
            "CE_40km": psd_CE_40km,
        }[detector_name]

    wf_kw = dict(fmax_over_fisco=fmax_over_fisco,
                 sigma_taper_over_fisco=sigma_taper_over_fisco,
                 phase_only=phase_only,
                 amplitude_model=amplitude_model,
                 waveform_func=waveform_func)

    # Frequency grid
    f = np.arange(f_low, f_high, df)

    # PSD
    Sn = psd_func(f)

    # Fiducial waveform + SNR
    theta = params.to_theta()
    h_fid = _h_from_theta(f, theta, f_low, **wf_kw)
    snr = compute_snr(f, h_fid, Sn)

    logger.info("%s: SNR = %.1f", detector_name, snr)

    # Derivatives
    derivs = compute_derivatives(f, theta, f_low, steps, **wf_kw)

    # Fisher matrix
    Gamma = compute_fisher_matrix(f, derivs, Sn)

    # Invert
    cov, cond = invert_fisher(Gamma)

    # Extract sigmas
    sigmas = {}
    for i, name in enumerate(PARAM_NAMES):
        diag = cov[i, i]
        sigmas[name] = np.sqrt(diag) if diag > 0 else np.inf

    result = FisherResult(
        param_names=PARAM_NAMES,
        fiducial=theta,
        fisher_matrix=Gamma,
        covariance=cov,
        condition_number=cond,
        snr=snr,
        sigmas=sigmas,
        detector_name=detector_name,
        f_low=f_low,
    )

    return result


def combine_fisher_results(
    results: list[FisherResult],
    detector_name: str,
) -> FisherResult:
    """Combine independent detector Fisher matrices into a reduced network result.

    This is a reduced network forecast: it sums Fisher information from
    constituent detectors assuming the same intrinsic waveform response in each
    detector, without sky localization, polarization, or antenna-pattern
    geometry.
    """
    if not results:
        raise ValueError("Need at least one Fisher result to combine")

    fiducial = results[0].fiducial.copy()
    for result in results[1:]:
        if not np.allclose(result.fiducial, fiducial):
            raise ValueError("All Fisher results must share the same fiducial parameters")

    fisher_matrix = np.sum([result.fisher_matrix for result in results], axis=0)
    covariance, condition_number = invert_fisher(fisher_matrix)
    snr = float(np.sqrt(np.sum([result.snr ** 2 for result in results])))

    sigmas = {}
    for i, name in enumerate(PARAM_NAMES):
        diag = covariance[i, i]
        sigmas[name] = np.sqrt(diag) if diag > 0 else np.inf

    return FisherResult(
        param_names=PARAM_NAMES,
        fiducial=fiducial,
        fisher_matrix=fisher_matrix,
        covariance=covariance,
        condition_number=condition_number,
        snr=snr,
        sigmas=sigmas,
        detector_name=detector_name,
        f_low=min(result.f_low for result in results),
    )


def run_fisher_network(
    detector_configs: list[tuple[str, Callable, float]],
    params: FiducialParams | None = None,
    detector_name: str = "network",
    f_high: float = 2048.0,
    df: float = 1.0 / 8.0,
    steps: dict[str, float] | None = None,
    fmax_over_fisco: float = 1.3,
    sigma_taper_over_fisco: float = 0.01,
    phase_only: bool = False,
    amplitude_model: str = "balance",
    waveform_func: Callable | None = None,
) -> FisherResult:
    """Run the reduced network Fisher forecast by summing detector Fishers."""
    component_results = []
    for det_label, psd_func, f_low in detector_configs:
        component_results.append(
            run_fisher(
                params=params,
                psd_func=psd_func,
                detector_name=det_label,
                f_low=f_low,
                f_high=f_high,
                df=df,
                steps=steps,
                fmax_over_fisco=fmax_over_fisco,
                sigma_taper_over_fisco=sigma_taper_over_fisco,
                phase_only=phase_only,
                amplitude_model=amplitude_model,
                waveform_func=waveform_func,
            )
        )

    return combine_fisher_results(component_results, detector_name=detector_name)


def print_result(r: FisherResult) -> None:
    """Pretty-print a Fisher result."""
    print(f"\n{'─' * 60}")
    print(f"  Detector: {r.detector_name}   f_low = {r.f_low:.0f} Hz")
    print(f"  SNR = {r.snr:.1f}")
    print(f"  Fisher condition number = {r.condition_number:.2e}")
    print(f"{'─' * 60}")
    print(f"  {'Parameter':<14} {'Fiducial':>12} {'Sigma':>14}")
    print(f"  {'─' * 42}")

    fid = FiducialParams.from_theta(r.fiducial)

    print(f"  {'sigma(Mc)/Mc':<14} {'':>12} {r.sigma('ln_Mc'):>14.6e}")
    print(f"  {'eta':<14} {fid.eta:>12.4f} {r.sigma('eta'):>14.6e}")
    print(f"  {'tc [s]':<14} {fid.tc:>12.4f} {r.sigma('tc'):>14.6e}")
    print(f"  {'phic [rad]':<14} {fid.phic:>12.4f} {r.sigma('phic'):>14.6e}")
    print(f"  {'sigma(dL)/dL':<14} {'':>12} {r.sigma('ln_dL'):>14.6e}")
    print(f"  {'lambda_RG':<14} {fid.lambda_RG:>12.4f} {r.sigma('lambda_RG'):>14.6e}")
    print(f"{'─' * 60}")


# ═══════════════════════════════════════════════════════════════════════
#  Validation script
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json
    import time
    from pathlib import Path

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    outdir = Path("results/rg_tail_forecast/validation")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RG Tail Fisher Forecast Validation")
    print("GW150914-like: Mc=28.3, eta=0.247, dL=410 Mpc, lambda_RG=1 (GR)")
    print("=" * 70)

    params = FiducialParams(Mc=28.3, eta=0.247, dL=410.0, lambda_RG=1.0)
    M_sec = params.Mc * MSUN_SEC / params.eta**(3.0 / 5.0)
    f_isco = 1.0 / (6.0**1.5 * PI * M_sec)
    print(f"\nM_total = {params.Mc / params.eta**(3.0/5.0):.1f} Msun")
    print(f"f_ISCO  = {f_isco:.1f} Hz")

    # ── Run for three detectors ──
    configs = [
        ("aLIGO",   psd_aLIGO,  20.0),
        ("ET-D",    psd_ET_D,    5.0),
        ("CE_40km", psd_CE_40km, 5.0),
    ]

    results = []
    for det_name, psd_func, f_low in configs:
        t0 = time.time()
        r = run_fisher(params=params, psd_func=psd_func,
                       detector_name=det_name, f_low=f_low)
        dt = time.time() - t0
        print_result(r)
        print(f"  Runtime: {dt:.1f} s")
        results.append(r)

    # ── Sanity check: quantify the waveform-level SNR shift from lambda_RG ──
    print("\n" + "=" * 70)
    print("Sanity check: SNR at lambda_RG=0 vs lambda_RG=1")
    print("=" * 70)
    for det_name, psd_func, f_low in configs:
        f_arr = np.arange(f_low, 2048.0, 0.125)
        Sn = psd_func(f_arr)

        h1 = h_of_f(f_arr, Mc=28.3, eta=0.247, dL=410.0,
                     lambda_RG=1.0, f_low=f_low)
        h0 = h_of_f(f_arr, Mc=28.3, eta=0.247, dL=410.0,
                     lambda_RG=0.0, f_low=f_low)
        snr1 = compute_snr(f_arr, h1, Sn)
        snr0 = compute_snr(f_arr, h0, Sn)
        pct = abs(snr1 - snr0) / snr1 * 100
        print(f"  {det_name}: SNR(lRG=1) = {snr1:.1f}, "
              f"SNR(lRG=0) = {snr0:.1f}, diff = {pct:.2f}%")

    # ── gamma_univ_22 at ISCO ──
    print("\n" + "=" * 70)
    print("gamma_univ_22 at ISCO frequencies")
    print("=" * 70)
    from src.rg_tail.waveform import (
        energy_real_over_M, _compute_khat_J, gamma_univ_22,
    )
    x_isco = np.array([1.0 / 6.0])
    khat_isco, J_isco = _compute_khat_J(x_isco, nu=0.247, m=2)
    gamma_isco = gamma_univ_22(khat_isco, J_isco)[0]
    print(f"  x_ISCO = {x_isco[0]:.4f}")
    print(f"  khat_ISCO = {khat_isco[0]:.6f}")
    print(f"  gamma_univ_22 = {gamma_isco:.6f}")
    print(f"  |gamma_univ_22| is O({abs(gamma_isco):.2e}), "
          f"{'OK (0.01-0.1)' if 0.005 < abs(gamma_isco) < 0.2 else 'UNEXPECTED'}")

    # ── Summary table ──
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Detector':<10} {'SNR':>8} {'sigma(Mc)/Mc':>14} {'sigma(eta)':>12} "
          f"{'sigma(dL)/dL':>14} {'sigma(lRG)':>12} {'cond':>10}")
    print("-" * 82)
    summary_data = []
    for r in results:
        row = {
            "detector": r.detector_name,
            "f_low": r.f_low,
            "SNR": r.snr,
            "sigma_Mc_over_Mc": r.sigma("ln_Mc"),
            "sigma_eta": r.sigma("eta"),
            "sigma_dL_over_dL": r.sigma("ln_dL"),
            "sigma_lambda_RG": r.sigma("lambda_RG"),
            "condition_number": r.condition_number,
        }
        summary_data.append(row)
        print(f"{r.detector_name:<10} {r.snr:>8.1f} {r.sigma('ln_Mc'):>14.6e} "
              f"{r.sigma('eta'):>12.6e} {r.sigma('ln_dL'):>14.6e} "
              f"{r.sigma('lambda_RG'):>12.6e} {r.condition_number:>10.2e}")

    # ── Save results ──
    with open(outdir / "fisher_validation.json", "w") as fp:
        json.dump(summary_data, fp, indent=2)
    print(f"\nSaved: {outdir / 'fisher_validation.json'}")
