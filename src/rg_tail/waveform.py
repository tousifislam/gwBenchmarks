"""RG-tail waveform ingredients and simple frequency-domain wrappers.

The clean physics target in this module is the Section IV comparable-mass
factorized correction from arXiv:2602.08833,

    hhat_22 = Shat_eff^(0) * T_22 * rho_22^2 * exp(i * delta_22),

with:
  - the effective source Shat_eff^(0),
  - the radiative-coordinate external tail factor T_22,
  - the residual amplitude rho_22 from Eq. (157),
  - the residual phase delta_22 from Eq. (158).

At 4PN, Section IV explicitly drops the internal-tail bracket in Eq. (131)
because it starts at 5PN, and it also drops the denominator in Eq. (135)
because it enters at 6PN. This module follows that choice for the comparable-
mass Section IV implementation.

The module also retains a test-mass branch used by the local validation
scripts. That branch is not the main comparable-mass model, but it is useful
for reproducing the paper's self-convergence checks.

The beyond-GR parameter ``lambda_RG`` is a repository-specific deformation
that scales the running angular momentum contribution. In this code base,
general relativity corresponds to ``lambda_RG = 1``.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import loggamma

# Physical constants in geometric units.
MSUN_SEC = 4.925491025543576e-06
MPC_SEC = 1.0292712503e14
GAMMA_E = 0.5772156649015329
PI = np.pi

_TEST_MASS_TOL = 1e-14
_MAX_LOG_REAL = 500.0
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PSD_DATA_DIR = _PROJECT_ROOT / "data" / "psds"


def _asarray(x: np.ndarray | float) -> np.ndarray:
    """Return x as a floating NumPy array."""
    return np.asarray(x, dtype=float)


def _poly_in_x(x: np.ndarray, coeffs: list[float]) -> np.ndarray:
    """Evaluate sum_n coeffs[n] * x^n."""
    result = np.zeros_like(x, dtype=float)
    x_power = np.ones_like(x, dtype=float)
    for coeff in coeffs:
        result += coeff * x_power
        x_power *= x
    return result


def _eulerlog2(x: np.ndarray) -> np.ndarray:
    """Eulerlog for m = 2: gamma_E + log(4 sqrt(x))."""
    return GAMMA_E + np.log(4.0 * np.sqrt(x))


def _isco_frequency(M_sec: float) -> float:
    """Schwarzschild ISCO GW frequency for the m = 2 mode."""
    return 1.0 / (6.0 ** 1.5 * PI * M_sec)


def _newtonian_spa_amplitude(
    f: np.ndarray,
    Mc_sec: float,
    dL_sec: float,
) -> np.ndarray:
    """Restricted Newtonian SPA amplitude for the dominant mode."""
    return (
        np.sqrt(5.0 / 24.0)
        * Mc_sec ** (5.0 / 6.0)
        / (dL_sec * PI ** (2.0 / 3.0))
        * f ** (-7.0 / 6.0)
    )


def _balance_law_spa_amplitude_correction(
    x: np.ndarray,
    nu: float,
) -> np.ndarray:
    """SPA amplitude correction implied by the same balance law as the phase.

    For the 22-only construction used here,

        h_22 = h_22^N * hhat_22,
        F_22 = F_N * |hhat_22|^2.

    The stationary-phase amplitude scales as the time-domain amplitude divided
    by sqrt(df/dt). Therefore the factor |hhat_22| in the time-domain
    amplitude cancels the sqrt(|hhat_22|^2) entering df/dt through the flux.
    Relative to the Newtonian restricted amplitude, the remaining correction is

        sqrt((dE/dx) / (dE_N/dx)) = sqrt(-2 * d(E/M)/dx / nu).
    """
    dE_dx = d_energy_real_over_M_dx(x, nu)
    correction_sq = -2.0 * dE_dx / nu
    return np.sqrt(np.maximum(correction_sq, 0.0))


def _fermi_taper(f: np.ndarray, f_isco: float, sigma_over_fisco: float) -> np.ndarray:
    """Smooth taper centered at f_ISCO."""
    sigma = sigma_over_fisco * f_isco
    exponent = np.clip((f - f_isco) / sigma, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(exponent))


def _waveform_band_mask(
    f: np.ndarray,
    f_low: float,
    f_cut: float,
) -> np.ndarray:
    """Return the band on which the waveform is evaluated."""
    return (f >= f_low) & (f < f_cut)


@lru_cache(maxsize=None)
def _load_psd_table(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a tabulated PSD file from data/psds."""
    path = _PSD_DATA_DIR / filename
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def _interp_tabulated_psd(
    f: np.ndarray | float,
    filename: str,
    f_low: float,
) -> np.ndarray:
    """Interpolate a tabulated PSD in log space."""
    f = _asarray(f)
    f_tab, psd_tab = _load_psd_table(filename)

    Sn = np.full_like(f, np.inf, dtype=float)
    valid = (
        (f >= max(f_low, f_tab[0]))
        & (f <= f_tab[-1])
        & np.isfinite(f)
        & (f > 0.0)
    )
    if np.any(valid):
        log_psd = np.log(psd_tab)
        Sn[valid] = np.exp(np.interp(f[valid], f_tab, log_psd))
    return Sn


# ============================================================================
# Test-mass quantities (Sec. III)
# ============================================================================

def E_circ_test_mass(x: np.ndarray | float) -> np.ndarray:
    """Exact Schwarzschild circular-orbit specific energy (Eq. 85)."""
    x = _asarray(x)
    return (1.0 - 2.0 * x) / np.sqrt(1.0 - 3.0 * x)


def p_phi_circ_test_mass(x: np.ndarray | float) -> np.ndarray:
    """Exact Schwarzschild circular-orbit angular momentum (Eq. 86)."""
    x = _asarray(x)
    return 1.0 / np.sqrt(x * (1.0 - 3.0 * x))


def ellhat_test_mass(omegahat: np.ndarray | float, i_max_a: int = 8) -> np.ndarray:
    """Test-mass running angular momentum lhat_2 from Eq. (175)."""
    omegahat = _asarray(omegahat)

    coeffs = [
        -214.0 / 105.0,
        -3390466.0 / 1157625.0,
        -153440219802466.0 / 15021833990625.0,
        -71638806585865707261481.0 / 1520451676706008921875.0,
        -270360664939833821554899493653643.0
        / 1099244369724415858768042968750.0,
    ]

    result = np.full_like(omegahat, 2.0)
    oh2 = omegahat ** 2
    oh2_power = oh2.copy()
    for idx, coeff in enumerate(coeffs, start=1):
        if 2 * idx > i_max_a:
            break
        result += coeff * oh2_power
        oh2_power *= oh2
    return result


def rho_22_test_mass(x: np.ndarray | float) -> np.ndarray:
    """Test-mass residual amplitude rho_22 from Eq. (112), through x^10."""
    x = _asarray(x)

    coeffs = [
        1.0,
        -43.0 / 42.0,
        -20555.0 / 10584.0,
        -4296031.0 / 4889808.0,
        9228174993589.0 / 800950550400.0,
        -8938613036677.0 / 2116091577600.0,
        -1060700697798333909671.0 / 24231643979185843200.0,
        3567168919606240724303840051.0 / 43991338062012939037440000.0,
        8339316227220569285816625738049.0 / 279101750471556914871889920000.0,
        -522338057689474511990262498143822507399.0
        / 857097472947610731676894961786880000.0,
        1523513000214555169284583871085138536795675131.0
        / 1333729377653777059562416250036563968000000.0,
    ]
    return _poly_in_x(x, coeffs)


def delta_22_test_mass(x: np.ndarray | float) -> np.ndarray:
    """Test-mass residual phase delta_22 from Eq. (114)."""
    x = _asarray(x)
    return (
        -17.0 / 3.0 * x ** 1.5
        - 259.0 / 81.0 * x ** 4.5
        - 58940243.0 / 3539025.0 * x ** 7.5
    )


def ftilde_22_test_mass(x: np.ndarray | float) -> np.ndarray:
    """Test-mass internal-tail residual amplitude from Eq. (113), through x^10."""
    x = _asarray(x)

    coeffs = [
        1.0,
        4391.0 / 2247.0,
        53185.0 / 2646.0,
        17096210.0 / 305613.0,
        4747421406107252.0 / 71641272277575.0,
        8197825650198689.0 / 18747248820300.0,
        93413981315288045717.0 / 265033606022345160.0,
        -84886593520942215307406177173.0 / 115729970007349756213155000.0,
        12091990099120207716578842317287.0 / 2316762577156478297276430000.0,
        -18745458158059179828839098304527937.0
        / 5506639808719744112850116685000.0,
        -12907954629421965590241825710607690689624960837.0
        / 465999378807314866788638416951557033375000.0,
    ]
    return _poly_in_x(x, coeffs)


def deltatilde_22_test_mass(x: np.ndarray | float) -> np.ndarray:
    """Test-mass internal-tail residual phase from Eq. (115)."""
    x = _asarray(x)
    return (
        25.0 / 3.0 * x ** 1.5
        + 12077.0 / 567.0 * x ** 4.5
        + 159283133.0 / 694575.0 * x ** 7.5
    )


# ============================================================================
# Comparable-mass Section IV ingredients
# ============================================================================

def energy_real_over_M(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Comparable-mass E_real / M along circular orbits (Eq. 145)."""
    x = _asarray(x)
    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * PI ** 2 / 96.0) * nu
        - 155.0 * nu2 / 96.0
        - 35.0 * nu3 / 5184.0
    )
    c4 = (
        -3969.0 / 128.0
        + (
            -123671.0 / 5760.0
            + 9037.0 * PI ** 2 / 1536.0
            + 896.0 * GAMMA_E / 15.0
            + 448.0 * np.log(16.0 * x) / 15.0
        ) * nu
        + (498449.0 / 3456.0 - 3157.0 * PI ** 2 / 576.0) * nu2
        + 301.0 * nu3 / 1728.0
        + 77.0 * nu4 / 31104.0
    )

    bracket = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4
    return 1.0 - 0.5 * nu * x * bracket


def p_phi_circ_over_muM(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Comparable-mass p_phi^circ / (mu M) along circular orbits (Eq. 146)."""
    x = _asarray(x)
    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu

    c0 = 1.0
    c1 = 3.0 / 2.0 + nu / 6.0
    c2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu2 / 24.0
    c3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 * PI ** 2 / 24.0) * nu
        + 31.0 * nu2 / 24.0
        + 7.0 * nu3 / 1296.0
    )
    c4 = (
        2835.0 / 128.0
        + (
            98869.0 / 5760.0
            - 128.0 * GAMMA_E / 3.0
            - 6455.0 * PI ** 2 / 1536.0
            - 64.0 * np.log(16.0 * x) / 3.0
        ) * nu
        + (356035.0 / 3456.0 - 2255.0 * PI ** 2 / 576.0) * nu2
        - 215.0 * nu3 / 1728.0
        - 55.0 * nu4 / 31104.0
    )

    bracket = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4
    return bracket / np.sqrt(x)


def Hhat_eff(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Effective source helper for even-parity modes.

    For nu > 0, this is the Eq. (132) inversion of E_real.
    For the test-mass branch, Section III/IV use E_circ directly as the source,
    so this helper returns E_circ_test_mass for convenience.
    """
    x = _asarray(x)
    if nu < _TEST_MASS_TOL:
        return E_circ_test_mass(x)

    E_over_M = energy_real_over_M(x, nu)
    return (E_over_M ** 2 - 1.0) / (2.0 * nu) + 1.0


def gamma_univ_22(
    khat: np.ndarray | float,
    J: np.ndarray | float,
    m: int = 2,
) -> np.ndarray:
    """Universal anomalous dimension gamma^univ_{2m} (Eq. 137)."""
    khat = _asarray(khat)
    J = _asarray(J)

    return (
        -214.0 / 105.0 * khat ** 2
        + 2.0 * m * J * khat ** 3 / 3.0
        - 3390466.0 / 1157625.0 * khat ** 4
        + 381863.0 * m * J * khat ** 5 / 99225.0
    )


def _compute_khat_J(
    x: np.ndarray | float,
    nu: float,
    m: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute khat and J entering Eq. (137)."""
    x = _asarray(x)

    if nu < _TEST_MASS_TOL:
        return _compute_khat_J_test_mass(x, m=m)

    E_over_M = energy_real_over_M(x, nu)
    Omega = x ** 1.5
    khat = E_over_M * m * Omega
    p_phi = p_phi_circ_over_muM(x, nu)
    J = p_phi / E_over_M ** 2
    return khat, J


def _compute_khat_J_test_mass(
    x: np.ndarray | float,
    m: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Test-mass khat and J using exact Schwarzschild quantities."""
    x = _asarray(x)
    E_over_M = E_circ_test_mass(x)
    Omega = x ** 1.5
    khat = E_over_M * m * Omega
    p_phi = p_phi_circ_test_mass(x)
    J = p_phi / E_over_M ** 2
    return khat, J


def ellhathat_22(
    x: np.ndarray | float,
    nu: float,
    lambda_RG: float = 1.0,
    i_max_a: int = 8,
) -> np.ndarray:
    """Running angular momentum for the (2,2) mode.

    The repository-specific parameter lambda_RG scales the running part only:

        ellhathat_22 = 2 + lambda_RG * gamma_univ_22

    In the test-mass branch this becomes the same scaling applied to
    ellhat_test_mass - 2.
    """
    x = _asarray(x)

    if nu < _TEST_MASS_TOL:
        omegahat = E_circ_test_mass(x) * 2.0 * x ** 1.5
        return 2.0 + lambda_RG * (ellhat_test_mass(omegahat, i_max_a=i_max_a) - 2.0)

    khat, J = _compute_khat_J(x, nu, m=2)
    return 2.0 + lambda_RG * gamma_univ_22(khat, J, m=2)


def tail_factor_T22(
    x: np.ndarray | float,
    nu: float,
    lambda_RG: float = 1.0,
    i_max_a: int = 8,
) -> np.ndarray:
    """Radiative-coordinate external tail factor T_22 (Eq. 155)."""
    x = _asarray(x)

    ell = 2.0
    m = 2.0
    phi0 = np.exp(17.0 / 12.0 - GAMMA_E) / 4.0

    if nu < _TEST_MASS_TOL:
        E_over_M = E_circ_test_mass(x)
    else:
        E_over_M = energy_real_over_M(x, nu)

    Omega = x ** 1.5
    k = m * Omega
    khat = E_over_M * k
    lhat = ellhathat_22(x, nu, lambda_RG=lambda_RG, i_max_a=i_max_a)
    r_omega = 1.0 / x

    log_tail = (
        np.log(120.0)
        + (lhat - ell) * np.log(2.0 * k * r_omega)
        + 2j * khat * np.log(2.0 * m * phi0)
        + loggamma(lhat - 1.0 - 2j * khat)
        - loggamma(2.0 * lhat + 2.0)
        + PI * khat
        - 0.5j * PI * (lhat - ell)
    )

    log_tail = np.where(
        np.real(log_tail) > _MAX_LOG_REAL,
        _MAX_LOG_REAL + 1j * np.imag(log_tail),
        log_tail,
    )
    return np.exp(log_tail)


def lambda_NS_inst_22(x: np.ndarray | float) -> np.ndarray:
    """Leading approximation to lambda_NS^inst for the (2,2) mode."""
    x = _asarray(x)
    return 1.0 - 500.0 * x ** 3 / 49.0


def rho_22(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Comparable-mass residual amplitude rho_22 (Eq. 157)."""
    x = _asarray(x)
    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu
    eulerlog2 = _eulerlog2(x)

    c1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    c2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu2 / 42336.0
    c3 = (
        -4296031.0 / 4889808.0
        + (41.0 * PI ** 2 / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu2 / 3259872.0
        + 10620745.0 * nu3 / 39118464.0
    )
    c4 = (
        9228174993589.0 / 800950550400.0
        + (
            -2487107795131.0 / 145627372800.0
            + 464.0 * eulerlog2 / 35.0
            - 9953.0 * PI ** 2 / 21504.0
        ) * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * PI ** 2 / 5376.0) * nu2
        - 2088847783.0 * nu3 / 11650189824.0
        + 70134663541.0 * nu4 / 512608352256.0
    )

    return 1.0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4


def delta_22(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Comparable-mass residual phase delta_22 (Eq. 158)."""
    x = _asarray(x)
    E_over_M = energy_real_over_M(x, nu)
    y = (E_over_M * x ** 1.5) ** (2.0 / 3.0)
    nu2 = nu * nu

    return (
        -17.0 / 3.0 * y ** 1.5
        - 24.0 * nu * y ** 2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu2 / 135.0) * y ** 3.5
        - 4976.0 * PI * nu * y ** 4 / 105.0
    )


def Shat_eff_0(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Even-parity effective source for the (2,2) mode."""
    x = _asarray(x)
    return Hhat_eff(x, nu)


def hhat_22(
    x: np.ndarray | float,
    nu: float,
    lambda_RG: float = 1.0,
    i_max_a: int = 8,
    include_internal_tail: bool = False,
) -> np.ndarray:
    """Clean Section IV factorized correction for the (2,2) mode.

    For the current comparable-mass implementation we keep only the external
    tail factor, exactly as discussed below Eq. (143) in Section IV.
    """
    x = _asarray(x)

    if include_internal_tail:
        raise NotImplementedError(
            "Internal-tail factors are not implemented. "
            "The current clean Section IV model keeps only the external tail."
        )

    source = Shat_eff_0(x, nu)
    tail = tail_factor_T22(x, nu, lambda_RG=lambda_RG, i_max_a=i_max_a)

    if nu < _TEST_MASS_TOL:
        residual_amplitude = rho_22_test_mass(x)
        residual_phase = delta_22_test_mass(x)
    else:
        residual_amplitude = rho_22(x, nu)
        residual_phase = delta_22(x, nu)

    return source * tail * residual_amplitude ** 2 * np.exp(1j * residual_phase)


# ============================================================================
# Frequency-domain wrappers used by the Fisher scripts
# ============================================================================

def d_energy_real_over_M_dx(x: np.ndarray | float, nu: float) -> np.ndarray:
    """Derivative d(E_real / M) / dx along circular orbits."""
    x = _asarray(x)

    if nu < _TEST_MASS_TOL:
        denom = np.power(1.0 - 3.0 * x, 1.5)
        return (6.0 * x - 1.0) / (2.0 * denom)

    nu2 = nu * nu
    nu3 = nu2 * nu
    nu4 = nu3 * nu

    c0 = 1.0
    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu2 / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * PI ** 2 / 96.0) * nu
        - 155.0 * nu2 / 96.0
        - 35.0 * nu3 / 5184.0
    )
    c4_const = (
        -3969.0 / 128.0
        + (-123671.0 / 5760.0 + 9037.0 * PI ** 2 / 1536.0 + 896.0 * GAMMA_E / 15.0) * nu
        + (498449.0 / 3456.0 - 3157.0 * PI ** 2 / 576.0) * nu2
        + 301.0 * nu3 / 1728.0
        + 77.0 * nu4 / 31104.0
    )
    c4_log = 448.0 * nu / 15.0
    c4 = c4_const + c4_log * np.log(16.0 * x)

    bracket = c0 + c1 * x + c2 * x ** 2 + c3 * x ** 3 + c4 * x ** 4
    d_bracket_dx = (
        c1
        + 2.0 * c2 * x
        + 3.0 * c3 * x ** 2
        + (4.0 * c4 + c4_log) * x ** 3
    )
    return -0.5 * nu * (bracket + x * d_bracket_dx)


def flux_22(
    x: np.ndarray | float,
    nu: float,
    lambda_RG: float = 1.0,
) -> np.ndarray:
    """Dominant-mode GW flux from the factorized (2,2) waveform.

    Using h_22 = h_22^N * hhat_22 with the Newtonian circular-orbit
    normalization, the 22-only flux is

        F_22 = (32 / 5) * nu^2 * x^5 * |hhat_22|^2.
    """
    x = _asarray(x)
    return (32.0 / 5.0) * nu ** 2 * x ** 5 * np.abs(
        hhat_22(x, nu, lambda_RG=lambda_RG)
    ) ** 2


def _cumulative_integral_to_upper(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute I_i = integral_{x_i}^{x_max} y(x) dx on an ascending grid."""
    if len(x) == 0:
        return np.array([], dtype=float)
    if len(x) == 1:
        return np.zeros(1, dtype=float)

    dx = np.diff(x)
    trap = 0.5 * (y[:-1] + y[1:]) * dx
    cumulative_from_low = np.concatenate(([0.0], np.cumsum(trap)))
    return cumulative_from_low[-1] - cumulative_from_low


def _spa_phase_from_22_flux(
    f: np.ndarray,
    M_sec: float,
    nu: float,
    lambda_RG: float = 1.0,
    x_ref: float | None = None,
) -> np.ndarray:
    """Orbital SPA phase from the 22-only balance law.

    The phase is computed from

        dt/dx = -M * (d(E_real / M)/dx) / F_22(x),

    together with the circular-orbit relation Omega = x^(3/2) / M and the
    standard m = 2 SPA combination Psi = DeltaPhi_GW - 2 pi f Delta t.

    The reference point x_ref plays the role of an arbitrary coalescence
    anchor; changing it only adds a constant plus a term linear in f, which
    are absorbed by phic and tc in the waveform model.
    """
    f = _asarray(f)
    if len(f) == 0:
        return np.array([], dtype=float)

    x = (PI * M_sec * f) ** (2.0 / 3.0)
    if x_ref is None:
        x_ref = x[-1]

    x_ref = float(max(x_ref, x[-1]))
    append_ref = x_ref > x[-1] * (1.0 + 1e-12)
    if append_ref:
        x_nodes = np.concatenate((x, [x_ref]))
    else:
        x_nodes = x

    dE_dx = d_energy_real_over_M_dx(x_nodes, nu)
    F22 = flux_22(x_nodes, nu, lambda_RG=lambda_RG)
    F22 = np.maximum(F22, 1e-300)
    dt_dx = -M_sec * dE_dx / F22
    dphi_gw_dx = 2.0 * (x_nodes ** 1.5 / M_sec) * dt_dx

    time_to_ref = _cumulative_integral_to_upper(x_nodes, dt_dx)
    gw_phase_to_ref = _cumulative_integral_to_upper(x_nodes, dphi_gw_dx)

    if append_ref:
        time_to_ref = time_to_ref[:-1]
        gw_phase_to_ref = gw_phase_to_ref[:-1]

    return gw_phase_to_ref - 2.0 * PI * f * time_to_ref

def _taylorf2_phase_full(f: np.ndarray, Mc_sec: float, eta: float) -> np.ndarray:
    """Standard TaylorF2 phase including the usual tail terms."""
    M_sec = Mc_sec / eta ** (3.0 / 5.0)
    v = (PI * M_sec * f) ** (1.0 / 3.0)
    v2 = v * v
    eta2 = eta * eta

    psi0 = 1.0
    psi2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    psi3 = -16.0 * PI
    psi4 = 15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta2 / 72.0
    psi5_coeff = 38645.0 * PI / 756.0 - 65.0 * PI * eta / 9.0
    psi5 = psi5_coeff * (1.0 + 3.0 * np.log(v))
    psi6_const = (
        11583231236531.0 / 4694215680.0
        - 640.0 * PI ** 2 / 3.0
        + (-15737765635.0 / 3048192.0 + 2255.0 * PI ** 2 / 12.0) * eta
        + 76055.0 * eta2 / 1728.0
        - 127825.0 * eta2 * eta / 1296.0
    )
    psi6 = psi6_const - 6848.0 * np.log(v) / 21.0
    psi7 = (
        77096675.0 * PI / 254016.0
        + 378515.0 * PI * eta / 1512.0
        - 74045.0 * PI * eta2 / 756.0
    )

    return (
        3.0
        / (128.0 * eta * v ** 5)
        * (
            psi0
            + psi2 * v2
            + psi3 * v ** 3
            + psi4 * v2 * v2
            + psi5 * v ** 5
            + psi6 * v ** 6
            + psi7 * v ** 7
        )
    )


def h_of_f(
    f: np.ndarray,
    Mc: float,
    eta: float,
    dL: float,
    tc: float = 0.0,
    phic: float = 0.0,
    lambda_RG: float = 1.0,
    f_low: float = 20.0,
    fmax_over_fisco: float = 1.3,
    sigma_taper_over_fisco: float = 0.01,
    phase_only: bool = False,
    amplitude_model: str = "balance",
) -> np.ndarray:
    """Frequency-domain 22-only waveform using the balance-law SPA phase.

    The orbital phase is computed from the 22-only flux implied by the current
    factorized waveform model:

        F_22(x; lambda_RG) = (32/5) * nu^2 * x^5 * |hhat_22(x; lambda_RG)|^2.

    By default, ``amplitude_model="balance"`` uses the stationary-phase
    amplitude correction implied by the same balance law. The previous project
    proxy is still available as ``amplitude_model="proxy"``:

        balance: A_N(f) * sqrt(-2 d(E/M)/dx / nu),
        proxy:   A_N(f) * |hhat_22(x; lambda_RG)|,
        newtonian: A_N(f).

    For the balance-law amplitude, the explicit |hhat_22| in the time-domain
    22-mode amplitude cancels against the |hhat_22| entering df/dt through the
    22-only flux. The RG deformation still enters the phase through F_22 and
    through arg(hhat_22).
    """
    f = _asarray(f)
    amplitude_model = amplitude_model.lower()
    if amplitude_model not in {"balance", "proxy", "newtonian"}:
        raise ValueError(
            "amplitude_model must be one of 'balance', 'proxy', or 'newtonian'"
        )

    Mc_sec = Mc * MSUN_SEC
    M_sec = Mc_sec / eta ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC
    nu = eta

    f_isco = _isco_frequency(M_sec)
    f_cut = fmax_over_fisco * f_isco
    valid = _waveform_band_mask(f, f_low=f_low, f_cut=f_cut)
    h = np.zeros_like(f, dtype=complex)
    if not np.any(valid):
        return h

    f_eval = f[valid]
    x = (PI * M_sec * f_eval) ** (2.0 / 3.0)
    amp_newt = _newtonian_spa_amplitude(f_eval, Mc_sec=Mc_sec, dL_sec=dL_sec)
    hhat_running = hhat_22(x, nu, lambda_RG=lambda_RG)
    x_ref = (PI * M_sec * f_cut) ** (2.0 / 3.0)
    psi_orb = _spa_phase_from_22_flux(
        f_eval,
        M_sec=M_sec,
        nu=nu,
        lambda_RG=lambda_RG,
        x_ref=x_ref,
    )
    phase = 2.0 * PI * f_eval * tc - phic - PI / 4.0 + psi_orb + np.angle(hhat_running)

    if amplitude_model == "balance":
        amp = amp_newt * _balance_law_spa_amplitude_correction(x, nu)
    elif amplitude_model == "newtonian":
        amp = amp_newt
    elif phase_only:
        hhat_gr = hhat_22(x, nu, lambda_RG=1.0)
        amp = amp_newt * np.abs(hhat_gr)
    else:
        amp = amp_newt * np.abs(hhat_running)

    taper = _fermi_taper(f_eval, f_isco=f_isco, sigma_over_fisco=sigma_taper_over_fisco)
    h[valid] = amp * np.exp(1j * phase) * taper
    return h


def h_of_f_taylorf2_baseline(
    f: np.ndarray,
    Mc: float,
    eta: float,
    dL: float,
    tc: float = 0.0,
    phic: float = 0.0,
    lambda_RG: float = 1.0,
    f_low: float = 20.0,
    fmax_over_fisco: float = 1.3,
    sigma_taper_over_fisco: float = 0.01,
    phase_only: bool = False,
) -> np.ndarray:
    """TaylorF2 baseline with a multiplicative tail-factor ratio.

    This is a local robustness cross-check, not the primary Section IV model.
    """
    f = _asarray(f)
    Mc_sec = Mc * MSUN_SEC
    M_sec = Mc_sec / eta ** (3.0 / 5.0)
    dL_sec = dL * MPC_SEC
    nu = eta

    f_isco = _isco_frequency(M_sec)
    f_cut = fmax_over_fisco * f_isco
    valid = _waveform_band_mask(f, f_low=f_low, f_cut=f_cut)
    h = np.zeros_like(f, dtype=complex)
    if not np.any(valid):
        return h

    f_eval = f[valid]
    x = (PI * M_sec * f_eval) ** (2.0 / 3.0)
    amp_newt = _newtonian_spa_amplitude(f_eval, Mc_sec=Mc_sec, dL_sec=dL_sec)
    psi_orb = _taylorf2_phase_full(f_eval, Mc_sec, eta)

    T_running = tail_factor_T22(x, nu, lambda_RG=lambda_RG)
    T_no_running = tail_factor_T22(x, nu, lambda_RG=0.0)
    safe = np.abs(T_no_running) > 1e-30
    T_ratio = np.where(safe, T_running / T_no_running, 1.0 + 0.0j)

    if phase_only:
        amp = amp_newt
    else:
        amp = amp_newt * np.abs(T_ratio)

    phase = 2.0 * PI * f_eval * tc - phic - PI / 4.0 + psi_orb + np.angle(T_ratio)
    taper = _fermi_taper(f_eval, f_isco=f_isco, sigma_over_fisco=sigma_taper_over_fisco)
    h[valid] = amp * np.exp(1j * phase) * taper
    return h


# ============================================================================
# PSD helpers and convenience functions
# ============================================================================

def psd_aLIGO(f: np.ndarray | float) -> np.ndarray:
    """Analytic approximation to aLIGO design sensitivity."""
    f = _asarray(f)
    f0, S0 = 215.0, 1e-49
    x = f / f0
    Sn = S0 * (
        x ** (-4.14)
        - 5.0 * x ** (-2.0)
        + 111.0 * (1.0 - x ** 2 + 0.5 * x ** 4) / (1.0 + 0.5 * x ** 2)
    )
    return np.where((f < 10.0) | (Sn <= 0.0), np.inf, Sn)


def psd_Voyager(f: np.ndarray | float) -> np.ndarray:
    """Tabulated Voyager PSD from the local data directory."""
    return _interp_tabulated_psd(f, "Voyager_from_gwfast_psd.txt", f_low=5.0)


def psd_ET_D(f: np.ndarray | float) -> np.ndarray:
    """Analytic approximation to ET-D sensitivity."""
    f = _asarray(f)
    f0, S0 = 200.0, 1.449e-52
    x = f / f0
    Sn = S0 * (
        x ** (-4.05)
        + 185.62 * x ** (-0.69)
        + 232.56
        * (
            1.0
            + 31.18 * x
            - 64.72 * x ** 2
            + 52.24 * x ** 3
            - 42.16 * x ** 4
            + 10.17 * x ** 5
            + 11.53 * x ** 6
        )
    )
    return np.where((f < 1.0) | (Sn <= 0.0), np.inf, Sn)


def psd_CE_20km(f: np.ndarray | float) -> np.ndarray:
    """Tabulated CE 20 km PSD from the repo data directory."""
    return _interp_tabulated_psd(f, "CE1_20km_from_gwfast_psd.txt", f_low=5.0)


def psd_CE_40km(f: np.ndarray | float) -> np.ndarray:
    """Analytic approximation to CE 40 km sensitivity."""
    f = _asarray(f)
    f0, S0 = 200.0, 4.0e-54
    x = f / f0
    Sn = S0 * (x ** (-4.1) + 30.0 * x ** (-0.6) + 45.0 * (1.0 + 2.0 * x ** 2))
    return np.where((f < 3.0) | (Sn <= 0.0), np.inf, Sn)


def compute_snr(f: np.ndarray, h: np.ndarray, Sn: np.ndarray) -> float:
    """Optimal matched-filter SNR."""
    f = _asarray(f)
    df = f[1] - f[0]
    valid = np.isfinite(Sn) & (Sn > 0.0)
    integrand = np.where(valid, np.abs(h) ** 2 / Sn, 0.0)
    return float(np.sqrt(4.0 * np.sum(integrand) * df))


def generate_waveform(
    Mc: float = 28.3,
    eta: float = 0.247,
    dL: float = 410.0,
    tc: float = 0.0,
    phic: float = 0.0,
    lambda_RG: float = 1.0,
    f_low: float = 20.0,
    f_high: float = 2048.0,
    df: float = 0.125,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate h(f) on a uniform frequency grid."""
    f = np.arange(f_low, f_high, df)
    h = h_of_f(f, Mc=Mc, eta=eta, dL=dL, tc=tc, phic=phic, lambda_RG=lambda_RG, f_low=f_low)
    return f, h
