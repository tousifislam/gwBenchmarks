import numpy as np
from scipy.special import loggamma


MSUN_SEC = 4.925491025543576e-6
MPC_SEC = 3.0856775814913673e22 / 299792458.0

_GAMMA_E = np.euler_gamma
_PHI0 = np.exp(17.0 / 12.0 - _GAMMA_E) / 4.0
_LOG_120 = np.log(120.0)


def _suffix_trapezoid(y, x):
    """Return I(x_i) = int_{x_i}^{x_max} y(x) dx for sorted ascending x."""
    out = np.zeros_like(x, dtype=float)
    if x.size < 2:
        return out
    seg = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    out[:-1] = np.cumsum(seg[::-1])[::-1]
    return out


def _conservative_sector(x, nu):
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    log16x = np.log(16.0 * x)

    c1 = -3.0 / 4.0 - nu / 12.0
    c2 = -27.0 / 8.0 + 19.0 * nu / 8.0 - nu * nu / 24.0
    c3 = (
        -675.0 / 64.0
        + (34445.0 / 576.0 - 205.0 * np.pi * np.pi / 96.0) * nu
        - 155.0 * nu * nu / 96.0
        - 35.0 * nu**3 / 5184.0
    )
    c4_const = (
        -3969.0 / 128.0
        + (-123671.0 / 5760.0 + 9037.0 * np.pi * np.pi / 1536.0 + 896.0 * _GAMMA_E / 15.0) * nu
        + (498449.0 / 3456.0 - 3157.0 * np.pi * np.pi / 576.0) * nu * nu
        + 301.0 * nu**3 / 1728.0
        + 77.0 * nu**4 / 31104.0
    )
    c4_log = (448.0 / 15.0) * nu
    c4 = c4_const + c4_log * log16x

    poly_c = 1.0 + c1 * x + c2 * x2 + c3 * x3 + c4 * x4
    E = 1.0 - 0.5 * nu * x * poly_c
    dE_dx = -(nu / 2.0) * (
        1.0 + 2.0 * c1 * x + 3.0 * c2 * x2 + 4.0 * c3 * x3 + x4 * (5.0 * c4 + c4_log)
    )

    d1 = 3.0 / 2.0 + nu / 6.0
    d2 = 27.0 / 8.0 - 19.0 * nu / 8.0 + nu * nu / 24.0
    d3 = (
        135.0 / 16.0
        + (-6889.0 / 144.0 + 41.0 * np.pi * np.pi / 24.0) * nu
        + 31.0 * nu * nu / 24.0
        + 7.0 * nu**3 / 1296.0
    )
    d4_const = (
        2835.0 / 128.0
        + (98869.0 / 5760.0 - 128.0 * _GAMMA_E / 3.0 - 6455.0 * np.pi * np.pi / 1536.0) * nu
        + (356035.0 / 3456.0 - 2255.0 * np.pi * np.pi / 576.0) * nu * nu
        - 215.0 * nu**3 / 1728.0
        - 55.0 * nu**4 / 31104.0
    )
    d4_log = -(64.0 / 3.0) * nu
    d4 = d4_const + d4_log * log16x

    poly_d = 1.0 + d1 * x + d2 * x2 + d3 * x3 + d4 * x4
    p_phi = x ** (-0.5) * poly_d

    H_eff = (E * E - 1.0) / (2.0 * nu) + 1.0
    return E, dE_dx, p_phi, H_eff


def _hhat22(x, nu, lambda_RG):
    E, dE_dx, p_phi, H_eff = _conservative_sector(x, nu)

    Omega_hat = x ** 1.5
    khat = 2.0 * E * Omega_hat
    J = p_phi / (E * E)

    gamma_univ = (
        -(214.0 / 105.0) * khat * khat
        + (4.0 / 3.0) * J * khat**3
        - (3390466.0 / 1157625.0) * khat**4
        + (763726.0 / 99225.0) * J * khat**5
    )
    ellhat = 2.0 + lambda_RG * gamma_univ

    logT = (
        _LOG_120
        + (ellhat - 2.0) * np.log(4.0 * np.sqrt(x))
        + 2.0j * khat * np.log(4.0 * _PHI0)
        + loggamma(ellhat - 1.0 - 2.0j * khat)
        - loggamma(2.0 * ellhat + 2.0)
        + np.pi * khat
        - 0.5j * np.pi * (ellhat - 2.0)
    )
    T22 = np.exp(logT)

    eulerlog2 = _GAMMA_E + np.log(4.0 * np.sqrt(x))
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x

    r1 = -43.0 / 42.0 + 55.0 * nu / 84.0
    r2 = -20555.0 / 10584.0 - 33025.0 * nu / 21168.0 + 19583.0 * nu * nu / 42336.0
    r3 = (
        -4296031.0 / 4889808.0
        + (41.0 * np.pi * np.pi / 192.0 - 48993925.0 / 9779616.0) * nu
        - 6292061.0 * nu * nu / 3259872.0
        + 10620745.0 * nu**3 / 39118464.0
    )
    r4 = (
        9228174993589.0 / 800950550400.0
        + (-2487107795131.0 / 145627372800.0 + 464.0 * eulerlog2 / 35.0 - 9953.0 * np.pi * np.pi / 21504.0) * nu
        + (10815863492353.0 / 640760440320.0 - 3485.0 * np.pi * np.pi / 5376.0) * nu * nu
        - 2088847783.0 * nu**3 / 11650189824.0
        + 70134663541.0 * nu**4 / 512608352256.0
    )
    rho = 1.0 + r1 * x + r2 * x2 + r3 * x3 + r4 * x4

    y = (E * x ** 1.5) ** (2.0 / 3.0)
    delta = (
        -17.0 * y ** 1.5 / 3.0
        - 24.0 * nu * y ** 2.5
        + (30995.0 * nu / 1134.0 + 962.0 * nu * nu / 135.0) * y ** 3.5
        - 4976.0 * np.pi * nu * y**4 / 105.0
    )

    hhat = H_eff * T22 * (rho * rho) * np.exp(1j * delta)
    return hhat, E, dE_dx


def h_of_f(
    f,
    Mc,
    eta,
    dL,
    tc=0.0,
    phic=0.0,
    lambda_RG=1.0,
    f_low=20.0,
    fmax_over_fisco=1.3,
    sigma_taper_over_fisco=0.01,
    phase_only=False,
):
    f = np.asarray(f, dtype=float)
    h = np.zeros(f.shape, dtype=np.complex128)

    if f.size == 0:
        return h
    if eta <= 0.0:
        return h

    nu = float(eta)
    M_sec = float(Mc) * MSUN_SEC / (nu ** (3.0 / 5.0))
    dL_sec = float(dL) * MPC_SEC
    if M_sec <= 0.0 or dL_sec <= 0.0:
        return h

    f_isco = 1.0 / (np.pi * (6.0 ** 1.5) * M_sec)
    f_cut = float(fmax_over_fisco) * f_isco
    sigma = float(sigma_taper_over_fisco) * f_isco

    if f_cut <= 0.0:
        return h

    mask = (f >= float(f_low)) & (f < f_cut) & (f > 0.0)
    if not np.any(mask):
        return h

    fv = f[mask]
    order = np.argsort(fv)
    fs = fv[order]

    x = (np.pi * M_sec * fs) ** (2.0 / 3.0)
    hhat, E, dE_dx = _hhat22(x, nu, float(lambda_RG))

    flux = (32.0 / 5.0) * nu * nu * x**5 * np.abs(hhat) ** 2
    flux = np.maximum(flux, np.finfo(float).tiny)

    dt_dx = -M_sec * dE_dx / flux
    dt_df = dt_dx * (2.0 / 3.0) * (x / fs)
    dt_df = np.maximum(dt_df, np.finfo(float).tiny)

    tau_to_high = _suffix_trapezoid(dt_df, fs)
    t_of_f = float(tc) - tau_to_high

    dphi_df = 2.0 * np.pi * fs * dt_df
    dphi_to_high = _suffix_trapezoid(dphi_df, fs)
    phi_gw = float(phic) - dphi_to_high

    psi_spa = 2.0 * np.pi * fs * t_of_f - phi_gw - np.pi / 4.0

    Mc_sec = float(Mc) * MSUN_SEC
    amp0 = np.sqrt(5.0 / 24.0) * (np.pi ** (-2.0 / 3.0)) * (Mc_sec ** (5.0 / 6.0)) / dL_sec
    amp = amp0 * fs ** (-7.0 / 6.0)

    # Smooth taper near f_cut, while enforcing exact hard cutoff through the mask.
    if sigma > 0.0:
        taper = 1.0 / (1.0 + np.exp((fs - f_cut) / sigma))
    else:
        taper = np.ones_like(fs)

    hs = amp * taper * np.exp(1j * psi_spa) * hhat

    if phase_only:
        hs = np.exp(1j * np.angle(hs))

    h_valid = np.empty_like(fv, dtype=np.complex128)
    h_valid[order] = hs
    h[mask] = h_valid
    return h
