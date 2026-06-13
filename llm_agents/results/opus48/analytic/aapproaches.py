"""Analytic closed-form approach registry (opus48, original work).

Categories:
  physics            : PN-inspired integrable-frequency closed forms
  functional         : Gaussian / Lorentzian amplitude ansatze
  matched_asymptotic : composite inspiral+merger+ringdown (2-term phase / blends)
  symbolic           : PySR (primary) + gplearn closed-form amplitude discovery
Mass variables: q, eta, delta_m, sqrt_eta.
"""

def _p(n, name, cat, amp, phase, mv, deg=3, notes=""):
    return dict(number=n, name=name, category=cat, kind="parametric",
                amp_form=amp, phase_form=phase, mass_var=mv, deg=deg, notes=notes)


def _sym(n, name, backend, mv, phase="integrable_t3", deg=3, notes=""):
    return dict(number=n, name=name, category="symbolic", kind="symbolic",
                backend=backend, mass_var=mv, phase_form=phase, deg=deg, notes=notes)


SPECS = [
    # physics (integrable-frequency PN-inspired)
    _p(1, "pn_t3_sech_eta", "physics", "powerlaw_sech", "integrable_t3", "eta",
       notes="TaylorT3-integrable phase + power-law/sech amplitude (baseline)."),
    _p(2, "pn_t3_sech_q", "physics", "powerlaw_sech", "integrable_t3", "q",
       notes="Same form, raw q parameterisation."),
    _p(3, "pn_t3_sech_delta", "physics", "powerlaw_sech", "integrable_t3", "delta_m",
       notes="Mass-difference parameterisation (third reparam)."),
    _p(4, "pn_t3_2term_eta", "physics", "powerlaw_sech", "integrable_t3_2term", "eta",
       notes="Two-PN-power integrable phase: captures late-inspiral drift."),
    _p(5, "pn_t3_sech_sqrteta", "physics", "powerlaw_sech", "integrable_t3", "sqrt_eta",
       notes="sqrt(eta) reparam."),
    _p(6, "pn_t3_sech_eta_deg4", "physics", "powerlaw_sech", "integrable_t3", "eta", deg=4,
       notes="Higher-degree eta polynomial for coefficients."),
    # functional-form amplitude
    _p(7, "gauss_amp_t3_eta", "functional", "two_gaussian", "integrable_t3", "eta",
       notes="Two-Gaussian amplitude (asymmetric merger peak)."),
    _p(8, "lorentz_amp_t3_eta", "functional", "lorentzian", "integrable_t3", "eta",
       notes="Lorentzian amplitude peak."),
    _p(9, "gauss_amp_t3_q", "functional", "two_gaussian", "integrable_t3", "q",
       notes="Two-Gaussian amplitude, q reparam."),
    _p(10, "lorentz_amp_2term_eta", "functional", "lorentzian", "integrable_t3_2term", "eta",
       notes="Lorentzian amplitude + 2-term phase."),
    _p(11, "gauss_amp_delta", "functional", "two_gaussian", "integrable_t3", "delta_m",
       notes="Two-Gaussian, mass-difference reparam."),
    # matched-asymptotic / composite (2-term phase = inspiral+merger matched)
    _p(12, "composite_2term_eta", "matched_asymptotic", "powerlaw_sech", "integrable_t3_2term", "eta",
       notes="Composite inspiral(PN)+merger(tanh)+ringdown(sech) matched, 2-term."),
    _p(13, "composite_2term_q", "matched_asymptotic", "powerlaw_sech", "integrable_t3_2term", "q",
       notes="Composite, q reparam."),
    _p(14, "composite_2term_delta", "matched_asymptotic", "powerlaw_sech", "integrable_t3_2term", "delta_m",
       notes="Composite, mass-difference reparam."),
    _p(15, "composite_lorentz_2term", "matched_asymptotic", "lorentzian", "integrable_t3_2term", "eta",
       notes="Composite with Lorentzian amplitude."),
    _p(16, "composite_2term_deg4", "matched_asymptotic", "powerlaw_sech", "integrable_t3_2term", "eta", 4,
       notes="Composite, degree-4 eta polynomials (reasoned: capture curvature)."),
    _p(17, "composite_sqrteta", "matched_asymptotic", "powerlaw_sech", "integrable_t3_2term", "sqrt_eta",
       notes="Composite, sqrt(eta) reparam."),
    # symbolic (PySR primary, gplearn)
    _sym(18, "pysr_amp_eta", "pysr", "eta", notes="PySR closed-form log-amplitude, eta."),
    _sym(19, "pysr_amp_delta", "pysr", "delta_m", notes="PySR closed-form log-amplitude, delta_m (2nd reparam)."),
    _sym(20, "gplearn_amp_eta", "gplearn", "eta", notes="gplearn closed-form log-amplitude."),
    _sym(21, "pysr_amp_q_2term", "pysr", "q", "integrable_t3_2term",
         notes="PySR amplitude + 2-term phase, q reparam."),
    _sym(22, "gplearn_amp_delta", "gplearn", "delta_m", notes="gplearn log-amplitude, mass-difference."),
]

_BY = {s["number"]: s for s in SPECS}


def get_spec(n):
    return _BY[n]
