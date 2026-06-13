"""Dispatch parametric vs symbolic approaches for the Analytic Bench."""
import amodels as M
import asymbolic as S


def run(spec, write=True, verbose=True):
    if spec["kind"] == "symbolic":
        return S.run_symbolic(spec, write=write, verbose=verbose)
    return M.run_parametric(spec, write=write, verbose=verbose)
