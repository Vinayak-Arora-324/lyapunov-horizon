"""Largest Lyapunov exponent by the two-trajectory (Benettin) method.

A reference trajectory and a slightly perturbed copy are integrated together.
Every `tau` time units the separation d is measured, log(d / d0) is
accumulated, and the perturbed copy is pulled back to distance d0 along the
current separation direction. The average growth rate converges to the
largest Lyapunov exponent lambda_1; the Lyapunov time is T_lambda = 1/lambda_1.

T_lambda is the natural clock for this project: an infinitesimal error grows
by a factor e every T_lambda, so no model, however good, can track a single
trajectory for more than a handful of Lyapunov times.
"""

from __future__ import annotations

import numpy as np

from .ks import KS


def largest_lyapunov(ks: KS, n_runs: int = 8, burn_in: float = 500.0,
                     t_total: float = 4000.0, tau: float = 1.0,
                     d0: float = 1e-8, seed: int = 0) -> dict:
    """Estimate lambda_1, averaged over `n_runs` independent initial conditions.

    Returns a dict with the mean, the standard error across runs, and the
    per-run estimates.
    """
    rng = np.random.default_rng(seed)
    u = ks.advance(ks.random_ic(n_runs, rng), burn_in)       # onto the attractor

    delta = rng.normal(size=u.shape)
    delta -= delta.mean(axis=-1, keepdims=True)              # keep zero mean
    delta *= d0 / np.linalg.norm(delta, axis=-1, keepdims=True)
    w = u + delta

    n_steps = int(round(t_total / tau))
    log_growth = np.zeros(n_runs)
    for _ in range(n_steps):
        both = ks.advance(np.concatenate([u, w]), tau)
        u, w = both[:n_runs], both[n_runs:]
        sep = w - u
        d = np.linalg.norm(sep, axis=-1)
        log_growth += np.log(d / d0)
        w = u + sep * (d0 / d)[:, None]

    per_run = log_growth / (n_steps * tau)
    return {
        "lambda_1": float(per_run.mean()),
        "stderr": float(per_run.std(ddof=1) / np.sqrt(n_runs)),
        "per_run": per_run.tolist(),
        "t_total": t_total,
        "tau": tau,
    }
