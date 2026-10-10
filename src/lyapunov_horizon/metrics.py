"""How to judge an emulator of a chaotic system.

Two questions, deliberately kept apart:

1. Weather — how long does a rollout track the true trajectory?
   Normalised RMSE over time, and the valid prediction time (VPT): the first
   time the error crosses a threshold, measured in Lyapunov times.

2. Climate — once it has inevitably diverged, does it still behave like the
   real system? Compared on the long-run energy spectrum, the distribution of
   u, and the temporal autocorrelation. All three are needed: a model that
   freezes in place (persistence) has the right spectrum and distribution but
   the wrong dynamics, and only the autocorrelation catches it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance


# ---------------------------------------------------------------- weather
def nrmse(pred: np.ndarray, true: np.ndarray, sigma: float) -> np.ndarray:
    """Spatial RMSE normalised by the attractor's std: (B, T, N) -> (B, T).

    0 is perfect, 1 is what predicting the mean gives, ~1.41 is the error
    between two unrelated states on the attractor. Diverged rollouts -> inf.
    """
    e = np.sqrt(np.mean((pred - true) ** 2, axis=-1)) / sigma
    return np.where(np.isfinite(e), e, np.inf)


def valid_prediction_time(err: np.ndarray, dt: float, lambda_1: float,
                          threshold: float = 0.5) -> np.ndarray:
    """First time each rollout's NRMSE exceeds `threshold`, in Lyapunov times."""
    above = err > threshold
    first = np.where(above.any(axis=1), above.argmax(axis=1), err.shape[1])
    return first * dt * lambda_1


# ---------------------------------------------------------------- climate
def energy_spectrum(u: np.ndarray) -> np.ndarray:
    """Time- and ensemble-averaged |u_k|^2 for each rfft wavenumber index."""
    u = u.reshape(-1, u.shape[-1])
    u = u[np.isfinite(u).all(axis=-1)]
    if len(u) == 0:
        return np.full(u.shape[-1] // 2 + 1, np.nan)
    return np.mean(np.abs(np.fft.rfft(u, axis=-1) / u.shape[-1]) ** 2, axis=0)


def autocorrelation(u: np.ndarray, max_lag: int) -> np.ndarray:
    """Temporal autocorrelation of u at a fixed point, averaged over points and
    trajectories: (B, T, N) -> (max_lag+1,)."""
    u = u - np.nanmean(u)
    var = np.nanmean(u * u)
    return np.array([np.nanmean(u[:, lag:] * u[:, :u.shape[1] - lag]) / var
                     for lag in range(max_lag + 1)])


def climate_scores(pred: np.ndarray, true: np.ndarray, sigma: float,
                   max_lag: int, k_max: int | None = None,
                   n_values: int = 200_000, seed: int = 0,
                   drop_diverged: bool = False) -> dict:
    """Compare long-run statistics of a free-running model with the truth.

    pred, true: (B, T, N) long trajectories (a burn-in already discarded).
    k_max defaults to N // 4, the protocol's range (the higher modes carry
    almost no energy, so their log-error would dominate).
    Returns:
      spectrum_err  mean |log10 E_pred(k) - log10 E_true(k)| over 1 <= k <= k_max
      pdf_w1        Wasserstein-1 distance between distributions of u, / sigma
      acf_err       RMS difference of the autocorrelation up to max_lag
      diverged      fraction of rollouts that blew up

    If any rollout diverged, the three statistics are NaN: scoring only the
    survivors would let a model that usually blows up look as good as one that
    never does. drop_diverged=True scores the surviving rollouts instead, for
    diagnostics only (e.g. during training); never use it for reported numbers.
    """
    ok = np.isfinite(pred).all(axis=(1, 2))
    diverged = float(np.mean(~ok))
    if not ok.any() or (not ok.all() and not drop_diverged):
        return {"spectrum_err": np.nan, "pdf_w1": np.nan, "acf_err": np.nan,
                "diverged": diverged}
    pred = pred[ok]

    Ep, Et = energy_spectrum(pred), energy_spectrum(true)
    k_max = true.shape[-1] // 4 if k_max is None else k_max
    with np.errstate(divide="ignore"):
        spec = np.mean(np.abs(np.log10(Ep[1:k_max + 1]) - np.log10(Et[1:k_max + 1])))

    rng = np.random.default_rng(seed)
    vp = rng.choice(pred.ravel(), size=min(n_values, pred.size), replace=False)
    vt = rng.choice(true.ravel(), size=min(n_values, true.size), replace=False)
    w1 = wasserstein_distance(vp, vt) / sigma

    acf = np.sqrt(np.mean((autocorrelation(pred, max_lag)
                           - autocorrelation(true, max_lag)) ** 2))
    return {"spectrum_err": float(spec), "pdf_w1": float(w1), "acf_err": float(acf),
            "diverged": diverged}
