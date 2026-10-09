"""The shared weather/climate evaluation protocol.

Every model, baseline or neural, is judged by these two functions, so their
numbers are directly comparable (see metrics.py for what each score means).

Weather: rollouts start every 2 Lyapunov times along each trajectory of a split
and run for 10 Lyapunov times. Reported: one-step NRMSE and the valid
prediction time (VPT) distribution; the NRMSE curves are kept for plotting.

Climate: the model free-runs from the first state of each trajectory for the
full trajectory length; the first 12 Lyapunov times are discarded and the rest
is compared with the true trajectories (spectrum, value distribution,
autocorrelation up to a 2-Lyapunov-time lag, divergence).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .baselines import Model, rollout
from .data import Dataset
from .metrics import climate_scores, nrmse, valid_prediction_time

# Protocol constants, in Lyapunov times.
WEATHER_SPACING = 2.0
WEATHER_HORIZON = 10.0
CLIMATE_BURN_IN = 12.0
CLIMATE_MAX_LAG = 2.0
VPT_THRESHOLD = 0.5


def lyapunov_steps(ds: Dataset, n_lyapunov: float) -> int:
    """Number of snapshots in `n_lyapunov` Lyapunov times (rounded)."""
    return int(round(n_lyapunov * ds.steps_per_lyapunov_time))


def weather_windows(traj: np.ndarray, spacing: int, horizon: int):
    """Start states and true continuations: (n_traj*n_starts, N), (.., horizon+1, N)."""
    starts = np.arange(0, traj.shape[1] - horizon, spacing)
    u0 = traj[:, starts].reshape(-1, traj.shape[-1])
    truth = np.stack([traj[:, s:s + horizon + 1] for s in starts], axis=1)
    return u0.astype(np.float64), truth.reshape(-1, horizon + 1, traj.shape[-1])


def perturb(u0: np.ndarray, ic_error: float, sigma: float, seed: int = 0) -> np.ndarray:
    """Add zero-mean noise with spatial RMS exactly `ic_error * sigma` to each state."""
    noise = np.random.default_rng(seed).normal(size=u0.shape)
    noise -= noise.mean(axis=-1, keepdims=True)
    noise *= ic_error * sigma / np.sqrt(np.mean(noise**2, axis=-1, keepdims=True))
    return u0 + noise


@dataclass
class WeatherResult:
    metrics: dict          # one_step_nrmse, vpt_mean, vpt_median, vpt_q10, vpt_q90
    nrmse: np.ndarray      # (n_rollouts, horizon+1); inf after divergence
    vpt: np.ndarray        # (n_rollouts,) in Lyapunov times

    @property
    def diverged(self) -> float:
        """Fraction of rollouts that blew up before the horizon."""
        return float(np.mean(~np.isfinite(self.nrmse[:, -1])))


@dataclass
class ClimateResult:
    metrics: dict          # spectrum_err, pdf_w1, acf_err, diverged
    run: np.ndarray        # (n_traj, T, N) free run after burn-in; NaN after divergence
    truth: np.ndarray      # (n_traj, T, N) matching true trajectories
    max_lag: int           # autocorrelation lag used, in steps


def evaluate_weather(model: Model, ds: Dataset, split: str = "test", *,
                     spacing: int | None = None, horizon: int | None = None,
                     ic_error: float = 0.0, seed: int = 0,
                     threshold: float = VPT_THRESHOLD) -> WeatherResult:
    """Short-range forecast skill of `model` on a split.

    spacing, horizon: in steps; default to the protocol (2 and 10 Lyapunov times).
    ic_error > 0 perturbs the start states (fraction of sigma, see `perturb`),
    e.g. to measure the chaos-limited ceiling of the true solver.
    """
    traj = getattr(ds, split)
    spacing = lyapunov_steps(ds, WEATHER_SPACING) if spacing is None else spacing
    horizon = lyapunov_steps(ds, WEATHER_HORIZON) if horizon is None else horizon
    u0, truth = weather_windows(traj, spacing, horizon)
    if ic_error > 0:
        u0 = perturb(u0, ic_error, ds.sigma, seed)
    err = nrmse(rollout(model, u0, horizon), truth, ds.sigma)
    vpt = valid_prediction_time(err, ds.dt, ds.lambda_1, threshold)
    metrics = {
        "one_step_nrmse": float(np.median(err[:, 1])),
        "vpt_mean": float(vpt.mean()), "vpt_median": float(np.median(vpt)),
        "vpt_q10": float(np.quantile(vpt, 0.1)), "vpt_q90": float(np.quantile(vpt, 0.9)),
    }
    return WeatherResult(metrics, err, vpt)


def evaluate_climate(model: Model, ds: Dataset, split: str = "test", *,
                     burn_in: int | None = None, max_lag: int | None = None,
                     k_max: int | None = None) -> ClimateResult:
    """Long-run statistics of `model` free-running from each trajectory's first state.

    burn_in, max_lag: in steps; default to the protocol (12 and 2 Lyapunov times).
    k_max: highest wavenumber index in the spectrum error; defaults to N // 4.
    """
    traj = getattr(ds, split)
    burn = lyapunov_steps(ds, CLIMATE_BURN_IN) if burn_in is None else burn_in
    max_lag = lyapunov_steps(ds, CLIMATE_MAX_LAG) if max_lag is None else max_lag
    k_max = traj.shape[-1] // 4 if k_max is None else k_max
    truth = traj[:, burn:].astype(np.float64)
    run = rollout(model, traj[:, 0].astype(np.float64), traj.shape[1] - 1)[:, burn:]
    metrics = climate_scores(run, truth, ds.sigma, max_lag=max_lag, k_max=k_max)
    return ClimateResult(metrics, run, truth, max_lag)
