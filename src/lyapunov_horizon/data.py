"""Generate and load KS trajectory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ks import KS


@dataclass
class Dataset:
    train: np.ndarray        # (n_traj, n_time, N)
    val: np.ndarray
    test: np.ndarray
    x: np.ndarray
    dt: float                # time between snapshots
    L: float
    lambda_1: float          # largest Lyapunov exponent

    @property
    def lyapunov_time(self) -> float:
        return 1.0 / self.lambda_1

    @property
    def steps_per_lyapunov_time(self) -> float:
        return self.lyapunov_time / self.dt

    @property
    def sigma(self) -> float:
        """Pointwise standard deviation of u over the training set."""
        return float(self.train.std())


def generate_split(ks: KS, n_traj: int, n_samples: int, dt: float,
                   burn_in: float, seed: int) -> np.ndarray:
    """n_traj independent trajectories on the attractor, n_samples+1 snapshots each."""
    rng = np.random.default_rng(seed)
    u0 = ks.advance(ks.random_ic(n_traj, rng), burn_in)
    return ks.integrate(u0, n_samples, dt).astype(np.float32)


def save(path: str | Path, ds: Dataset, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, train=ds.train, val=ds.val, test=ds.test, x=ds.x,
                        dt=ds.dt, L=ds.L, lambda_1=ds.lambda_1, **meta)


def load(path: str | Path) -> Dataset:
    with np.load(path) as f:
        return Dataset(train=f["train"], val=f["val"], test=f["test"], x=f["x"],
                       dt=float(f["dt"]), L=float(f["L"]),
                       lambda_1=float(f["lambda_1"]))
