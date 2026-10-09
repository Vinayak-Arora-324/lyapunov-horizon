"""Simple models that any learned emulator has to beat.

Every model maps a batch of states u_n (shape (B, N)) to u_{n+1}, one
snapshot interval later. `rollout` then feeds predictions back in,
autoregressively — exactly how an LLM generates text, token by token.
"""

from __future__ import annotations

import itertools

import numpy as np

from .ks import KS


class Model:
    name = "model"

    def fit(self, train: np.ndarray) -> "Model":
        return self

    def step(self, u: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class Persistence(Model):
    """Tomorrow looks like today: u_{n+1} = u_n."""
    name = "Persistence"

    def step(self, u):
        return u.copy()


class Climatology(Model):
    """Always predict the long-run mean field. Its error defines 'no skill'."""
    name = "Climatology"

    def fit(self, train):
        self.mean = train.mean(axis=(0, 1))
        return self

    def step(self, u):
        return np.broadcast_to(self.mean, u.shape).copy()


class LinearDMD(Model):
    """Best linear one-step map, u_{n+1} = A u_n + b, by ridge regression.

    This is (exact) dynamic mode decomposition with an affine term. A linear
    model can only produce growing, decaying or oscillating modes, so it
    cannot sustain chaos — a useful lower bar.
    """
    name = "Linear (DMD)"

    def __init__(self, ridge: float = 1e-6):
        self.ridge = ridge

    def fit(self, train):
        X = train[:, :-1].reshape(-1, train.shape[-1]).astype(np.float64)
        Y = train[:, 1:].reshape(-1, train.shape[-1]).astype(np.float64)
        F = np.hstack([X, np.ones((len(X), 1))])
        self.W = _ridge(F, Y - X, self.ridge)
        return self

    def step(self, u):
        return u + np.hstack([u, np.ones((len(u), 1))]) @ self.W


class LocalPolynomial(Model):
    """A learned local rule: u_i(t+dt) = u_i + f(u_{i-r}, ..., u_{i+r}).

    f is a quadratic polynomial in the 2r+1 neighbouring values, shared across
    all positions (translation invariance, like a CNN or a cellular automaton).
    KS is itself local and quadratic (u u_x + derivatives), so this is a
    strong, physics-shaped baseline — fitted in closed form by least squares.

    `input_noise` (as a fraction of sigma_u) adds Gaussian noise to the inputs
    while keeping clean targets. Without it, the rule is accurate on the
    attractor but amplifies small-scale errors it never saw in training, and
    every rollout blows up within a few steps. With it, the rule learns to
    pull perturbed states back towards the attractor.
    """

    def __init__(self, radius: int = 5, input_noise: float = 0.0,
                 ridge: float = 1e-8, max_rows: int = 400_000, seed: int = 0):
        self.r, self.ridge, self.max_rows, self.seed = radius, ridge, max_rows, seed
        self.input_noise = input_noise
        w = 2 * radius + 1
        self.pairs = list(itertools.combinations_with_replacement(range(w), 2))
        self.name = (f"Local quadratic (r={radius}" +
                     (f", noise {input_noise:g}σ)" if input_noise else ")"))

    def _features(self, u):
        """(B, N) -> (B, N, n_features): bias, window, pairwise products."""
        win = np.stack([np.roll(u, -s, axis=-1)
                        for s in range(-self.r, self.r + 1)], axis=-1)
        a, b = zip(*self.pairs)
        quad = win[..., list(a)] * win[..., list(b)]
        ones = np.ones(win.shape[:-1] + (1,))
        return np.concatenate([ones, win, quad], axis=-1)

    def fit(self, train):
        rng = np.random.default_rng(self.seed)
        n_traj, n_t, N = train.shape
        # Each snapshot pair gives N training rows (one per position); subsample.
        n_snap = min(n_traj * (n_t - 1), max(1, self.max_rows // N))
        idx = rng.choice(n_traj * (n_t - 1), size=n_snap, replace=False)
        ti, tt = np.divmod(idx, n_t - 1)
        X = train[ti, tt].astype(np.float64)
        Y = train[ti, tt + 1].astype(np.float64)
        if self.input_noise:
            noise = rng.normal(size=X.shape)
            noise -= noise.mean(axis=-1, keepdims=True)
            X = X + self.input_noise * train.std() * noise
        F = self._features(X).reshape(-1, 1 + 2 * self.r + 1 + len(self.pairs))
        self.w = _ridge(F, (Y - X).reshape(-1, 1), self.ridge)[:, 0]
        return self

    def step(self, u):
        return u + self._features(u) @ self.w


class PerfectModel(Model):
    """The true solver. Used to show the chaos ceiling: even the exact
    equations lose track of the trajectory if the initial state is slightly off."""

    def __init__(self, ks: KS, dt: float):
        self.ks, self.dt = ks, dt
        self.name = "True solver"

    def step(self, u):
        return self.ks.advance(u, self.dt)


def _ridge(F, Y, alpha):
    """Least squares with a ridge penalty scaled to the feature magnitudes."""
    G = F.T @ F
    G[np.diag_indices_from(G)] += alpha * np.trace(G) / len(G)
    return np.linalg.solve(G, F.T @ Y)


def rollout(model: Model, u0: np.ndarray, n_steps: int,
            blowup: float = 100.0) -> np.ndarray:
    """Autoregressive rollout: (B, N) -> (B, n_steps+1, N).

    Trajectories whose amplitude exceeds `blowup` are marked as diverged
    (NaN from then on) and no longer stepped.
    """
    B, N = u0.shape
    out = np.full((B, n_steps + 1, N), np.nan)
    out[:, 0] = u0
    u = u0.astype(np.float64)
    alive = np.ones(B, dtype=bool)
    for n in range(1, n_steps + 1):
        if not alive.any():
            break
        with np.errstate(all="ignore"):
            nxt = model.step(u[alive])
        ok = np.isfinite(nxt).all(axis=-1) & (np.abs(nxt).max(axis=-1) < blowup)
        idx = np.flatnonzero(alive)
        out[idx[ok], n] = nxt[ok]
        u[idx[ok]] = nxt[ok]
        alive[idx[~ok]] = False
    return out
