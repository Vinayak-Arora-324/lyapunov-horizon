"""Batched spectral solver for the 1D Kuramoto–Sivashinsky equation.

    u_t + u u_x + u_xx + u_xxxx = 0,    x in [0, L), periodic.

Time stepping is ETDRK4 (exponential time differencing, 4th-order Runge–Kutta)
following Kassam & Trefethen, "Fourth-order time-stepping for stiff PDEs",
SIAM J. Sci. Comput. 26 (2005). The stiff linear part (u_xx + u_xxxx) is
integrated exactly in Fourier space; only the nonlinear term is approximated.

Everything is vectorised over a leading batch axis, so many trajectories are
integrated at once — that is what makes data generation fast on a CPU.
"""

from __future__ import annotations

import numpy as np


class KS:
    """Kuramoto–Sivashinsky solver on a periodic domain.

    Parameters
    ----------
    L : domain length. L = 22 is the standard chaotic benchmark.
    N : number of grid points (even).
    h : internal ETDRK4 time step.
    dealias : apply the 2/3 rule to the nonlinear term. Off by default;
        at the resolutions used here the aliased modes are damped by
        exp(-h k^4) to well below round-off anyway.
    """

    def __init__(self, L: float = 22.0, N: int = 64, h: float = 0.05,
                 dealias: bool = False, n_contour: int = 32):
        if N % 2:
            raise ValueError("N must be even")
        self.L, self.N, self.h = float(L), int(N), float(h)

        k = 2 * np.pi / L * np.arange(N // 2 + 1)          # rfft wavenumbers
        self.k = k
        lin = k**2 - k**4                                   # linear operator
        self.lin = lin

        # Derivative factor for the nonlinear term  -1/2 d/dx (u^2).
        # The Nyquist mode of an odd derivative is set to zero (standard practice).
        kd = k.copy()
        kd[-1] = 0.0
        self.g = -0.5j * kd
        if dealias:
            self.g[k > (2 / 3) * k.max()] = 0.0

        # ETDRK4 coefficients via contour integrals (avoids cancellation error
        # in the phi-functions when h*lin is small).
        self.E = np.exp(h * lin)
        self.E2 = np.exp(h * lin / 2)
        r = np.exp(1j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
        LR = h * lin[:, None] + r[None, :]
        self.Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
        self.f1 = h * np.real(np.mean(
            (-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=1))
        self.f2 = h * np.real(np.mean(
            (2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
        self.f3 = h * np.real(np.mean(
            (-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))

    # ------------------------------------------------------------------ grid
    @property
    def x(self) -> np.ndarray:
        return np.arange(self.N) * self.L / self.N

    @property
    def dx(self) -> float:
        return self.L / self.N

    # -------------------------------------------------------------- stepping
    def _nonlin(self, v: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(v, n=self.N, axis=-1)
        return self.g * np.fft.rfft(u * u, axis=-1)

    def step_spectral(self, v: np.ndarray) -> np.ndarray:
        """One ETDRK4 step of size h, in Fourier space."""
        Nv = self._nonlin(v)
        a = self.E2 * v + self.Q * Nv
        Na = self._nonlin(a)
        b = self.E2 * v + self.Q * Na
        Nb = self._nonlin(b)
        c = self.E2 * a + self.Q * (2 * Nb - Nv)
        Nc = self._nonlin(c)
        return (self.E * v + Nv * self.f1 + 2 * (Na + Nb) * self.f2
                + Nc * self.f3)

    def advance(self, u: np.ndarray, t: float) -> np.ndarray:
        """Advance physical-space state(s) u by time t (a multiple of h)."""
        n = self._n_substeps(t)
        v = np.fft.rfft(np.asarray(u, dtype=np.float64), axis=-1)
        for _ in range(n):
            v = self.step_spectral(v)
        return np.fft.irfft(v, n=self.N, axis=-1)

    def integrate(self, u0: np.ndarray, n_samples: int, dt: float) -> np.ndarray:
        """Integrate and return n_samples+1 snapshots spaced dt apart.

        u0 has shape (N,) or (batch, N); the output has shape
        (n_samples+1, N) or (batch, n_samples+1, N). Index 0 is u0.
        """
        u0 = np.asarray(u0, dtype=np.float64)
        squeeze = u0.ndim == 1
        if squeeze:
            u0 = u0[None]
        m = self._n_substeps(dt)
        out = np.empty((u0.shape[0], n_samples + 1, self.N))
        out[:, 0] = u0
        v = np.fft.rfft(u0, axis=-1)
        for i in range(1, n_samples + 1):
            for _ in range(m):
                v = self.step_spectral(v)
            out[:, i] = np.fft.irfft(v, n=self.N, axis=-1)
        return out[0] if squeeze else out

    def _n_substeps(self, t: float) -> int:
        n = int(round(t / self.h))
        if n < 0 or not np.isclose(n * self.h, t, rtol=0, atol=1e-9):
            raise ValueError(f"time {t} is not a non-negative multiple of h={self.h}")
        return n

    # --------------------------------------------------------- initial data
    def random_ic(self, batch: int, rng: np.random.Generator,
                  n_modes: int = 8, amplitude: float = 0.1) -> np.ndarray:
        """Small random zero-mean initial conditions built from low Fourier modes."""
        x = self.x
        u = np.zeros((batch, self.N))
        for m in range(1, n_modes + 1):
            a, b = rng.normal(size=(2, batch, 1))
            q = 2 * np.pi * m / self.L
            u += a * np.cos(q * x) + b * np.sin(q * x)
        return amplitude * u / np.sqrt(n_modes)
