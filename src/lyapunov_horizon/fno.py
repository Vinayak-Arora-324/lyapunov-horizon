"""A small 1D Fourier Neural Operator (FNO) emulator, in PyTorch.

The network learns the one-step map u_n -> u_{n+1}. Each layer mixes
information globally through a learned filter on the lowest Fourier modes
(a convolution over the whole periodic domain, done in Fourier space) and
locally through a pointwise linear map, followed by a GELU. Both operations
commute with shifts of the grid, so the network is translation-equivariant
by construction, like the KS equation itself.

Two physics-motivated choices:

- It predicts the increment u_{n+1} - u_n, scaled to unit variance (the
  increment is only ~5% of sigma_u per snapshot), rather than the next state.
- The predicted increment has zero spatial mean: KS conserves the mean of u,
  so the emulator does too, exactly.

`FNOModel` wraps a trained network in the numpy `Model` interface, so it is
rolled out and scored by exactly the same code as the baselines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .baselines import Model


class SpectralConv1d(nn.Module):
    """Multiply the lowest `modes` Fourier coefficients by a learned complex
    (channels x channels) matrix per mode; discard the rest."""

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / channels
        self.weight = nn.Parameter(
            scale * torch.randn(channels, channels, modes, dtype=torch.cfloat))

    def forward(self, h):  # (B, C, N) -> (B, C, N)
        hf = torch.fft.rfft(h)
        out = torch.zeros_like(hf)
        m = min(self.modes, hf.shape[-1])
        out[..., :m] = torch.einsum("bik,iok->bok", hf[..., :m], self.weight[..., :m])
        return torch.fft.irfft(out, n=h.shape[-1])


class FNO1d(nn.Module):
    """u / sigma (B, N) -> scaled increment (B, N), with zero spatial mean.

    lift (1 -> width) -> `layers` x [spectral conv + pointwise conv, GELU]
    -> project (width -> 2 width -> 1).
    """

    def __init__(self, width: int = 48, modes: int = 16, layers: int = 4):
        super().__init__()
        self.config = {"width": width, "modes": modes, "layers": layers}
        self.lift = nn.Conv1d(1, width, 1)
        self.spectral = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(layers))
        self.project = nn.Sequential(nn.Conv1d(width, 2 * width, 1), nn.GELU(),
                                     nn.Conv1d(2 * width, 1, 1))

    def forward(self, x):
        h = self.lift(x[:, None])
        for spec, pw in zip(self.spectral, self.pointwise):
            h = F.gelu(spec(h) + pw(h))
        dx = self.project(h)[:, 0]
        return dx - dx.mean(dim=-1, keepdim=True)


class Emulator(nn.Module):
    """Physical units in and out: u_n (B, N) -> u_{n+1} (B, N).

    sigma: std of u (input scale); delta_scale: std of u_{n+1} - u_n (output scale).
    """

    def __init__(self, net: FNO1d, sigma: float, delta_scale: float):
        super().__init__()
        self.net = net
        self.register_buffer("sigma", torch.tensor(float(sigma)))
        self.register_buffer("delta_scale", torch.tensor(float(delta_scale)))

    def increment(self, u):
        """Predicted (u_{n+1} - u_n) / delta_scale."""
        return self.net(u / self.sigma)

    def forward(self, u):
        return u + self.delta_scale * self.increment(u)

    def unroll(self, u, k: int):
        """K autoregressive steps with gradients: (B, N) -> (B, k, N)."""
        out = []
        for _ in range(k):
            u = self(u)
            out.append(u)
        return torch.stack(out, dim=1)


def reflect(u):
    """The KS symmetry u(x) -> -u(-x) on the periodic grid (index j -> -j mod N)."""
    if isinstance(u, np.ndarray):
        return -np.roll(u[..., ::-1], 1, axis=-1)
    return -torch.roll(torch.flip(u, dims=[-1]), 1, dims=-1)


def increment_scale(train: np.ndarray) -> float:
    """Std of the one-step increments u_{n+1} - u_n over a (n_traj, n_time, N) array."""
    return float(np.std(train[:, 1:] - train[:, :-1]))


class FNOModel(Model):
    """A trained `Emulator` behind the numpy `Model.step` interface."""

    def __init__(self, emulator: Emulator, name: str = "FNO"):
        self.emulator = emulator.eval()
        self.name = name

    @torch.no_grad()
    def step(self, u):
        x = torch.from_numpy(np.ascontiguousarray(u, dtype=np.float32))
        return self.emulator(x).numpy().astype(np.float64)


def save_checkpoint(path: str | Path, emulator: Emulator, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": emulator.net.config, "state_dict": emulator.state_dict(),
                **meta}, path)


def load_checkpoint(path: str | Path) -> tuple[Emulator, dict]:
    """Returns the emulator and the extra metadata saved with it."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = FNO1d(**ckpt.pop("config"))
    emulator = Emulator(net, sigma=1.0, delta_scale=1.0)
    emulator.load_state_dict(ckpt.pop("state_dict"))
    return emulator.eval(), ckpt
