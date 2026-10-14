import numpy as np
import pytest

torch = pytest.importorskip("torch")

from lyapunov_horizon.baselines import rollout
from lyapunov_horizon.fno import (FNO1d, Emulator, FNOModel, increment_scale,
                                  load_checkpoint, reflect, save_checkpoint)
from lyapunov_horizon.ks import KS

N = 64


def make_emulator(seed=0, sigma=1.2, delta=0.06):
    torch.manual_seed(seed)
    return Emulator(FNO1d(width=8, modes=6, layers=2), sigma, delta)


def test_shapes_and_zero_mean_increment():
    emu = make_emulator()
    u = torch.randn(5, N)
    out = emu(u)
    assert out.shape == (5, N)
    # KS conserves the spatial mean, and so does the emulator.
    assert torch.allclose(out.mean(-1), u.mean(-1), atol=1e-6)


def test_translation_equivariance():
    emu = make_emulator()
    u = torch.randn(3, N)
    shift = 11
    a = emu(torch.roll(u, shift, dims=-1))
    b = torch.roll(emu(u), shift, dims=-1)
    assert torch.allclose(a, b, atol=1e-5)


def test_reflection_is_a_ks_symmetry():
    ks = KS()
    rng = np.random.default_rng(0)
    u = ks.advance(ks.random_ic(2, rng), 50.0)
    assert np.allclose(ks.advance(reflect(u), 0.25), reflect(ks.advance(u, 0.25)), atol=1e-12)
    assert np.allclose(reflect(reflect(u)), u)
    assert torch.allclose(reflect(torch.from_numpy(u)), torch.from_numpy(reflect(u)))


def test_unroll_matches_repeated_steps():
    emu = make_emulator()
    u = torch.randn(4, N)
    seq = emu.unroll(u, 3)
    assert torch.allclose(seq[:, 2], emu(emu(emu(u))))


def test_model_wrapper_works_with_rollout():
    m = FNOModel(make_emulator())
    u0 = np.random.default_rng(0).normal(size=(3, N))
    out = m.step(u0)
    assert out.dtype == np.float64 and out.shape == (3, N)
    traj = rollout(m, u0, 5)
    assert traj.shape == (3, 6, N) and np.isfinite(traj).all()


def test_checkpoint_round_trip(tmp_path):
    emu = make_emulator(sigma=1.3, delta=0.07)
    save_checkpoint(tmp_path / "m.pt", emu, selected="x")
    loaded, meta = load_checkpoint(tmp_path / "m.pt")
    assert meta["selected"] == "x"
    assert float(loaded.sigma) == pytest.approx(1.3)
    u = torch.randn(2, N)
    assert torch.equal(loaded(u), emu.eval()(u))


def test_increment_scale():
    t = np.cumsum(np.ones((2, 5, 3)), axis=1)  # increments all 1
    t[:, :, 0] += np.arange(5) * 2              # first column increments 3
    assert increment_scale(t) == pytest.approx(np.std([1, 1, 3] * 8))


def test_training_learns_a_linear_map():
    """A few hundred Adam steps on u_{n+1} = u_n + 0.05 * roll(u_n) - mean
    should beat predicting zero increment by a wide margin."""
    torch.manual_seed(0)
    u = torch.randn(512, N)
    target_inc = 0.05 * (torch.roll(u, 1, -1) - u)
    emu = Emulator(FNO1d(width=16, modes=16, layers=2), 1.0, 0.05)
    opt = torch.optim.Adam(emu.parameters(), lr=3e-3)
    for _ in range(300):
        loss = ((emu(u) - u - target_inc) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05 * (target_inc ** 2).mean().item()
