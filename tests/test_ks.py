import numpy as np
import pytest

from lyapunov_horizon.ks import KS
from lyapunov_horizon.lyapunov import largest_lyapunov


@pytest.fixture(scope="module")
def attractor_states():
    ks = KS()
    rng = np.random.default_rng(0)
    return ks.advance(ks.random_ic(4, rng), 300.0)


def test_linear_modes_match_exact_solution():
    """A tiny single-mode perturbation evolves as exp((k^2 - k^4) t)."""
    ks = KS()
    q = 2 * np.pi * 3 / ks.L                       # mode 3: k^2 - k^4 < 0, decays
    u0 = 1e-8 * np.cos(q * ks.x)
    u = ks.advance(u0, 5.0)
    expected = u0 * np.exp((q**2 - q**4) * 5.0)
    assert np.allclose(u, expected, rtol=1e-6, atol=1e-20)


def test_mean_is_conserved(attractor_states):
    u = KS().advance(attractor_states, 50.0)
    assert np.abs(u.mean(axis=-1)).max() < 1e-12


def test_fourth_order_convergence(attractor_states):
    ref = KS(h=0.005).advance(attractor_states, 10.0)
    err = [np.abs(KS(h=h).advance(attractor_states, 10.0) - ref).max()
           for h in (0.1, 0.05)]
    order = np.log2(err[0] / err[1])
    assert 3.3 < order < 4.7, order


def test_attractor_statistics(attractor_states):
    """L=22 dynamics stay bounded with O(1) amplitude (std ~ 1.2)."""
    U = KS().integrate(attractor_states, 400, 0.25)
    assert 0.9 < U.std() < 1.5
    assert np.abs(U).max() < 5


def test_integrate_shapes():
    ks = KS()
    u0 = np.zeros(ks.N)
    assert ks.integrate(u0, 3, 0.25).shape == (4, ks.N)
    assert ks.integrate(np.zeros((2, ks.N)), 3, 0.25).shape == (2, 4, ks.N)
    with pytest.raises(ValueError):
        ks.integrate(u0, 3, 0.123)


@pytest.mark.slow
def test_lyapunov_exponent_matches_literature():
    """Cvitanovic, Davidchack & Siminos (2010) report lambda_1 = 0.048 for L=22."""
    r = largest_lyapunov(KS(), n_runs=4, t_total=2000.0)
    assert abs(r["lambda_1"] - 0.048) < 0.005
