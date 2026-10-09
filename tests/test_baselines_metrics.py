import numpy as np

from lyapunov_horizon.baselines import LinearDMD, LocalPolynomial, Model, Persistence, rollout
from lyapunov_horizon.ks import KS
from lyapunov_horizon.metrics import autocorrelation, nrmse, valid_prediction_time


def small_dataset(n_traj=4, n_samples=200):
    ks = KS()
    rng = np.random.default_rng(0)
    u0 = ks.advance(ks.random_ic(n_traj, rng), 200.0)
    return ks.integrate(u0, n_samples, 0.25)


def test_rollout_marks_divergence():
    class Doubler(Model):
        def step(self, u):
            return 2 * u
    out = rollout(Doubler(), np.ones((2, 8)), 10, blowup=10.0)
    assert np.isfinite(out[:, 3]).all()        # 8 < 10
    assert np.isnan(out[:, 4:]).all()          # 16 > 10 -> diverged


def test_valid_prediction_time():
    err = np.array([[0.0, 0.1, 0.6, 0.2], [0.0, 0.1, 0.2, 0.3]])
    vpt = valid_prediction_time(err, dt=1.0, lambda_1=0.5, threshold=0.5)
    assert np.allclose(vpt, [1.0, 2.0])        # first crossing at step 2; none -> full length


def test_nrmse_is_zero_for_perfect_prediction_and_inf_for_nan():
    u = np.random.default_rng(1).normal(size=(3, 5, 8))
    assert np.allclose(nrmse(u, u, 1.0), 0)
    bad = u.copy()
    bad[0, 2] = np.nan
    assert np.isinf(nrmse(bad, u, 1.0)[0, 2])


def test_autocorrelation_starts_at_one():
    u = small_dataset()
    acf = autocorrelation(u, 20)
    assert np.isclose(acf[0], 1.0) and acf[20] < acf[0]


def test_linear_model_fits_and_local_rule_beats_persistence_one_step():
    data = small_dataset()
    train, test = data[:3], data[3:]
    u, nxt = test[0, :-1], test[0, 1:]
    err = {}
    for m in (Persistence(), LinearDMD(), LocalPolynomial(radius=5, max_rows=50_000)):
        m.fit(train)
        err[m.name] = np.sqrt(np.mean((m.step(u) - nxt) ** 2))
    assert err["Local quadratic (r=5)"] < 0.2 * err["Persistence"]
