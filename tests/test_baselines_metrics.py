import numpy as np

from lyapunov_horizon.baselines import LinearDMD, LocalPolynomial, Model, Persistence, rollout
from lyapunov_horizon.ks import KS
from lyapunov_horizon.metrics import (autocorrelation, climate_scores, energy_spectrum, nrmse,
                                      valid_prediction_time)


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


def test_climate_scores_do_not_score_survivors_by_default():
    """One blown-up run out of four must not leave the others to be scored alone."""
    true = small_dataset()
    pred = true.copy()
    pred[0, 50:] = np.nan
    strict = climate_scores(pred, true, sigma=1.2, max_lag=20)
    assert strict["diverged"] == 0.25
    for k in ("spectrum_err", "pdf_w1", "acf_err"):
        assert np.isnan(strict[k])
    loose = climate_scores(pred, true, sigma=1.2, max_lag=20, drop_diverged=True)
    assert loose["diverged"] == 0.25
    assert np.isfinite([loose["spectrum_err"], loose["pdf_w1"], loose["acf_err"]]).all()


def test_climate_scores_default_k_max_is_quarter_grid():
    true = small_dataset()
    pred = true.copy()
    pred[..., :] += 0.01 * np.cos(2 * np.pi * 20 * np.arange(64) / 64)  # mode 20 > N/4
    assert np.isclose(energy_spectrum(pred)[1:17], energy_spectrum(true)[1:17]).all()
    default = climate_scores(pred, true, sigma=1.2, max_lag=20)["spectrum_err"]
    assert default == climate_scores(pred, true, sigma=1.2, max_lag=20, k_max=16)["spectrum_err"]
    assert default < climate_scores(pred, true, sigma=1.2, max_lag=20, k_max=32)["spectrum_err"]
