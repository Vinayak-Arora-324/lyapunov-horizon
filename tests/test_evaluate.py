from dataclasses import replace
from functools import lru_cache

import numpy as np

from lyapunov_horizon.baselines import LinearDMD, LocalPolynomial, Model, Persistence, rollout
from lyapunov_horizon.data import Dataset
from lyapunov_horizon.evaluate import (
    evaluate_climate,
    evaluate_weather,
    lyapunov_steps,
    perturb,
    weather_windows,
)
from lyapunov_horizon.ks import KS
from lyapunov_horizon.metrics import climate_scores, nrmse, valid_prediction_time

DT = 0.25
N = 8          # grid size of the synthetic datasets
LAMBDA = 0.4   # -> 10 snapshots per Lyapunov time


def synthetic_dataset(n_time=250, constant=False, seed=0):
    """Tiny dataset with 10 steps per Lyapunov time. constant=True makes every
    trajectory frozen in time, so Persistence is exact."""
    rng = np.random.default_rng(seed)

    def make(n):
        if constant:
            return np.repeat(rng.normal(size=(n, 1, N)), n_time, axis=1)
        return rng.normal(size=(n, n_time, N))

    return Dataset(train=make(3), val=make(2), test=make(4),
                   x=np.arange(N) * 22.0 / N, dt=DT, L=22.0, lambda_1=LAMBDA)


@lru_cache(maxsize=1)
def real_dataset():
    """Short real KS data: 20 snapshots per Lyapunov time, 301 snapshots each."""
    ks = KS()
    rng = np.random.default_rng(1)
    u0 = ks.advance(ks.random_ic(10, rng), 200.0)
    traj = ks.integrate(u0, 300, DT).astype(np.float32)
    return Dataset(train=traj[:4], val=traj[4:6], test=traj[6:10],
                   x=ks.x, dt=DT, L=22.0, lambda_1=0.2)


def reference_weather(m, ds, ic_error):
    """Copy of the original inline protocol from scripts/run_baselines.py."""
    spl = ds.steps_per_lyapunov_time
    horizon = int(round(10 * spl))
    u0, truth = weather_windows(ds.test, int(round(2 * spl)), horizon)
    rng = np.random.default_rng(0)
    noise_ic = rng.normal(size=u0.shape)
    noise_ic -= noise_ic.mean(axis=-1, keepdims=True)
    noise_ic *= ic_error * ds.sigma / np.sqrt(np.mean(noise_ic**2, axis=-1, keepdims=True))
    pred = rollout(m, u0 + noise_ic, horizon)
    err = nrmse(pred, truth, ds.sigma)
    vpt = valid_prediction_time(err, ds.dt, ds.lambda_1)
    return {"one_step_nrmse": float(np.median(err[:, 1])),
            "vpt_mean": float(vpt.mean()), "vpt_median": float(np.median(vpt)),
            "vpt_q10": float(np.quantile(vpt, 0.1)), "vpt_q90": float(np.quantile(vpt, 0.9))}


def reference_climate(m, ds):
    """Copy of the original inline climate protocol from scripts/run_baselines.py."""
    spl = ds.steps_per_lyapunov_time
    T = ds.test.shape[1] - 1
    burn = int(round(12 * spl))
    pred = rollout(m, ds.test[:, 0].astype(np.float64), T)[:, burn:]
    return climate_scores(pred, ds.test[:, burn:].astype(np.float64), ds.sigma,
                          max_lag=int(round(2 * spl)), k_max=ds.test.shape[-1] // 4)


def assert_same_metrics(a, b):
    assert a.keys() == b.keys()
    for k in a:
        if np.isnan(a[k]) and np.isnan(b[k]):
            continue
        assert a[k] == b[k], k


# ------------------------------------------------------------ weather_windows
def test_weather_windows_shapes_and_start_states():
    traj = np.arange(3 * 50 * N, dtype=float).reshape(3, 50, N)
    u0, truth = weather_windows(traj, spacing=5, horizon=10)
    n_starts = len(np.arange(0, 50 - 10, 5))            # 8
    assert u0.shape == (3 * n_starts, N)
    assert truth.shape == (3 * n_starts, 11, N)
    assert u0.dtype == np.float64
    assert np.array_equal(u0, truth[:, 0])


def test_weather_windows_spacing_and_no_overrun():
    traj = np.stack([np.full((50, N), 1000.0 * i) + np.arange(50)[:, None]
                     for i in range(3)])                  # value = 1000*traj + time
    u0, truth = weather_windows(traj, spacing=5, horizon=10)
    starts = u0[:, 0] % 1000
    assert np.array_equal(starts.reshape(3, -1), np.tile(np.arange(0, 40, 5), (3, 1)))
    # Each window is the true continuation of its start, and never runs past the end.
    assert np.array_equal(truth[:, :, 0] % 1000, starts[:, None] + np.arange(11))
    assert truth[:, -1, 0].max() % 1000 <= 49


# ------------------------------------------------------------------- perturb
def test_perturb_is_zero_mean_with_exact_rms():
    u0 = np.random.default_rng(2).normal(size=(5, N))
    ic_error, sigma = 0.03, 1.7
    d = perturb(u0, ic_error, sigma, seed=4) - u0
    assert np.allclose(d.mean(axis=-1), 0.0, atol=1e-12)
    assert np.allclose(np.sqrt(np.mean(d**2, axis=-1)), ic_error * sigma)


def test_perturb_deterministic_per_seed():
    u0 = np.zeros((4, N))
    a = perturb(u0, 0.1, 1.0, seed=3)
    assert np.array_equal(a, perturb(u0, 0.1, 1.0, seed=3))
    assert not np.allclose(a, perturb(u0, 0.1, 1.0, seed=4))


# ------------------------------------------------------- evaluate_weather basics
def test_exact_model_has_zero_error_on_frozen_data():
    ds = synthetic_dataset(constant=True)
    res = evaluate_weather(Persistence(), ds)
    assert np.all(res.nrmse == 0)
    assert res.diverged == 0.0
    assert res.metrics["one_step_nrmse"] == 0.0


def test_vpt_of_surviving_rollouts_is_censored_one_step_past_horizon():
    # Never crossing the threshold is reported as crossing at the first step
    # after the window, (horizon + 1) * dt, so it ranks above a crossing at the
    # last step (convention of valid_prediction_time; the published numbers use it).
    ds = synthetic_dataset(constant=True)
    res = evaluate_weather(Persistence(), ds)
    assert np.allclose(res.vpt, (lyapunov_steps(ds, 10.0) + 1) * DT * LAMBDA)


def test_metric_keys_and_types():
    ds = synthetic_dataset()
    m = evaluate_weather(Persistence(), ds).metrics
    assert set(m) == {"one_step_nrmse", "vpt_mean", "vpt_median", "vpt_q10", "vpt_q90"}
    assert all(type(v) is float for v in m.values())


def test_default_windows_follow_protocol():
    ds = synthetic_dataset(n_time=250)
    S, H = lyapunov_steps(ds, 2.0), lyapunov_steps(ds, 10.0)
    res = evaluate_weather(Persistence(), ds)
    n_starts = len(np.arange(0, 250 - H, S))
    assert res.nrmse.shape == (4 * n_starts, H + 1)
    assert res.vpt.shape == (4 * n_starts,)


def test_explicit_spacing_and_horizon_are_used():
    ds = synthetic_dataset(n_time=250)
    res = evaluate_weather(Persistence(), ds, spacing=7, horizon=13)
    n_starts = len(np.arange(0, 250 - 13, 7))
    assert res.nrmse.shape == (4 * n_starts, 14)


def test_split_argument_selects_split():
    ds = synthetic_dataset()
    a = evaluate_weather(Persistence(), ds, split="val").metrics
    b = evaluate_weather(Persistence(), replace(ds, test=ds.val)).metrics
    assert a == b
    assert a != evaluate_weather(Persistence(), ds, split="test").metrics


def test_diverging_model_is_flagged():
    class Affine(Model):
        name = "2u+1"

        def step(self, u):
            return 2 * u + 1

    ds = synthetic_dataset()
    res = evaluate_weather(Affine(), ds)
    assert res.diverged == 1.0
    assert np.isinf(res.nrmse[:, -1]).all()
    assert np.isfinite(res.vpt).all() and res.vpt.max() < 1.0


def test_ic_error_gives_one_step_error_equal_to_ic_error():
    ds = synthetic_dataset(constant=True)
    noisy = evaluate_weather(Persistence(), ds, ic_error=1e-2).metrics
    clean = evaluate_weather(Persistence(), ds, ic_error=0.0).metrics
    assert np.isclose(noisy["one_step_nrmse"], 1e-2, rtol=1e-6)
    assert np.isclose(clean["one_step_nrmse"], 0.0, atol=1e-12)


# ------------------------------------------------------- evaluate_climate basics
def test_climate_keys_shapes_and_default_windows():
    ds = synthetic_dataset(n_time=200)
    burn, lag = lyapunov_steps(ds, 12.0), lyapunov_steps(ds, 2.0)
    res = evaluate_climate(Persistence(), ds)
    assert set(res.metrics) == {"spectrum_err", "pdf_w1", "acf_err", "diverged"}
    assert res.run.shape == (4, 200 - burn, N)
    assert res.truth.shape == res.run.shape
    assert res.max_lag == lag == 20
    assert burn == 120


def test_climate_diverging_model_gives_nan():
    class Affine(Model):
        def step(self, u):
            return 2 * u + 1

    res = evaluate_climate(Affine(), synthetic_dataset(n_time=200))
    assert res.metrics["diverged"] == 1.0
    for k in ("spectrum_err", "pdf_w1", "acf_err"):
        assert np.isnan(res.metrics[k])
    assert np.isnan(res.run).all()


def test_climate_persistence_fails_autocorrelation_on_real_ks():
    ds = real_dataset()
    res = evaluate_climate(Persistence(), ds)
    assert res.metrics["diverged"] == 0.0
    assert res.metrics["acf_err"] > 0.2     # frozen run has acf = 1 at every lag


# ----------------------------------------------- equivalence with reference code
def test_weather_matches_reference_linear_dmd():
    ds = real_dataset()
    m = LinearDMD().fit(ds.train)
    assert_same_metrics(evaluate_weather(m, ds, ic_error=1e-3).metrics,
                        reference_weather(m, ds, 1e-3))


def test_weather_matches_reference_local_polynomial():
    ds = real_dataset()
    m = LocalPolynomial(radius=3, input_noise=0.01).fit(ds.train)
    assert_same_metrics(evaluate_weather(m, ds, ic_error=1e-3).metrics,
                        reference_weather(m, ds, 1e-3))


def test_climate_matches_reference_linear_dmd():
    ds = real_dataset()
    m = LinearDMD().fit(ds.train)
    assert_same_metrics(evaluate_climate(m, ds).metrics, reference_climate(m, ds))


def test_climate_matches_reference_local_polynomial():
    ds = real_dataset()
    m = LocalPolynomial(radius=3, input_noise=0.01).fit(ds.train)
    assert_same_metrics(evaluate_climate(m, ds).metrics, reference_climate(m, ds))
