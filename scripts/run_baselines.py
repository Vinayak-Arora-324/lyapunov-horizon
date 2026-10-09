"""Fit and evaluate the baselines; write results/baselines.json and figures/.

    python scripts/run_baselines.py
    python scripts/run_baselines.py --data data/ks_L22_N64.npz --no-figures

The weather and climate protocol lives in lyapunov_horizon/evaluate.py and is
shared with every other model. The local rule's radius and input noise are chosen on the validation set only.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lyapunov_horizon.baselines import (Climatology, LinearDMD, LocalPolynomial,
                                        Persistence, PerfectModel, rollout)
from lyapunov_horizon.data import load
from lyapunov_horizon.evaluate import (VPT_THRESHOLD, evaluate_climate,
                                       evaluate_weather)
from lyapunov_horizon.ks import KS
from lyapunov_horizon.metrics import autocorrelation, energy_spectrum


def select_local_rule(ds, spl, log):
    """Pick radius and input noise for the local rule by validation VPT."""
    # 8 Lyapunov-time rollouts with truncated step counts, kept as originally
    # run so the selected rule (and hence the published numbers) is unchanged.
    best = None
    for r in (6, 7, 8):
        for noise in (0.01, 0.02, 0.03):
            m = LocalPolynomial(radius=r, input_noise=noise).fit(ds.train)
            res = evaluate_weather(m, ds, "val", spacing=int(2 * spl), horizon=int(8 * spl))
            vpt = res.metrics["vpt_mean"]
            stable = res.diverged == 0
            log(f"    r={r} noise={noise:.2f}σ  val VPT {vpt:.2f}"
                f"{'' if stable else '  (some rollouts diverge)'}")
            if stable and (best is None or vpt > best[0]):
                best = (vpt, r, noise)
    if best is None:
        raise RuntimeError("no stable local rule found")
    return best[1], best[2]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/ks_L22_N64.npz")
    p.add_argument("--ic-error", type=float, default=1e-3,
                   help="initial-condition error (fraction of sigma) for the true-solver ceiling")
    p.add_argument("--out", default="results/baselines.json")
    p.add_argument("--figures", default="figures")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # model names contain σ (Windows consoles)
    log = lambda s: print(s, flush=True)

    ds = load(a.data)
    spl = ds.steps_per_lyapunov_time
    sigma = ds.sigma
    log(f"Data: {a.data}  L={ds.L:g}  sigma_u={sigma:.3f}  "
        f"Lyapunov time={ds.lyapunov_time:.1f} ({spl:.0f} steps)")

    t0 = time.time()
    log("Selecting the local rule on the validation set...")
    r, noise = select_local_rule(ds, spl, log)

    models = [Persistence(), Climatology(), LinearDMD(),
              LocalPolynomial(radius=r, input_noise=noise)]
    for m in models:
        m.fit(ds.train)
    N = ds.train.shape[-1]
    ks = KS(L=ds.L, N=N)
    perfect = PerfectModel(ks, ds.dt)
    perfect.name = f"True solver, IC error {a.ic_error:g}σ"

    # ------------------------------------------------------------ weather
    results, curves = {}, {}
    for m in models + [perfect]:
        w = evaluate_weather(m, ds, ic_error=a.ic_error if m is perfect else 0.0)
        if not curves:
            log(f"Weather: {w.nrmse.shape[0]} rollouts of {w.nrmse.shape[1] - 1} steps "
                f"(10 Lyapunov times)")
        curves[m.name] = w.nrmse
        results[m.name] = w.metrics
        log(f"  {m.name:38s} 1-step {w.metrics['one_step_nrmse']:.1e}"
            f"   VPT {w.metrics['vpt_mean']:.2f} Lyapunov times")

    # ------------------------------------------------------------ climate
    T = ds.test.shape[1] - 1
    long_runs = {}
    for m in models:
        if isinstance(m, Climatology):
            continue
        c = evaluate_climate(m, ds)
        if not long_runs:
            true_long, max_lag = c.truth, c.max_lag
            log(f"Climate: free runs of {T} steps ({T / spl:.0f} Lyapunov times), "
                f"statistics after {T + 1 - c.run.shape[1]} steps")
        long_runs[m.name] = c.run
        sc = c.metrics
        results[m.name].update(sc)
        log(f"  {m.name:38s} spectrum {sc['spectrum_err']:.2f}  pdf W1 {sc['pdf_w1']:.3f}"
            f"  ACF {sc['acf_err']:.3f}  diverged {sc['diverged']:.0%}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "data": a.data, "lambda_1": ds.lambda_1, "lyapunov_time": ds.lyapunov_time,
        "sigma_u": sigma, "local_rule": {"radius": r, "input_noise": noise},
        "vpt_threshold": VPT_THRESHOLD, "models": results}, indent=2))
    log(f"Saved {out}  [{time.time() - t0:.0f}s]")

    if not a.no_figures:
        make_figures(ds, spl, sigma, models, curves, long_runs, true_long,
                     max_lag, Path(a.figures))
        log(f"Figures in {a.figures}/")


def make_figures(ds, spl, sigma, models, curves, long_runs, true_long, max_lag, fig_dir):
    import matplotlib.pyplot as plt

    from lyapunov_horizon.plotting import (DIVERGING, INK, INK_2, TRUTH,
                                           apply_style, color_for)
    apply_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    local = next(m for m in models if isinstance(m, LocalPolynomial))

    # 1. Space-time: truth, local rule, difference -------------------------
    n = int(round(6 * spl))
    u0 = ds.test[:1, 0].astype(np.float64)
    true = ds.test[0, :n + 1]
    pred = rollout(local, u0, n)[0]
    t = np.arange(n + 1) / spl
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.2), sharex=True,
                             gridspec_kw={"hspace": 0.35})
    vmax = 3.0
    panels = [(true, "True KS trajectory"),
              (pred, f"{local.name}, rolled out from the same start"),
              (pred - true, "Difference")]
    for ax, (field, title) in zip(axes, panels):
        im = ax.pcolormesh(t, ds.x, field.T, cmap=DIVERGING, vmin=-vmax, vmax=vmax,
                           shading="auto", rasterized=True)
        ax.set_title(title)
        ax.set_ylabel("x")
        ax.grid(False)
    axes[-1].set_xlabel("time  (Lyapunov times)")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("u", color=INK_2)
    cb.outline.set_visible(False)
    fig.savefig(fig_dir / "spacetime.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Error growth -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(1.0, color=INK_2, lw=1, ls="--")
    ax.text(9.95, 0.98, "predicting the mean", ha="right", va="top", color=INK_2, fontsize=9)
    ax.axhline(0.5, color=INK_2, lw=1, ls=":")
    ax.text(10, 0.52, "valid-prediction threshold", ha="right", va="bottom",
            color=INK_2, fontsize=9)
    for name, err in curves.items():
        if name == "Climatology":
            continue
        tt = np.arange(err.shape[1]) / spl
        ax.plot(tt, np.median(err, axis=0), color=color_for(name), label=name)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("time since start of rollout  (Lyapunov times)")
    ax.set_ylabel("normalised RMSE")
    ax.set_title("How long does each model track the true trajectory?")
    ax.legend(loc="lower right", fontsize=9)
    fig.text(0.01, -0.02, f"Median over {len(err)} test rollouts. An error of ~1.41 means "
             "prediction and truth are unrelated states on the attractor.",
             color=INK_2, fontsize=8.5)
    fig.savefig(fig_dir / "error_growth.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Climate: spectrum and autocorrelation ------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    k = 2 * np.pi / ds.L * np.arange(true_long.shape[-1] // 2 + 1)
    kmax = true_long.shape[-1] // 4
    a1.semilogy(k[1:kmax + 1], energy_spectrum(true_long)[1:kmax + 1],
                color=TRUTH, label="Truth", lw=2.5)
    lags = np.arange(max_lag + 1) / spl
    a2.plot(lags, autocorrelation(true_long, max_lag), color=TRUTH, label="Truth", lw=2.5)
    for name, run in long_runs.items():
        ok = np.isfinite(run).all(axis=(1, 2))
        if not ok.any():
            continue
        c = color_for(name)
        a1.semilogy(k[1:kmax + 1], energy_spectrum(run[ok])[1:kmax + 1],
                    color=c, label=name, ls="--" if name == "Persistence" else "-")
        a2.plot(lags, autocorrelation(run[ok], max_lag), color=c, label=name,
                ls="--" if name == "Persistence" else "-")
    a1.set_xlabel("wavenumber k")
    a1.set_ylabel("energy  ⟨|û_k|²⟩")
    a1.set_title("Long-run energy spectrum")
    a2.set_xlabel("lag  (Lyapunov times)")
    a2.set_ylabel("autocorrelation of u")
    a2.set_title("Temporal autocorrelation")
    a2.axhline(0, color=INK_2, lw=0.8)
    a1.legend(fontsize=9, loc="lower left")
    fig.text(0.01, -0.03, "Free-running models after 12 Lyapunov times of spin-up. "
             "Persistence matches the spectrum exactly but never moves, which only "
             "the autocorrelation reveals.", color=INK_2, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "climate.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
