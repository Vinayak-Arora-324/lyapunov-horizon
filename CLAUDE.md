# lyapunov-horizon — context for Claude Code

## What this project is

A learned ("LLM-style") emulator of the 1D Kuramoto–Sivashinsky (KS) equation, trained on a laptop CPU. Instead of solving the PDE, a model learns the one-step map u(t) → u(t + dt) from simulated data, the way an LLM learns next-token prediction. It is then rolled out autoregressively, feeding its own predictions back in.

The central question: **how far past the chaos-limited prediction horizon can a learned model stay useful?** Every model is judged on two separate axes:

- **Weather:** how long does a rollout track the true trajectory, measured in Lyapunov times?
- **Climate:** after it has inevitably diverged, does it still behave like the real system (spectrum, value distribution, temporal autocorrelation)?

Owner: Vinayak, an MSci Physics student. This is a personal research project, so explain the physics and ML reasoning behind changes, not only the code.

## Environment

- WSL on Windows. The repo lives at `/mnt/e/Projects/lyapunov-horizon`. It sits on the Windows drive, so file I/O is slower than in the WSL home folder, but computation runs at full speed.
- CPU only: AMD Ryzen 7 7840HS (8 cores / 16 threads). No CUDA. Keep each training run under ~1 hour.
- The existing tests passed on Python 3.14. If PyTorch has no wheel for 3.14, create the venv with Python 3.12 (`uv venv --python 3.12`).
- Set-up: `pip install -e ".[dev]"`, then `python scripts/generate_data.py` (~1 min, writes `data/ks_L22_N64.npz`, ~100 MB, git-ignored), then `python scripts/run_baselines.py` (~13 min; the local-rule sweep is single-threaded), then `pytest` (`-m "not slow"` skips the Lyapunov test).
- Run git from WSL only, not Git for Windows as well, to avoid line-ending and file-mode noise.

## Code layout

```
src/lyapunov_horizon/
  ks.py         Batched pseudo-spectral ETDRK4 solver (Kassam & Trefethen 2005). KS(L=22, N=64, h=0.05)
  lyapunov.py   Largest Lyapunov exponent, two-trajectory (Benettin) method
  data.py       Dataset dataclass, generate_split(), save()/load(); ds.sigma = std of train
  baselines.py  Model interface (fit/step), Persistence, Climatology, LinearDMD,
                LocalPolynomial, PerfectModel, rollout() (NaN-marks blow-ups)
  metrics.py    nrmse, valid_prediction_time, energy_spectrum, autocorrelation, climate_scores
  evaluate.py   Shared protocol: evaluate_weather(model, ds, split) / evaluate_climate(...)
                -> WeatherResult / ClimateResult (metrics dict + curves/runs for plots)
  plotting.py   Shared figure style; fixed colour per model (color_for)
scripts/
  generate_data.py   Lyapunov estimate + train/val/test (64/16/32 trajectories x 4001 snapshots)
  run_baselines.py   Selects the local rule on val, evaluates on test, writes results/ and figures/
tests/          test_ks.py (solver), test_baselines_metrics.py, test_evaluate.py
```

Every model implements `step(u: np.ndarray[B, N]) -> np.ndarray[B, N]`, which advances one snapshot (dt = 0.25). `rollout(model, u0, n_steps)` handles the autoregressive loop and divergence. New models should fit this interface so that all evaluation code is shared.

## Fixed setup and evaluation protocol (do not change silently)

- KS: u_t + u u_x + u_xx + u_xxxx = 0, periodic, L = 22, N = 64, snapshot dt = 0.25, solver h = 0.05, burn-in 500.
- λ₁ ≈ 0.049 (literature: 0.048, Cvitanović et al. 2010). Lyapunov time T_λ ≈ 20.4 time units ≈ 82 snapshots. **All times are reported in T_λ.**
- Errors are normalised by σ_u (the std of the training set, ≈ 1.19). 1.0 = predicting the mean; ≈1.41 = two unrelated attractor states.
- **Weather:** start a rollout every 2 T_λ along each test trajectory, 10 T_λ long (640 rollouts). Report the median NRMSE curve and the valid prediction time (VPT): the first time NRMSE > 0.5.
- **Climate:** free-run from each test trajectory's first state for the full length (49 T_λ), discard the first 12 T_λ, compare with the true test data. Metrics: spectrum error (mean |Δlog10 E(k)| for k ≤ N/4), Wasserstein-1 of u / σ, autocorrelation RMS error up to a 2 T_λ lag, and the fraction of runs that diverged.
- **All hyperparameter and model selection uses the validation split only.** The test split is for final numbers.

## Current baseline results (test split)

| Model | 1-step NRMSE | VPT (T_λ) | Spectrum err | W₁ | ACF err |
|---|---|---|---|---|---|
| Persistence | 0.038 | 0.22 | 0.11 | 0.03 | 0.71 |
| Linear (DMD) | 0.038 | 0.22 | 2.60 | 0.74 | 0.64 |
| Local quadratic (r=7, input noise 0.01σ) | 0.003 | 0.94 | 0.49 | 0.07 | 0.37 |
| True solver, 0.1% initial error | — | 6.1 | — | — | — |

What these results already show (keep these in mind):

1. **The best linear model is persistence.** On a statistically stationary attractor the energy in each Fourier mode is constant on average, which forces the best linear one-step predictor to be almost the identity. This is physics, not a bug.
2. **One-step accuracy ≠ stable rollouts.** Without input noise, the local rule is 20× better per step than persistence, yet every rollout blows up. Training on noisy inputs (with clean targets) stabilises it. Expect the same problem with neural models.
3. **Static statistics can be fooled.** Persistence has a perfect spectrum and distribution because it never moves. The local rule's free runs fall into a near-periodic oscillation (period ≈ 1 T_λ) rather than chaos, and only the autocorrelation catches it. Always report all climate metrics together.
4. **Chaos sets the ceiling.** Even the exact equations lose the trajectory after ~6 T_λ with a 0.1% initial error.

## Next steps (in order)

### 1. Commit and reproduce
- Make the first commit (check `git status` first; `.claude/` lock files must not be committed).
- Regenerate the data, rerun the baselines, and confirm the numbers match `results/baselines.json` closely.

### 2. Refactor the evaluation into the package
- Move the weather/climate protocol out of `scripts/run_baselines.py` into `src/lyapunov_horizon/evaluate.py`, with functions like `evaluate_weather(model, ds, ...)` and `evaluate_climate(model, ds, ...)` that return metric dicts and the curves needed for plotting.
- `run_baselines.py` must call these functions and **reproduce the existing results exactly** (same seeds, same windows).
- Add tests for the new module.

### 3. First neural emulator: a 1D Fourier Neural Operator (PyTorch, CPU)
- Add torch as an optional dependency (`[project.optional-dependencies] torch = ["torch"]`) so the core package still installs without it.
- `src/lyapunov_horizon/fno.py`: a small 1D FNO (lifting to width ~32–64, ~4 spectral layers, ~16 Fourier modes, GELU, ~100k–500k params). It's periodic by construction, which matches the domain. Normalise inputs by σ_u. **Predict the increment** u_{n+1} − u_n, scaled.
- Wrap the trained network in a `Model` subclass whose `step()` takes and returns numpy arrays (`torch.no_grad()`), so `rollout()` and all metrics are reused unchanged.
- `scripts/train_fno.py`: AdamW, cosine learning-rate schedule, `torch.set_num_threads(8)`, a fixed seed, and logging of train/val loss plus a validation VPT every few epochs. Save checkpoints to `checkpoints/` (add it to `.gitignore`) and final metrics to `results/fno*.json`.
- Training in two phases:
  1. One-step MSE with input-noise injection (tune the noise level on val; start around 0.01σ).
  2. Multi-step (unrolled) loss: roll out K steps inside training and backpropagate through the rollout, with a curriculum such as K = 2 → 4 → 8 → 16. This is the standard fix for rollout instability.
- Optional: data augmentation using the KS symmetries, translation (already built into the FNO) and reflection u(x) → −u(−x).
- Add the FNO to the figures (`error_growth.png` and `climate.png`) and to the README results table.

**Success criteria for step 3:** mean test VPT > 0.94 T_λ (beats the local rule); 0% divergence in 49-T_λ free runs; climate metrics better than the local rule (spectrum error < 0.49, ACF error < 0.37), with the autocorrelation not showing the periodic lock-in.

### Later (roadmap in README)
- Probabilistic emulator: ensembles, calibration, CRPS.
- Latent (JEPA-style) prediction vs. raw-state prediction.
- Larger domains (L = 44, 100).

## Conventions
- Keep the `Model.step` interface and the shared evaluation; never evaluate a new model with different code than the baselines.
- Report times in Lyapunov times and errors normalised by σ_u.
- When a result changes, update the README table and figures in the same change.
- Tests must pass (`pytest -m "not slow"` for quick runs).
