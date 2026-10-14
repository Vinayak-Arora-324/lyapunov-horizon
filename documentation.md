# lyapunov-horizon documentation

1. [Overview](#overview)
2. [The system](#the-system)
3. [Setup](#setup)
4. [Running the pipeline](#running-the-pipeline)
5. [Data](#data)
6. [Evaluation protocol](#evaluation-protocol)
7. [Adding a model](#adding-a-model)
8. [Results](#results)
9. [The FNO emulator](#the-fno-emulator)
10. [Code layout](#code-layout)
11. [Roadmap](#roadmap)
12. [References](#references)

## Overview

The project asks one question: **how far past the chaos-limited prediction horizon can a learned model stay useful?** Instead of solving the PDE, a model learns to predict the next state from the current one (self-supervised, like next-token prediction in an LLM) and is then rolled out autoregressively, feeding its own predictions back in. Every model is judged twice:

- **Weather:** how long does it track the true trajectory, measured in Lyapunov times?
- **Climate:** once it has inevitably diverged, does it still behave like the real system?

Everything runs on a laptop CPU.

## The system

$$u_t + u\,u_x + u_{xx} + u_{xxxx} = 0, \qquad x \in [0, L),\ \text{periodic}$$

The defaults use the standard chaotic benchmark: L = 22, with 64 grid points and snapshots every 0.25 time units. The term $u_{xx}$ injects energy at large scales, $u_{xxxx}$ removes it at small scales, and $u\,u_x$ moves it between them. The result is spatiotemporal chaos, a 1D cousin of turbulence.

- **Solver:** pseudo-spectral ETDRK4 (Kassam & Trefethen 2005), batched over trajectories, with internal step h = 0.05 (5 substeps per snapshot). Tests check 4th-order convergence, conservation of the mean, and the evolution of small linear modes.
- **Lyapunov exponent:** measured with the two-trajectory (Benettin) method at **λ₁ ≈ 0.049 ± 0.001**. This matches the published 0.048 (Cvitanović, Davidchack & Siminos 2010). One **Lyapunov time** is T_λ = 1/λ₁ ≈ 20.4 time units, or about 82 snapshots (81.6), and every result is measured in units of it.

## Setup

Python ≥ 3.10; developed on Python 3.14 under WSL.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\activate
pip install -e ".[dev]"
```

The FNO emulator needs PyTorch (the `torch` extra). On a CPU-only machine, install the CPU build to avoid downloading CUDA libraries:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The editable install records the repository's absolute path. If you move the repository, run `pip install -e ".[dev]"` again.

## Running the pipeline

```bash
python scripts/generate_data.py   # ~1 min: Lyapunov exponent + 64/16/32 train/val/test trajectories
python scripts/run_baselines.py   # ~13 min (mostly single-threaded): fits, evaluates, writes results/ and figures/
python scripts/train_fno.py       # ~25 min on 8 threads: trains the FNO, writes checkpoints/fno.pt and results/fno.json
python scripts/run_baselines.py --fno checkpoints/fno.pt   # figures with the FNO included
pytest                            # all tests; pytest -m "not slow" skips the Lyapunov check
```

`generate_data.py` options: `--L`, `--N` (domain length and grid; e.g. `--L 44 --N 128` for a larger, more chaotic box), `--h` (solver step), `--dt` (snapshot interval), `--burn-in`, `--samples` (snapshots per trajectory), `--n-train`, `--n-val`, `--n-test`, `--seed`, `--out`. The default output is `data/ks_L22_N64.npz`.

`run_baselines.py` options: `--data` (input dataset), `--ic-error` (initial-condition error for the true-solver ceiling, as a fraction of σ_u; default 1e-3), `--out` (default `results/baselines.json`), `--figures` (default `figures/`), `--no-figures`, `--fno` (a trained FNO checkpoint to draw in the figures; its numbers stay in `results/fno.json`).

`train_fno.py` options: architecture (`--width` 48, `--modes` 20, `--layers` 4), `--noise` (phase-1 input noise as a fraction of σ_u; default 0.003), `--phase1-epochs` (12), `--batch` (256), `--lr` (1e-3), `--eval-every` (phase-1 epochs between validation scores; 2), `--curriculum` (phase-2 stages as `K:steps:batch`; default `2:1500:128,4:1500:64,8:1500:32,16:1500:32`), `--lr2` (phase-2 learning rate; 1e-4), `--no-phase2`, `--init` (start from a checkpoint and skip phase 1), `--max-spectrum-err` / `--max-acf-err` (climate conditions for selection; 0.49 / 0.37), `--threads` (8), `--seed` (0), `--skip-test` (validation only, for sweeps), `--tag` (output file names; default `fno`).

## Data

`generate_data.py` writes two files:

- `data/ks_L22_N64.npz` (~100 MB, git-ignored, so regenerate it rather than committing it). Arrays `train`, `val`, `test` of shape (trajectories, 4001, 64), float32, plus `x`, `dt`, `L`, `lambda_1`, `N`, `h` and `lambda_1_stderr`. Load it with `lyapunov_horizon.data.load()`, which returns a `Dataset`.
- `data/ks_L22_N64.json`: the arguments used, the per-run Lyapunov estimates and σ_u.

| Split | Trajectories | Snapshots each | Length | Seed |
|---|---|---|---|---|
| train | 64 | 4001 | 49 T_λ | `seed` |
| val | 16 | 4001 | 49 T_λ | `seed + 1` |
| test | 32 | 4001 | 49 T_λ | `seed + 2` |

Each trajectory starts from a random initial condition and is integrated for 500 time units before saving, so every snapshot lies on the attractor. The separate seeds make the splits independent. The Lyapunov estimate uses `seed + 1000`.

`Dataset.sigma` is the pointwise standard deviation of u over the **training split** (σ_u ≈ 1.19). All errors are normalised by it.

## Evaluation protocol

Every model, baseline or neural, is evaluated by the same two functions in `lyapunov_horizon/evaluate.py`, so the numbers are directly comparable. Multiples of T_λ are rounded to whole snapshots.

**All hyperparameter and model selection uses the validation split only.** The test split is used once, for the final numbers.

### Weather: `evaluate_weather(model, ds, split="test")`

- A rollout starts every 2 T_λ along each trajectory and runs for 10 T_λ: 640 rollouts on the test split.
- **NRMSE**: the spatial RMS error divided by σ_u. 0 is perfect, 1.0 is what predicting the mean gives, and ≈ 1.41 is the error between two unrelated states on the attractor.
- **Valid prediction time (VPT)**: the first time the NRMSE exceeds 0.5, in Lyapunov times. A rollout that never crosses is assigned one step past the horizon (≈ 10 T_λ).
- Reported: the median one-step NRMSE and the mean, median, 10% and 90% quantiles of VPT. The NRMSE curves are kept for plotting.
- `ic_error > 0` perturbs the start states by that fraction of σ_u, e.g. to measure how quickly the exact equations themselves lose the trajectory.

### Climate: `evaluate_climate(model, ds, split="test")`

The model free-runs from each trajectory's first state for the full trajectory length (49 T_λ). The first 12 T_λ are discarded and the remaining 37 T_λ are compared with the true trajectories:

| Metric | Definition |
|---|---|
| `spectrum_err` | mean \|log₁₀ E_pred(k) − log₁₀ E_true(k)\| over wavenumbers 1 ≤ k ≤ N/4 |
| `pdf_w1` | Wasserstein-1 distance between the distributions of u, divided by σ_u |
| `acf_err` | RMS difference of the temporal autocorrelation, up to a 2 T_λ lag |
| `diverged` | fraction of free runs that blew up |

All four must be read together. A model that freezes in place (persistence) has the right spectrum and distribution but the wrong dynamics, and only the autocorrelation catches it.

If any free run diverges, `spectrum_err`, `pdf_w1` and `acf_err` are NaN. Scoring only the surviving runs would let a model that usually blows up look as good as one that never does. `drop_diverged=True` scores the survivors instead. It is meant for diagnostics during training, never for reported numbers.

## Adding a model

Every model subclasses `lyapunov_horizon.baselines.Model`:

```python
class Model:
    name = "model"

    def fit(self, train: np.ndarray) -> "Model":   # train: (n_traj, n_time, N)
        return self

    def step(self, u: np.ndarray) -> np.ndarray:    # (B, N) -> (B, N), one snapshot ahead
        raise NotImplementedError
```

`step` advances a batch of states by one snapshot interval (dt = 0.25). It must treat each row independently: `rollout()` stops passing a trajectory to `step` once it has diverged, so the row index of a given trajectory changes over a rollout. A neural model should take and return NumPy arrays (running the network under `torch.no_grad()`), so that the same evaluation code applies.

`rollout(model, u0, n_steps)` runs the autoregressive loop and returns (B, n_steps + 1, N). A trajectory that becomes non-finite or exceeds amplitude 100 is marked as diverged: it is NaN from then on and is no longer stepped.

```python
from lyapunov_horizon.data import load
from lyapunov_horizon.evaluate import evaluate_climate, evaluate_weather

ds = load("data/ks_L22_N64.npz")
model = MyModel().fit(ds.train)
print(evaluate_weather(model, ds, "val").metrics)   # tune on val
print(evaluate_climate(model, ds, "val").metrics)
```

## Results

Test split.

| Model | 1-step error | Valid prediction time (T_λ) | Spectrum error | Distribution W₁ | Autocorrelation error |
|---|---|---|---|---|---|
| Persistence (u_{n+1} = u_n) | 0.038 | 0.22 | 0.11 | 0.03 | 0.71 |
| Linear / DMD | 0.038 | 0.22 | 2.60 | 0.74 | 0.64 |
| Local quadratic rule (r = 7) | 0.003 | 0.94 | 0.49 | 0.07 | 0.37 |
| **FNO** (199k params, two-phase training) | **0.0001** | **4.51** | **0.07** | **0.003** | **0.004** |
| *True solver, 0.1% initial error* | — | *6.1* | — | — | — |

Errors are normalised by σ_u; lower is better for every column except the valid prediction time. No model diverged in its free runs. The FNO's numbers are in `results/fno.json`, and it is described in [The FNO emulator](#the-fno-emulator). Climatology (always predict the mean field) has a one-step error of 0.95 and a VPT of 0.01 T_λ; it sets the "no skill" level and is left out of the climate comparison. The local rule's radius (r = 7) and training-input noise (0.01 σ_u) were chosen on the validation set. The full numbers are in `results/baselines.json`.

![True trajectory vs. the local-rule baseline](figures/spacetime.png)

![Error growth](figures/error_growth.png)

![Long-run statistics](figures/climate.png)

### What the baselines already show

1. **The best linear model is persistence.** On a statistically stationary attractor, the energy in each Fourier mode is constant on average. That forces the best linear one-step predictor to be almost the identity map, so DMD can't beat "nothing changes" in the short term. In free runs it slowly decays towards the mean field.
2. **One-step accuracy ≠ stable rollouts.** Without training noise, the local quadratic rule is 20× more accurate per step than persistence, yet every rollout blows up within a fraction of a Lyapunov time. It has never seen small-scale errors, so it amplifies them. Adding 1% noise to the training inputs teaches it to pull perturbed states back towards the attractor. This is the same instability that learned emulators face, in a model small enough to understand completely.
3. **Static statistics can be fooled.** Persistence has a perfect spectrum and value distribution because it never moves. Only the autocorrelation catches it. Meanwhile the local rule settles into an almost periodic oscillation instead of chaos: the spectrum looks plausible, but the autocorrelation says it's the wrong system.
4. **The ceiling is chaos itself.** Even the exact equations, started with a 0.1% error, lose the trajectory after about 6 Lyapunov times. No model can do much better without better initial data, because the horizon grows only like ln(1/error).

The target for a learned emulator is to beat **0.94 T_λ** while also getting the climate right.

## The FNO emulator

`lyapunov_horizon/fno.py`, trained by `scripts/train_fno.py`.

### Architecture

A 1D Fourier Neural Operator: lift u/σ_u to 48 channels, then 4 layers that each add a *spectral convolution* (a learned complex 48 × 48 matrix on each of the lowest 20 Fourier modes, all higher modes dropped) to a pointwise linear map, followed by GELU, then project back to one channel. 199k parameters.

- **Translation-equivariant by construction.** Both operations in each layer commute with shifts of the periodic grid, as the KS equation does, so the network never has to learn that physics is the same everywhere.
- **Predicts the increment.** The output is (u_{n+1} − u_n), scaled by its standard deviation (0.062, about 5% of σ_u), and added to u_n. Predicting the full state would bury the change under the much larger state.
- **Conserves the mean.** KS conserves the spatial mean of u, so the predicted increment has its mean subtracted. The emulator conserves the mean exactly, rather than approximately.
- **Covers every scored wavenumber.** The climate metric scores wavenumbers up to N/4 = 16. With 16 learned modes (0–15), k = 16 was the first mode with no learned filter: its energy came only from side effects of the nonlinearity and sat 1.6–3.1 decades above the truth, which dominated the spectrum error. 20 modes cut the spectrum error from 0.23 to 0.09 (4-epoch validation runs).

`FNOModel` wraps the network in the numpy `Model.step` interface, so it is rolled out and scored by exactly the same code as the baselines.

### Training

- **Phase 1, one-step loss (12 epochs, ~14 min).** Each input gets zero-mean Gaussian noise of 0.003 σ_u while the target stays clean, so the network learns to pull slightly-off states back towards the attractor (the same trick that stabilised the local rule). AdamW, learning rate 1e-3 with cosine decay, batch 256. Each sample is reflected, u(x) → −u(−x), with probability 1/2: an exact symmetry of KS, so it doubles the data for free.
- **Phase 2, unrolled loss (~11 min).** Starting from the best phase-1 network, the model is rolled out K steps inside training and the loss is backpropagated through the whole rollout, so it is penalised for the errors that its own predictions compound into. Curriculum K = 2 → 4 → 8 → 16, 1500 steps each, learning rate 1e-4.
- **Loss.** MSE of the predicted states, in units of the increment scale, averaged over the K steps (K = 1 is exactly the phase-1 loss). Gradients are clipped at norm 1.
- **Selection.** After every second phase-1 epoch and every phase-2 stage, the model is scored on the validation split with the shared protocol. The kept model has the highest validation VPT among checkpoints that never diverge and whose validation climate beats the local rule's (spectrum error < 0.49, ACF error < 0.37). The test split is used once, at the end.

### What the training choices showed

All of these comparisons are on the validation split; the runs are in `results/sweeps/`.

1. **Input noise trades weather against climate.** With 4 epochs of phase 1:

   | Input noise (σ_u) | VPT (T_λ) | Spectrum error | W₁ | ACF error |
   |---|---|---|---|---|
   | 0.0003 | 4.05 | 1.85 | 0.46 | 0.284 |
   | 0.001 | 4.08 | 1.27 | 0.19 | 0.147 |
   | **0.003** | 3.59 | 0.23 | 0.004 | 0.008 |
   | 0.01 | 2.56 | 0.16 | 0.007 | 0.012 |
   | 0.03 | 1.52 | 0.18 | 0.010 | 0.040 |

   Less noise gives a slightly better forecaster but a worse climate: below 0.003 σ_u the free runs' small-scale tail (k ≥ 12) sits 1.5–9 decades too high, and the value distribution drifts. Noise is what teaches the network to damp off-attractor small-scale content; with too little, nothing removes the small errors it makes every step, and the free runs settle onto a different, noisier state. This is why selection includes the climate condition: on VPT alone it would choose 0.001 σ_u.
2. **Phase 2 helps, at the right learning rate.** Starting phase 2 at 3e-4 (30× above where phase 1's schedule ends) knocked the model out of its minimum, and no stage beat phase 1 (best 4.04 vs 4.05). At 1e-4 and 3e-5, every run's best stage beat phase 1 (4.55–4.81 vs 4.05). On test, phase 2 raises VPT from 4.08 to 4.51 T_λ and cuts the autocorrelation error from 0.012 to 0.004, with the spectrum unchanged.
3. **Which curriculum stage is best is not robust.** In two phase-2 runs from the same phase-1 model with the same settings and different random batches, the best stage was K = 8 in one and K = 16 in the other. Validation VPT moves by about ±0.5 T_λ between checkpoints, so differences of that size are noise.
4. **Unrolled training was not needed for stability here.** Unlike the local rule without noise, no FNO checkpoint trained with noise ≥ 0.0003 σ_u diverged in any rollout or free run. Noise injection plus the architecture already gave stable rollouts; what phase 2 added was accuracy over longer horizons.

### Where this leaves the horizon question

The FNO tracks the true trajectory for 4.5 Lyapunov times on average (10–90% range 2.6–6.8), 4.8× longer than the local rule. The exact equations started with a 0.1% error manage 6.1. The FNO's one-step error is 1.2 × 10⁻⁴ σ_u, so the remaining gap is roughly what chaos does to an error of that size: errors grow by e per Lyapunov time, and ln(0.5 / 1.2 × 10⁻⁴) ≈ 8.3, against ln(0.5 / 10⁻³) ≈ 6.2 for the true solver. The model falls short of that because it adds a fresh error of that size at every step (82 per Lyapunov time), not just once at the start. In free runs it keeps the right climate: spectrum, value distribution and autocorrelation all match the truth, with none of the local rule's periodic lock-in.

## Code layout

```
src/lyapunov_horizon/
  ks.py         ETDRK4 solver
  lyapunov.py   largest Lyapunov exponent
  data.py       Dataset, dataset generation, save() / load()
  baselines.py  Model interface, persistence, climatology, DMD, local quadratic rule, rollout()
  metrics.py    NRMSE, valid prediction time, spectrum, distribution, autocorrelation
  evaluate.py   the shared weather/climate protocol: evaluate_weather(), evaluate_climate()
  plotting.py   shared figure style and fixed colour per model
  fno.py        1D Fourier Neural Operator, Emulator (physical units), FNOModel (numpy Model), checkpoints
scripts/
  generate_data.py   simulate the KS equation and save the splits
  run_baselines.py   select the local rule on val, evaluate on test, write results/ and figures/ (--fno: add the FNO to the figures)
  train_fno.py       two-phase FNO training, validation selection, test evaluation
tests/
  test_ks.py                  solver
  test_baselines_metrics.py   baselines, rollout, metrics
  test_evaluate.py            evaluation protocol
  test_fno.py                 FNO: equivariance, mean conservation, KS reflection symmetry, wrapper, checkpoints
```

## Roadmap

- [x] Solver, data, Lyapunov time
- [x] Baselines and evaluation (weather + climate)
- [x] Neural emulator: a 1D FNO in PyTorch on CPU (4.5 T_λ, right climate)
- [x] Training tricks for stable rollouts: noise injection, multi-step (unrolled) loss
- [ ] Transformer over spatial patches, as a second architecture
- [ ] Probabilistic emulator: ensembles, calibration, CRPS
- [ ] Latent (JEPA-style) prediction vs. raw-state prediction
- [ ] Larger domains (L = 44, 100): more chaos, shorter horizons

## References

- Kassam & Trefethen, *Fourth-order time-stepping for stiff PDEs*, SIAM J. Sci. Comput. 26 (2005)
- Cvitanović, Davidchack & Siminos, *On the state space geometry of the Kuramoto–Sivashinsky flow in a periodic domain*, SIAM J. Appl. Dyn. Syst. 9 (2010), [arXiv:0709.2944](https://arxiv.org/abs/0709.2944)
- Pathak et al., *Model-free prediction of large spatiotemporally chaotic systems from data: a reservoir computing approach*, PRL 120 (2018)
- Vlachas et al., *Backpropagation algorithms and reservoir computing in recurrent neural networks for the forecasting of complex spatiotemporal dynamics*, Neural Networks 126 (2020)
