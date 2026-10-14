"""Train the FNO emulator in two phases, select on validation, evaluate on test.

    python scripts/train_fno.py                                   # full run (~45 min)
    python scripts/train_fno.py --tag fno_noise0.02 --noise 0.02 --phase1-epochs 4 \\
        --no-phase2 --skip-test                                  # short val-only sweep run
    python scripts/train_fno.py --init checkpoints/fno_phase1.pt --lr2 1e-4 \\
        --tag fno_p2 --skip-test                                 # phase 2 only

Phase 1: one-step loss. Inputs get zero-mean Gaussian noise (fraction `--noise`
of sigma_u), targets stay clean, so the network learns to pull slightly-off
states back towards the attractor (the trick that stabilised the local rule).

Phase 2: multi-step loss. Starting from the best phase-1 network, roll out K
steps inside training and backpropagate through the whole rollout, with a
curriculum over K (default 2 -> 4 -> 8 -> 16). The network then sees, and is
penalised for, the errors that its own predictions compound into.

Both losses are MSE on the state error divided by the one-step increment scale,
so K = 1 is exactly the phase-1 loss. Each sample is reflected, u(x) -> -u(-x),
with probability 1/2 (an exact KS symmetry).

Selection: after every evaluation the model is scored on the validation split
with the shared weather/climate protocol; the kept model has the highest
validation VPT among those whose validation rollouts and free runs never
diverge and whose validation climate beats the local rule's (spectrum error
< 0.49, ACF error < 0.37 by default). Without the climate condition, selection
drifts towards low input noise, which forecasts slightly longer but whose free
runs settle on the wrong statistics (an over-energetic small-scale tail). The test split is touched once, at the end (not at all with
--skip-test). Writes checkpoints/<tag>_phase1.pt, checkpoints/<tag>.pt and
results/<tag>.json.
"""

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lyapunov_horizon.data import load
from lyapunov_horizon.evaluate import evaluate_climate, evaluate_weather
from lyapunov_horizon.fno import (FNO1d, Emulator, FNOModel, increment_scale,
                                  load_checkpoint, reflect, save_checkpoint)


def parse_curriculum(s):
    """'2:1500:128,4:1500:64' -> [(K, steps, batch), ...]"""
    return [tuple(int(v) for v in stage.split(":")) for stage in s.split(",")]


def cosine_lr(step, total, lr_max, lr_min, warmup):
    if step < warmup:
        return lr_max * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


class Windows:
    """Random (start state, next K states) windows from a (n_traj, n_time, N) tensor,
    with input noise and random reflection."""

    def __init__(self, traj, sigma, noise, gen):
        self.traj, self.sigma, self.noise, self.gen = traj, sigma, noise, gen

    def batch(self, traj_idx, t_idx, k):
        offs = torch.arange(k + 1)
        w = self.traj[traj_idx[:, None], t_idx[:, None] + offs]   # (B, k+1, N)
        flip = torch.rand(len(w), generator=self.gen) < 0.5
        w = torch.where(flip[:, None, None], reflect(w), w)
        u0 = w[:, 0]
        if self.noise:
            eps = torch.randn(u0.shape, generator=self.gen)
            u0 = u0 + self.noise * self.sigma * (eps - eps.mean(-1, keepdim=True))
        return u0, w[:, 1:]

    def random(self, batch, k):
        n_traj, n_time, _ = self.traj.shape
        ti = torch.randint(n_traj, (batch,), generator=self.gen)
        tt = torch.randint(n_time - k, (batch,), generator=self.gen)
        return self.batch(ti, tt, k)


def rollout_loss(emu, u0, target):
    """Mean squared state error over the K predicted steps, in units of the increment scale."""
    pred = emu.unroll(u0, target.shape[1])
    return (((pred - target) / emu.delta_scale) ** 2).mean()


class Selector:
    """Validation scoring and best-model bookkeeping."""

    def __init__(self, ds, log, max_spectrum_err, max_acf_err):
        self.ds, self.log, self.history = ds, log, []
        self.max_spectrum_err, self.max_acf_err = max_spectrum_err, max_acf_err
        self.best = None   # (val VPT, label, state_dict)

    def __call__(self, emu, label, train_loss, extra=None):
        t0 = time.time()
        m = FNOModel(emu)
        w = evaluate_weather(m, self.ds, "val")
        c = evaluate_climate(m, self.ds, "val")
        emu.train()
        vpt, cm = w.metrics["vpt_mean"], c.metrics
        # NaN climate scores (any divergence) compare False, so they never qualify.
        eligible = (w.diverged == 0 and cm["diverged"] == 0
                    and cm["spectrum_err"] < self.max_spectrum_err
                    and cm["acf_err"] < self.max_acf_err)
        rec = {"label": label, "train_loss": train_loss, "val_one_step_nrmse":
               w.metrics["one_step_nrmse"], "val_vpt": vpt, "val_weather_diverged": w.diverged,
               **{f"val_{k}": v for k, v in c.metrics.items()}, **(extra or {})}
        self.history.append(rec)
        is_best = eligible and (self.best is None or vpt > self.best[0])
        if is_best:
            self.best = (vpt, label, copy.deepcopy(emu.state_dict()))
        self.log(f"  [{label}] loss {train_loss:.4f}  val 1-step {rec['val_one_step_nrmse']:.1e}"
                 f"  VPT {vpt:.3f}  diverged {w.diverged:.0%}/{c.metrics['diverged']:.0%}"
                 f"  spec {c.metrics['spectrum_err']:.2f}  ACF {c.metrics['acf_err']:.3f}"
                 f"{'  *best*' if is_best else ''}  ({time.time() - t0:.0f}s)")
        return rec


def train_phase1(emu, data, a, select, log):
    opt = torch.optim.AdamW(emu.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    n_traj, n_time, _ = data.traj.shape
    n_pairs = n_traj * (n_time - 1)
    steps_per_epoch = n_pairs // a.batch
    total = a.phase1_epochs * steps_per_epoch
    step = 0
    for epoch in range(1, a.phase1_epochs + 1):
        t0, run = time.time(), 0.0
        perm = torch.randperm(n_pairs, generator=data.gen)
        for i in range(steps_per_epoch):
            idx = perm[i * a.batch:(i + 1) * a.batch]
            u0, target = data.batch(idx // (n_time - 1), idx % (n_time - 1), 1)
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, a.lr, a.lr * 0.01, warmup=500)
            loss = rollout_loss(emu, u0, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emu.parameters(), a.clip)
            opt.step()
            run += loss.item()
            step += 1
        log(f"phase 1 epoch {epoch}/{a.phase1_epochs}: loss {run / steps_per_epoch:.4f}"
            f"  [{time.time() - t0:.0f}s]")
        if epoch % a.eval_every == 0 or epoch == a.phase1_epochs:
            select(emu, f"phase1 epoch {epoch}", run / steps_per_epoch)


def train_phase2(emu, data, a, select, log):
    opt = torch.optim.AdamW(emu.parameters(), lr=a.lr2, weight_decay=a.weight_decay)
    for k, steps, batch in parse_curriculum(a.curriculum):
        t0, run, recent = time.time(), 0.0, []
        for step in range(steps):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, steps, a.lr2, a.lr2 * 0.1, warmup=100)
            u0, target = data.random(batch, k)
            loss = rollout_loss(emu, u0, target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emu.parameters(), a.clip)
            opt.step()
            recent.append(loss.item())
            if (step + 1) % 500 == 0 or step + 1 == steps:
                run = float(np.mean(recent))
                log(f"phase 2 K={k} step {step + 1}/{steps}: loss {run:.4f}"
                    f"  [{time.time() - t0:.0f}s]")
                recent = []
        select(emu, f"phase2 K={k}", run)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/ks_L22_N64.npz")
    p.add_argument("--tag", default="fno", help="name for checkpoints/ and results/ files")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--modes", type=int, default=20)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--noise", type=float, default=0.003, help="input noise, fraction of sigma_u")
    p.add_argument("--phase1-epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=2, help="phase-1 epochs between val scores")
    p.add_argument("--no-phase2", action="store_true")
    p.add_argument("--init", help="start from this checkpoint and skip phase 1 "
                                  "(it is scored on val as the phase-1 candidate)")
    p.add_argument("--curriculum", default="2:1500:128,4:1500:64,8:1500:32,16:1500:32",
                   help="phase-2 stages K:steps:batch")
    p.add_argument("--lr2", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-spectrum-err", type=float, default=0.49,
                   help="selection: max validation spectrum error (local rule: 0.49)")
    p.add_argument("--max-acf-err", type=float, default=0.37,
                   help="selection: max validation ACF error (local rule: 0.37)")
    p.add_argument("--skip-test", action="store_true", help="validation only (for sweeps)")
    a = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    log = lambda s: print(s, flush=True)

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    ds = load(a.data)
    sigma, delta = ds.sigma, increment_scale(ds.train)
    if a.init:
        emu, init_meta = load_checkpoint(a.init)
        emu.train()
        a.width, a.modes, a.layers = (emu.net.config[k] for k in ("width", "modes", "layers"))
        a.noise = init_meta["args"]["noise"]
    else:
        emu = Emulator(FNO1d(a.width, a.modes, a.layers), sigma, delta)
    n_params = sum(p.numel() for p in emu.parameters())
    log(f"Data: {a.data}  sigma_u={sigma:.3f}  increment scale={delta:.4f}  "
        f"FNO width={a.width} modes={a.modes} layers={a.layers}: {n_params / 1e3:.0f}k params")
    data = Windows(torch.from_numpy(ds.train), sigma, a.noise, gen)
    select = Selector(ds, log, a.max_spectrum_err, a.max_acf_err)
    t_start = time.time()

    if a.init:
        log(f"Phase 1 skipped: starting from {a.init} (input noise {a.noise:g}σ)")
        select(emu, f"init {a.init}", float("nan"))
    else:
        log(f"Phase 1: one-step loss, input noise {a.noise:g}σ, {a.phase1_epochs} epochs")
        train_phase1(emu, data, a, select, log)
    if select.best is None:
        raise RuntimeError("no phase-1 checkpoint was stable with an acceptable climate on validation")
    phase1_label, phase1_state = select.best[1], copy.deepcopy(select.best[2])
    ckpt_dir = Path("checkpoints")
    meta = {"args": vars(a), "data": a.data}
    emu.load_state_dict(phase1_state)
    if not a.init:
        save_checkpoint(ckpt_dir / f"{a.tag}_phase1.pt", emu, selected=phase1_label, **meta)
    log(f"Phase 1 done [{(time.time() - t_start) / 60:.1f} min]; continuing from {phase1_label}")

    if not a.no_phase2:
        log(f"Phase 2: unrolled loss, curriculum {a.curriculum}")
        train_phase2(emu, data, a, select, log)
    final_label = select.best[1]
    emu.load_state_dict(select.best[2])
    save_checkpoint(ckpt_dir / f"{a.tag}.pt", emu, selected=final_label, **meta)
    log(f"Selected {final_label} (val VPT {select.best[0]:.3f})  "
        f"[{(time.time() - t_start) / 60:.1f} min total]")

    result = {"data": a.data, "args": vars(a), "params": n_params, "increment_scale": delta,
              "train_minutes": (time.time() - t_start) / 60, "phase1_selected": phase1_label,
              "selected": final_label, "history": select.history}
    if not a.skip_test:
        log("Test split:")
        result["test"] = {}
        for label, state in [("phase 1", phase1_state), ("final", select.best[2])]:
            emu.load_state_dict(state)
            m = FNOModel(emu)
            metrics = {**evaluate_weather(m, ds).metrics, **evaluate_climate(m, ds).metrics}
            result["test"][label] = metrics
            log(f"  {label:8s} 1-step {metrics['one_step_nrmse']:.1e}  VPT {metrics['vpt_mean']:.2f}"
                f"  spectrum {metrics['spectrum_err']:.2f}  W1 {metrics['pdf_w1']:.3f}"
                f"  ACF {metrics['acf_err']:.3f}  diverged {metrics['diverged']:.0%}")
    out = Path("results") / f"{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    log(f"Saved {out}")


if __name__ == "__main__":
    main()
