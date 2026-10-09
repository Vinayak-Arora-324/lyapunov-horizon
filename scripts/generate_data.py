"""Simulate the KS equation and save train/val/test trajectories.

    python scripts/generate_data.py                 # defaults: L=22, N=64
    python scripts/generate_data.py --L 44 --N 128  # a larger, more chaotic box

Also estimates the largest Lyapunov exponent and stores it with the data,
so every later script measures time in Lyapunov times.
"""

import argparse
import json
import time
from pathlib import Path

from lyapunov_horizon.data import Dataset, generate_split, save
from lyapunov_horizon.ks import KS
from lyapunov_horizon.lyapunov import largest_lyapunov


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=float, default=22.0, help="domain length")
    p.add_argument("--N", type=int, default=64, help="grid points")
    p.add_argument("--h", type=float, default=0.05, help="solver time step")
    p.add_argument("--dt", type=float, default=0.25, help="time between saved snapshots")
    p.add_argument("--burn-in", type=float, default=500.0,
                   help="time discarded before saving, to land on the attractor")
    p.add_argument("--samples", type=int, default=4000, help="snapshots per trajectory")
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-test", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    a = p.parse_args()

    out = Path(a.out or f"data/ks_L{a.L:g}_N{a.N}.npz")
    ks = KS(L=a.L, N=a.N, h=a.h)

    t0 = time.time()
    print(f"Estimating largest Lyapunov exponent (L={a.L:g}, N={a.N})...")
    lyap = largest_lyapunov(ks, seed=a.seed + 1000)
    lam = lyap["lambda_1"]
    print(f"  lambda_1 = {lam:.4f} +/- {lyap['stderr']:.4f}"
          f"  ->  Lyapunov time = {1 / lam:.1f} time units"
          f"  ({1 / lam / a.dt:.0f} snapshots)  [{time.time() - t0:.0f}s]")

    splits = {}
    # Separate seeds give independent initial conditions for each split.
    for i, (name, n) in enumerate([("train", a.n_train), ("val", a.n_val), ("test", a.n_test)]):
        t1 = time.time()
        splits[name] = generate_split(ks, n, a.samples, a.dt, a.burn_in, seed=a.seed + i)
        span = a.samples * a.dt * lam
        print(f"  {name:5s}: {n} trajectories x {a.samples + 1} snapshots"
              f"  ({span:.0f} Lyapunov times each)  [{time.time() - t1:.0f}s]")

    ds = Dataset(**splits, x=ks.x, dt=a.dt, L=a.L, lambda_1=lam)
    save(out, ds, N=a.N, h=a.h, lambda_1_stderr=lyap["stderr"])
    out.with_suffix(".json").write_text(json.dumps(
        {"args": vars(a), "lyapunov": lyap, "sigma_u": ds.sigma}, indent=2))
    print(f"Saved {out}  (sigma_u = {ds.sigma:.3f})  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
