# lyapunov-horizon

Learned emulators for the 1D Kuramoto–Sivashinsky equation: rollout horizons, long-run statistics and calibration.

The project asks one question: **how far past the chaos-limited prediction horizon can a learned model stay useful?** A model learns to predict the next state from the current one and is then rolled out autoregressively. It is judged on *weather* (how long it tracks the true trajectory, in Lyapunov times) and *climate* (whether it still behaves like the real system after it has diverged). Everything runs on a laptop CPU.

The best model so far, a 199k-parameter Fourier Neural Operator, tracks the true trajectory for **4.5 Lyapunov times** on average (the best baseline manages 0.94; the exact equations started with a 0.1% error manage 6.1), and its free runs keep the right spectrum, value distribution and autocorrelation.

![True trajectory vs. the local-rule baseline](figures/spacetime.png)

```bash
pip install -e ".[dev]"
pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/generate_data.py
python scripts/train_fno.py
python scripts/run_baselines.py --fno checkpoints/fno.pt
```

**See [documentation.md](documentation.md)** for the physics, setup, data format, evaluation protocol, how to add a model, the results, the FNO emulator, and the roadmap.
