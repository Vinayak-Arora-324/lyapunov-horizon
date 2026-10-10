# lyapunov-horizon

Learned emulators for the 1D Kuramoto–Sivashinsky equation: rollout horizons, long-run statistics and calibration.

The project asks one question: **how far past the chaos-limited prediction horizon can a learned model stay useful?** A model learns to predict the next state from the current one and is then rolled out autoregressively. It is judged on *weather* (how long it tracks the true trajectory, in Lyapunov times) and *climate* (whether it still behaves like the real system after it has diverged). Everything runs on a laptop CPU.

![True trajectory vs. the local-rule baseline](figures/spacetime.png)

```bash
pip install -e ".[dev]"
python scripts/generate_data.py
python scripts/run_baselines.py
```

**See [documentation.md](documentation.md)** for the physics, setup, data format, evaluation protocol, how to add a model, the baseline results, and the roadmap.
