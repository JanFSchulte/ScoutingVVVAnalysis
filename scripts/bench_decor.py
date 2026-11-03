"""Time one stage-2 objective evaluation (decorrelation penalty + gradient) at production size.

Usage: python scripts/bench_decor.py [N_QCD]
"""

import sys
import time

import numpy as np

from vvv.ml.train import DecorState

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
cfg = {"decor_n_bins": 10, "decor_bin_tau_scale": 0.25, "decor_tail_min_prob": 2e-4, "decor_tail_n": 10,
       "decor_n_thresholds": 15, "decor_score_tau": 0.2}
rng = np.random.default_rng(1)
Z = np.column_stack([rng.uniform(0, 250, N), rng.normal(5, 1, N), rng.uniform(0, 1, N)])
w = rng.exponential(1.0, N)
margins = rng.normal(0, 1, (N, 6))
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device", torch.cuda.get_device_name(0) if device == "cuda" else "cpu", flush=True)
ref = None
for dtype in ("float64", "float32"):
  state = DecorState(Z, w, np.ones(N, dtype=bool), dict(cfg, decor_dtype=dtype), device)
  for i in range(2):
    t = time.time()
    raw, g, h = state(margins)
    if device == "cuda":
        torch.cuda.synchronize()
    print(f"{dtype} N={N}: penalty {raw:.6g}, grad+hess in {time.time() - t:.2f} s", flush=True)
  if ref is None:
    ref = (raw, g)
  else:
    print("float32 vs float64: penalty rel diff", abs(raw - ref[0]) / ref[0],
          " grad rel diff (p99)", np.quantile(np.abs(g - ref[1]) / (np.abs(ref[1]).max()), 0.99))
print("peak GPU memory GB", torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0)
