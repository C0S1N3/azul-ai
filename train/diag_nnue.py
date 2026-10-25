"""NNUE training-pipeline diagnostics (worklog 2026-10-14).

Question: is the 617k outcome-label dataset unlearnable (data problem) or is
the training pipeline broken (bug)? "mae stuck at constant baseline 0.385" is
suspicious — even noisy outcome labels should let a net beat the prior.

Checks:
  1. label distribution + honest constant-predictor baseline (BCE & MAE on val)
  2. feature health: NaN/Inf, constant features, std percentiles
  3. score_diff (feat 154) correlation with label — sanity of alignment
  4. linear probe (logistic regression): does a LINEAR model beat baseline
     out-of-sample?  (if yes, data is informative; MLP should beat it too)
  5. tiny-overfit: can the MLP drive train loss -> 0 on 4096 samples?
     (if no, the pipeline itself is broken)

Run:
  python \
      train\\diag_nnue.py data\\train_outcome_big.bin
"""

import sys

import numpy as np
import torch
import torch.nn as nn

FEATURE_DIM = 170


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/train_outcome_big.bin"
    raw = np.fromfile(path, dtype=np.float32)
    rec = FEATURE_DIM + 1
    assert raw.size % rec == 0, f"{path}: {raw.size} not multiple of {rec}"
    raw = raw.reshape(-1, rec)
    X, y = raw[:, :FEATURE_DIM], raw[:, FEATURE_DIM]
    n = len(y)
    print(f"samples: {n}  from {path}")

    # split by position in file == split by games (written sequentially)
    k = int(n * 0.9)
    Xtr, ytr, Xva, yva = X[:k], y[:k], X[k:], y[k:]

    # 1. labels + honest baseline (train-mean predictor evaluated on val)
    vals, counts = np.unique(y, return_counts=True)
    print("label dist:", {float(v): int(c) for v, c in zip(vals, counts)})
    p = ytr.mean()
    base_bce = float(-(yva * np.log(p) + (1 - yva) * np.log(1 - p)).mean())
    base_mae = float(np.abs(yva - p).mean())
    print(f"baseline(val): BCE {base_bce:.4f}  MAE {base_mae:.4f}  (train-mean {p:.4f})")

    # 2. feature health
    print("NaN:", int(np.isnan(X).sum()), " Inf:", int(np.isinf(X).sum()))
    std = X.std(axis=0)
    print(f"constant features (std<1e-6): {int((std < 1e-6).sum())}/{FEATURE_DIM}")
    print("std percentiles:", np.percentile(std, [0, 25, 50, 75, 100]).round(3).tolist())

    # 3. alignment sanity: score_diff (feat 154) vs label
    sd = X[:, 154]
    print(f"feat154 score_diff: std {float(sd.std()):.4f}  corr(label) {float(np.corrcoef(sd, y)[0, 1]):.4f}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).to(dev).unsqueeze(1)
    Xv = torch.from_numpy(Xva).to(dev)
    yv = torch.from_numpy(yva).to(dev).unsqueeze(1)
    lossf = nn.BCELoss()

    # 4. linear probe (logistic regression) — out-of-sample check
    torch.manual_seed(0)
    lin = nn.Linear(FEATURE_DIM, 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    ntr = Xt.shape[0]
    for ep in range(30):
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, 8192):
            idx = perm[i : i + 8192]
            loss = lossf(torch.sigmoid(lin(Xt[idx])), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 10 == 9:
            lin.eval()
            with torch.no_grad():
                pv = torch.sigmoid(lin(Xv))
                vl = lossf(pv, yv).item()
                mae = (pv - yv).abs().mean().item()
            lin.train()
            print(f"linear ep{ep + 1}: val BCE {vl:.4f}  val MAE {mae:.4f}  (baseline {base_bce:.4f}/{base_mae:.4f})")

    # 5. tiny-overfit: pipeline must be able to memorize 4096 samples
    m = 4096
    Xs, ys = Xt[:m], yt[:m]
    mlp = nn.Sequential(
        nn.Linear(FEATURE_DIM, 128), nn.ReLU(),
        nn.Linear(128, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 1), nn.Sigmoid(),
    ).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    for step in range(1, 3001):
        loss = lossf(mlp(Xs), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 600 == 0:
            print(f"overfit step{step}: train BCE {loss.item():.4f}")


if __name__ == "__main__":
    main()
