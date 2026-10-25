"""NNUE value-net training (plan stage 2.4 — Rzepecki recipe, default).

Recipe (Rzepecki §5.3.3 measurements, user decision: the paper's measured
results take priority):
  - fully-connected value net: 64-128-16-16-1 (dim must match Rust FEATURE_DIM)
  - hidden ReLU, output sigmoid, trained with BCE against {loss=0, draw=0.5, win=1}
  - AdamW, lr 0.001 -> decay, ~20 epochs, weight_decay 1e-4
  - batch 1024-4096

Input data: binary produced by `azul gen-data` — records of
  [170 f32 features][1 f32 target]  (target in {0.0, 0.5, 1.0}).

Output: a weights file loadable by azul_nnue::net::Mlp::from_bytes (Rust).
Format: [u32 n_dims][u32 dim0..dimN-1] then per layer: [f32 weights (out*in row-major)][f32 biases].

Run:
  python \
      train\\train_nnue.py data\\train.bin data\\val.bin out\\nnue.bin --epochs 20
"""

import argparse
import struct
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("ERROR: torch not found. Use a Python 3.13 env (has torch 2.9.1+cu128).", file=sys.stderr)
    raise

FEATURE_DIM = 171  # v2: index 170 = settle_diff (h0 simulated-settlement diff)


def load_dataset(path):
    """Load [FEATURE_DIM f32 + 1 f32] records into (X, y) float32 arrays."""
    raw = np.fromfile(path, dtype=np.float32)
    rec = FEATURE_DIM + 1
    if raw.size % rec != 0:
        raise ValueError(f"{path}: size {raw.size} not a multiple of {rec}")
    raw = raw.reshape(-1, rec)
    return raw[:, :FEATURE_DIM].copy(), raw[:, FEATURE_DIM].copy()


class NnueNet(nn.Module):
    def __init__(self, dims=(FEATURE_DIM, 128, 16, 16, 1)):
        super().__init__()
        assert dims[0] == FEATURE_DIM and dims[-1] == 1
        self.dims = list(dims)
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)          # ReLU — NOT ClippedReLU (R-5.3.3)
            else:
                x = torch.sigmoid(x)       # BCE + sigmoid (Rzepecki winning recipe)
        return x


def export_rust(net, path):
    """Write weights in azul_nnue::net::Mlp::from_bytes format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(net.dims)))
        for d in net.dims:
            f.write(struct.pack("<I", d))
        for layer in net.layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)  # (out, in)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)    # (out,)
            # Rust layout: w[j*fan_in + i] == W[j, i] — row-major (out, in) matches.
            f.write(W.tobytes(order="C"))
            f.write(b.tobytes(order="C"))
    print(f"exported rust weights -> {path}")


def train(train_path, val_path, out_path, epochs=20, batch=2048, lr=1e-3, wd=1e-4, device=None, val_frac=0.0, hidden=(128, 16, 16)):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    Xtr, ytr = load_dataset(train_path)
    print(f"train: {Xtr.shape[0]} samples")
    Xva, yva = (None, None)
    if val_path:
        Xva, yva = load_dataset(val_path)
        print(f"val:   {Xva.shape[0]} samples")
    elif val_frac > 0.0:
        # internal split: file is written game-by-game, so a tail slice is a
        # fair held-out set of whole games
        k = int(Xtr.shape[0] * (1.0 - val_frac))
        Xva, yva = Xtr[k:], ytr[k:]
        Xtr, ytr = Xtr[:k], ytr[:k]
        print(f"val:   {Xva.shape[0]} samples (internal tail {val_frac:.0%})")
        pm = float(ytr.mean())
        base_bce = float(-(yva * np.log(pm) + (1 - yva) * np.log(1 - pm)).mean())
        base_mae = float(np.abs(yva - pm).mean())
        print(f"val baseline: BCE {base_bce:.4f}  MAE {base_mae:.4f}")

    Xt = torch.from_numpy(Xtr).to(device)
    yt = torch.from_numpy(ytr).to(device).unsqueeze(1)

    net = NnueNet((FEATURE_DIM, *hidden, 1)).to(device)
    print(f"net: {net.dims}")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    # Rzepecki: LR step decay (gamma 0.1-0.3 every few epochs).
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, epochs // 3), gamma=0.3)
    loss_fn = nn.BCELoss()  # targets already in {0, 0.5, 1}

    n = Xt.shape[0]
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        total_loss, batches = 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, yb = Xt[idx], yt[idx]
            pred = net(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            batches += 1
        sched.step()
        msg = f"epoch {epoch+1:2d}/{epochs}  loss {total_loss/max(1,batches):.4f}  lr {sched.get_last_lr()[0]:.2e}"
        if Xva is not None:
            net.eval()
            with torch.no_grad():
                Xv = torch.from_numpy(Xva).to(device)
                yv = torch.from_numpy(yva).to(device).unsqueeze(1)
                pv = net(Xv)
                vloss = loss_fn(pv, yv).item()
                # calibration: predicted vs actual
                mae = (pv - yv).abs().mean().item()
                msg += f"  val_loss {vloss:.4f}  mae {mae:.4f}"
        print(msg)

    export_rust(net, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_bin")
    ap.add_argument("val_bin", nargs="?", default=None)
    ap.add_argument("out", nargs="?", default="out/nnue.bin")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--out", dest="out_opt", default=None,
                    help="output weights path (overrides positional out)")
    ap.add_argument("--dims", default="128,16,16",
                    help="hidden layer sizes, comma-separated (default 128,16,16)")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="internal held-out tail fraction (ignored if val_bin given)")
    args = ap.parse_args()
    out = args.out_opt or args.out
    hidden = tuple(int(x) for x in args.dims.split(",") if x)
    train(args.train_bin, args.val_bin, out, args.epochs, args.batch, args.lr, args.wd,
          val_frac=args.val_frac, hidden=hidden)


if __name__ == "__main__":
    main()
