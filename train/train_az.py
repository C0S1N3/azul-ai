"""AlphaZero policy+value two-headed net training (plan stage 5).

Architecture (must match crates/azul-az/src/net.rs exactly):
  trunk  : 171 -> 256 -> 256, ReLU between layers
  policy : 256 -> 180 logits (log_softmax applied to the logits at training time)
  value  : 256 -> 1, tanh

Loss:
  policy = cross-entropy against the root visit distribution (soft target,
           equivalent to KL divergence + a constant)
  value  = MSE (target in {-1, 0, 1}, from the side to move's perspective)
  total loss = policy_loss + value_loss (equal weights).

Optimisation: AdamW + OneCycleLR.

Input data: binary records produced by `azul az-selfplay` (little-endian, one
record per position):
  [171 f32 obs][180 f32 policy target (root visit distribution)][1 f32 value target]

Export format (exactly aligned with PvNet::from_bytes in net.rs, little-endian):
  [u32 n_dims = 2][u32 171][u32 256][u32 256][u32 180][u32 1]
  trunk layer 0 : [f32 x (256*171) weights][f32 x 256 biases]
  trunk layer 1 : [f32 x (256*256) weights][f32 x 256 biases]
  policy head   : [f32 x (180*256) weights][f32 x 180 biases]
  value head    : [f32 x (1*256) weights][f32 x 1 biases]
Weights are all row-major (out, in), i.e. PyTorch nn.Linear.weight written out
as-is in C order.

Run:
  python \
      train\\train_az.py out\\az_smoke.bin out\\az_smoke_net.bin --epochs 1 --batch 64
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

FEATURE_DIM = 171   # 与 azul_nnue::features::FEATURE_DIM 一致
ACTION_DIM = 180    # 与 azul_az::action::ACTION_DIM 一致
TRUNK_HIDDEN = 256  # 与 azul_az::net::TRUNK_HIDDEN 一致
VALUE_OUT = 1
RECORD = FEATURE_DIM + ACTION_DIM + VALUE_OUT


def load_dataset(path):
    """Load [171 f32 obs][180 f32 policy][1 f32 value] records."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % RECORD != 0:
        raise ValueError(f"{path}: size {raw.size} not a multiple of {RECORD}")
    raw = raw.reshape(-1, RECORD)
    obs = raw[:, :FEATURE_DIM].copy()
    policy = raw[:, FEATURE_DIM : FEATURE_DIM + ACTION_DIM].copy()
    value = raw[:, FEATURE_DIM + ACTION_DIM].copy()
    return obs, policy, value


class AzNet(nn.Module):
    """trunk(171-256-256 ReLU) + policy(180 logits) + value(tanh)."""

    def __init__(self, hidden=TRUNK_HIDDEN):
        super().__init__()
        self.trunk_dims = [FEATURE_DIM, hidden, hidden]
        self.trunk = nn.ModuleList(
            nn.Linear(self.trunk_dims[i], self.trunk_dims[i + 1]) for i in range(2))
        self.policy_head = nn.Linear(hidden, ACTION_DIM)
        self.value_head = nn.Linear(hidden, VALUE_OUT)

    def forward(self, x):
        for layer in self.trunk:
            x = torch.relu(layer(x))
        return self.policy_head(x), torch.tanh(self.value_head(x))


def export_rust(net, path):
    """Write weights in azul_az::net::PvNet::from_bytes format (strictly aligned)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 2))  # n_dims
        for d in net.trunk_dims:
            f.write(struct.pack("<I", d))
        f.write(struct.pack("<I", ACTION_DIM))  # policy_out
        f.write(struct.pack("<I", VALUE_OUT))   # value_out
        layers = list(net.trunk) + [net.policy_head, net.value_head]
        for layer in layers:
            W = layer.weight.detach().cpu().numpy().astype(np.float32)  # (out, in)
            b = layer.bias.detach().cpu().numpy().astype(np.float32)    # (out,)
            # Rust: w[j*fan_in + i] == W[j, i] — row-major (out, in) 原样写出。
            f.write(W.tobytes(order="C"))
            f.write(b.tobytes(order="C"))
    print(f"exported rust weights -> {path}")


def train(train_path, out_path, epochs=20, batch=256, lr=1e-3, wd=1e-4, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    X, P, V = load_dataset(train_path)
    print(f"train: {X.shape[0]} samples")

    Xt = torch.from_numpy(X).to(device)
    Pt = torch.from_numpy(P).to(device)
    Vt = torch.from_numpy(V).to(device).unsqueeze(1)

    net = AzNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n = Xt.shape[0]
    steps = max(1, (n + batch - 1) // batch) * epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)

    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n, device=device)
        total_p, total_v, batches = 0.0, 0.0, 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, pb, vb = Xt[idx], Pt[idx], Vt[idx]
            logits, vpred = net(xb)
            # policy: 对根访问分布的交叉熵（soft target CE = KL + const）。
            logp = torch.log_softmax(logits, dim=1)
            ploss = -(pb * logp).sum(dim=1).mean()
            vloss = nn.functional.mse_loss(vpred, vb)
            loss = ploss + vloss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_p += ploss.item()
            total_v += vloss.item()
            batches += 1
        print(
            f"epoch {epoch+1:2d}/{epochs}  policy {total_p/max(1,batches):.4f}  "
            f"value {total_v/max(1,batches):.4f}  lr {sched.get_last_lr()[0]:.2e}"
        )

    export_rust(net, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_bin")
    ap.add_argument("out", nargs="?", default="out/az_net.bin")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    args = ap.parse_args()
    train(args.train_bin, args.out, args.epochs, args.batch, args.lr, args.wd)


if __name__ == "__main__":
    main()
