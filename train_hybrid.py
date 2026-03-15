#!/usr/bin/env python3
"""
Resonance+Janus hybrid — the strongest architecture.
4-way gated attention: QKV + RRPRAM + Janus self-resonance + Kuramoto field.
12 blocks, E=384. Char-level. PyTorch + CUDA.

Usage:
  python3 train_hybrid.py --data leo_train.txt --steps 15000
"""

import argparse
import math
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = 256


def cfg(depth=12):
    T = 64 if depth >= 8 else 32
    E = depth * 32
    H = 4 if depth >= 4 else 2
    D = E // H
    B = depth
    M = E * 2
    return dict(T=T, E=E, H=H, D=D, B=B, M=M)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class KuramotoChamber(nn.Module):
    """6 coupled oscillators — Dario field component"""
    def __init__(self, E, n=6):
        super().__init__()
        self.n = n
        self.omega = nn.Parameter(torch.randn(n) * 0.1)
        self.K = nn.Parameter(torch.randn(n, n) * 0.01)
        self.proj = nn.Linear(n, E, bias=False)

    def forward(self, phase):
        theta = self.omega * phase
        for _ in range(3):
            dtheta = torch.zeros_like(theta)
            for i in range(self.n):
                for j in range(self.n):
                    dtheta[i] += self.K[i, j] * torch.sin(theta[j] - theta[i])
            theta = theta + 0.1 * dtheta
        return self.proj(torch.sin(theta))


class HybridAttention(nn.Module):
    """4-way: QKV + RRPRAM + Janus self-resonance, learned gate per head"""
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        # QKV
        self.wq = nn.Linear(E, H * D, bias=False)
        self.wk = nn.Linear(E, H * D, bias=False)
        self.wv = nn.Linear(E, H * D, bias=False)
        # RRPRAM (with own value projection)
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2.0 / E) ** 0.5)
        self.wvr = nn.Linear(E, H * D, bias=False)
        # Janus self-resonance
        self.wj = nn.Linear(E, E, bias=False)
        # 3-way gate per head
        self.gate = nn.Parameter(torch.zeros(H, 3))
        # Output
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D
        scale = 1.0 / (D ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # === QKV ===
        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)
        qkv_attn = (torch.matmul(q, k.transpose(-2, -1)) * scale).masked_fill(mask, float('-inf'))
        qkv_out = torch.matmul(F.softmax(qkv_attn, dim=-1), v)

        # === RRPRAM ===
        rv = self.wvr(x).view(B, T, H, D).transpose(1, 2)
        r_attn = torch.einsum('bte,het->bht', x, self.wr[:, :, :T]).unsqueeze(2)
        r_attn = (r_attn.expand(B, H, T, T).clone() * scale).masked_fill(mask, float('-inf'))
        rrp_out = torch.matmul(F.softmax(r_attn, dim=-1), rv)

        # === Janus self-resonance ===
        echo = self.wj(x)  # [B,T,E]
        echo_back = F.linear(echo, self.wj.weight.T)  # [B,T,E]
        scores = (x * echo_back).sum(dim=-1) / (E ** 0.5)  # [B,T]
        j_attn = (scores.unsqueeze(-1) * scores.unsqueeze(-2)).masked_fill(mask, float('-inf'))
        j_attn = F.softmax(j_attn, dim=-1).unsqueeze(1).expand(B, H, T, T)
        j_val = echo.view(B, T, H, D).transpose(1, 2)
        jan_out = torch.matmul(j_attn, j_val)

        # === 3-way gate ===
        g = F.softmax(self.gate, dim=-1)  # [H, 3]
        out = (g[:, 0].view(1, H, 1, 1) * qkv_out +
               g[:, 1].view(1, H, 1, 1) * rrp_out +
               g[:, 2].view(1, H, 1, 1) * jan_out)

        return self.wo(out.transpose(1, 2).contiguous().view(B, T, H * D))


class HybridBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.rms1 = RMSNorm(E)
        self.attn = HybridAttention(E, H, D, T)
        self.rms2 = RMSNorm(E)
        self.w_gate = nn.Linear(E, M, bias=False)
        self.w_up = nn.Linear(E, M, bias=False)
        self.w_down = nn.Linear(M, E, bias=False)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        h = self.rms2(x)
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class ResonanceJanus(nn.Module):
    """θ = ε + γ + αδ — Resonance+Janus hybrid"""
    def __init__(self, c):
        super().__init__()
        E, T, B = c['E'], c['T'], c['B']
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([
            HybridBlock(E, c['H'], c['D'], T, c['M']) for _ in range(B)
        ])
        self.rms_f = RMSNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.kuramoto = KuramotoChamber(E)
        self.dario_scale = nn.Parameter(torch.tensor(0.1))
        self.T = T

    def forward(self, idx, step=0):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.rms_f(x)
        logits = self.head(x)
        # Dario field overlay
        phase = torch.tensor(float(step) * 0.01, device=idx.device)
        field = self.kuramoto(phase)
        logits = logits + self.dario_scale * (x @ field.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
        return logits


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def load_data(path):
    with open(path, 'rb') as f:
        return torch.tensor(list(f.read()), dtype=torch.long)


def get_batch(data, T, bs, device):
    ix = torch.randint(0, len(data) - T - 1, (bs,))
    x = torch.stack([data[i:i+T] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def train(data_path, depth, steps, save_path, lr=3e-4, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = cfg(depth)
    model = ResonanceJanus(c).to(device)
    T = c['T']
    n = count_params(model)

    print(f"\n{'='*60}")
    print(f"  RESONANCE+JANUS HYBRID — depth={depth}")
    print(f"  E={c['E']} H={c['H']} D={c['D']} T={T} B={c['B']} M={c['M']}")
    print(f"  params: {n:,} ({n/1e6:.2f}M)")
    print(f"  device={device}, lr={lr}, batch={batch_size}, steps={steps}")
    print(f"{'='*60}")

    data = load_data(data_path)
    print(f"  data: {len(data)} bytes ({len(data)/1024:.1f}KB)")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    t0 = time.time()
    best = float('inf')

    for step in range(1, steps + 1):
        x, y = get_batch(data, T, batch_size, device)
        logits = model(x, step=step)
        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if loss.item() < best:
            best = loss.item()

        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            print(f"  step {step:5d}/{steps}  loss={loss.item():.4f}  "
                  f"best={best:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"{step/dt:.1f} steps/s", flush=True)

        if step % 2000 == 0 and save_path:
            ckpt = save_path.replace('.bin', f'_step{step}.bin')
            with open(ckpt, 'wb') as f:
                f.write(struct.pack('i', n))
                for name, p in model.named_parameters():
                    d = p.detach().cpu().float()
                    if d.dim() == 2 and 'emb' not in name:
                        d = d.T.contiguous()
                    f.write(d.numpy().tobytes())
            print(f"  saved {ckpt}")

    # Save final
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(struct.pack('i', n))
            for name, p in model.named_parameters():
                d = p.detach().cpu().float()
                if d.dim() == 2 and 'emb' not in name:
                    d = d.T.contiguous()
                f.write(d.numpy().tobytes())
        print(f"  saved {save_path} ({n:,} params)")

    dt = time.time() - t0
    print(f"\n  DONE: {steps} steps in {dt:.1f}s ({steps/dt:.1f} steps/s)")
    print(f"  final loss={loss.item():.4f}  best={best:.4f}")

    # Gate analysis
    for i, blk in enumerate(model.blocks):
        g = F.softmax(blk.attn.gate, dim=-1).detach().cpu()
        qkv = [f'{v:.3f}' for v in g[:, 0]]
        rrp = [f'{v:.3f}' for v in g[:, 1]]
        jan = [f'{v:.3f}' for v in g[:, 2]]
        print(f"  block {i}: QKV=[{','.join(qkv)}] RRPRAM=[{','.join(rrp)}] Janus=[{','.join(jan)}]")

    # Generate
    print(f"\n  --- sample (temp=0.8) ---")
    model.eval()
    with torch.no_grad():
        seed = b"Q: who are you\nA: "
        ctx = torch.tensor([list(seed)], dtype=torch.long, device=device)
        out = list(seed)
        for _ in range(300):
            if ctx.shape[1] > T:
                ctx = ctx[:, -T:]
            logits = model(ctx, step=steps)
            logits = logits[0, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            out.append(nxt)
            if nxt == 10 and len(out) > len(seed) + 50:
                break
            ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)
        print(f"  {bytes(out).decode('utf-8', errors='replace')}")

    return best


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--save', default='resonance_janus_hybrid.bin')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch', type=int, default=32)
    a = p.parse_args()
    train(a.data, a.depth, a.steps, a.save, a.lr, a.batch)
