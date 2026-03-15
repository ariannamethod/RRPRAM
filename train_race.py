#!/usr/bin/env python3
"""
3-way race training: RRPRAM vs Haze vs Resonance
PyTorch + CUDA on A100. Character-level transformers.

Usage:
  python3 train_race.py --arch rrpram --data leo_full.txt --depth 12 --steps 10000
  python3 train_race.py --arch haze   --data leo_full.txt --depth 12 --steps 10000
  python3 train_race.py --arch resonance --data leo_full.txt --depth 12 --steps 10000
"""

import argparse
import math
import struct
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Config from depth (matches C code: E = depth * 32) ───

def cfg_from_depth(depth):
    T = 64 if depth >= 8 else 32
    E = depth * 32
    H = 4 if depth >= 4 else 2
    D = E // H
    B = depth
    M = E * 2
    return dict(T=T, E=E, H=H, D=D, B=B, M=M)


VOCAB = 256


# ═══════════════════════════════════════════════════════════
# RRPRAM: Positional resonance attention
# attn[i,j] = x_i · Wr[:,j]  — linear, positional
# ═══════════════════════════════════════════════════════════

class RRPRAMAttention(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        self.wv = nn.Linear(E, H * D, bias=False)
        # Wr: each head has [E, T] positional pattern matrix
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2.0 / E) ** 0.5)
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D

        v = self.wv(x).view(B, T, H, D).transpose(1, 2)  # [B,H,T,D]

        # RRPRAM: attn[b,h,i,j] = x[b,i,:] @ wr[h,:,j]
        # x: [B,T,E], wr: [H,E,T] -> attn: [B,H,T,T]
        attn = torch.einsum('bte,het->bht', x, self.wr[:, :, :T]).unsqueeze(2)
        # broadcast: attn[b,h,i,j] same for all i (positional pattern)
        attn = attn.expand(B, H, T, T).clone()
        attn = attn / (D ** 0.5)
        # CAUSAL MASK — cannot attend to future positions
        causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(causal, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # [B,H,T,D]
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


class RRPRAMBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.ln1 = nn.LayerNorm(E)
        self.attn = RRPRAMAttention(E, H, D, T)
        self.ln2 = nn.LayerNorm(E)
        self.w1 = nn.Linear(E, M)
        self.w2 = nn.Linear(M, E)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = F.gelu(self.w1(self.ln2(x)))
        x = x + self.w2(h)
        return x


class RRPRAM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, T, B = cfg['E'], cfg['T'], cfg['B']
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([
            RRPRAMBlock(E, cfg['H'], cfg['D'], T, cfg['M']) for _ in range(B)
        ])
        self.ln_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.T = T

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# HAZE: Hybrid RRPRAM + Content attention with learned gate
# ═══════════════════════════════════════════════════════════

class HazeAttention(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        # Content path (QKV)
        self.wq = nn.Linear(E, H * D, bias=False)
        self.wk = nn.Linear(E, H * D, bias=False)
        self.wv = nn.Linear(E, H * D, bias=False)
        # RRPRAM path
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2.0 / E) ** 0.5)
        # Learned gate per head
        self.alpha = nn.Parameter(torch.zeros(H))  # sigmoid(0) = 0.5
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)

        # Content attention
        content_attn = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        content_attn = content_attn.masked_fill(mask, float('-inf'))
        content_attn = F.softmax(content_attn, dim=-1)

        # RRPRAM attention
        rrpram_attn = torch.einsum('bte,het->bht', x, self.wr[:, :, :T]).unsqueeze(2)
        rrpram_attn = rrpram_attn.expand(B, H, T, T).clone()
        rrpram_attn = rrpram_attn / (D ** 0.5)
        # CAUSAL MASK — same mask as content path
        rrpram_attn = rrpram_attn.masked_fill(mask, float('-inf'))
        rrpram_attn = F.softmax(rrpram_attn, dim=-1)

        # Hybrid gate
        alpha = torch.sigmoid(self.alpha).view(1, H, 1, 1)
        attn = alpha * rrpram_attn + (1 - alpha) * content_attn

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


class HazeBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.ln1 = nn.LayerNorm(E)
        self.attn = HazeAttention(E, H, D, T)
        self.ln2 = nn.LayerNorm(E)
        self.w1 = nn.Linear(E, M)
        self.w2 = nn.Linear(M, E)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = F.gelu(self.w1(self.ln2(x)))
        x = x + self.w2(h)
        return x


class Haze(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, T, B = cfg['E'], cfg['T'], cfg['B']
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([
            HazeBlock(E, cfg['H'], cfg['D'], T, cfg['M']) for _ in range(B)
        ])
        self.ln_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.T = T

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.head(x)


# ═══════════════════════════════════════════════════════════
# RESONANCE: Dual RRPRAM+Content + SwiGLU + Dario field + Kuramoto
# θ = ε + γ + αδ
# ═══════════════════════════════════════════════════════════

class KuramotoChamber(nn.Module):
    """6 coupled oscillators — Dario field component"""
    def __init__(self, E, n_chambers=6):
        super().__init__()
        self.n = n_chambers
        self.omega = nn.Parameter(torch.randn(n_chambers) * 0.1)
        self.K = nn.Parameter(torch.randn(n_chambers, n_chambers) * 0.01)
        self.proj = nn.Linear(n_chambers, E, bias=False)

    def forward(self, step_phase):
        # step_phase: scalar or [B] — current training phase
        theta = self.omega * step_phase
        # Kuramoto coupling
        for _ in range(3):  # 3 sync iterations
            dtheta = torch.zeros_like(theta)
            for i in range(self.n):
                for j in range(self.n):
                    dtheta[i] += self.K[i, j] * torch.sin(theta[j] - theta[i])
            theta = theta + 0.1 * dtheta
        signal = torch.sin(theta)  # [n_chambers]
        return self.proj(signal)  # [E]


class ResonanceAttention(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D, self.T = H, D, T
        self.wq = nn.Linear(E, H * D, bias=False)
        self.wk = nn.Linear(E, H * D, bias=False)
        self.wv = nn.Linear(E, H * D, bias=False)
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2.0 / E) ** 0.5)
        self.alpha = nn.Parameter(torch.zeros(H))
        self.wo = nn.Linear(H * D, E, bias=False)

    def forward(self, x):
        B, T, E = x.shape
        H, D = self.H, self.D

        q = self.wq(x).view(B, T, H, D).transpose(1, 2)
        k = self.wk(x).view(B, T, H, D).transpose(1, 2)
        v = self.wv(x).view(B, T, H, D).transpose(1, 2)

        content_attn = torch.matmul(q, k.transpose(-2, -1)) / (D ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        content_attn = content_attn.masked_fill(mask, float('-inf'))
        content_attn = F.softmax(content_attn, dim=-1)

        rrpram_attn = torch.einsum('bte,het->bht', x, self.wr[:, :, :T]).unsqueeze(2)
        rrpram_attn = rrpram_attn.expand(B, H, T, T).clone()
        rrpram_attn = rrpram_attn / (D ** 0.5)
        # CAUSAL MASK — same mask as content path
        rrpram_attn = rrpram_attn.masked_fill(mask, float('-inf'))
        rrpram_attn = F.softmax(rrpram_attn, dim=-1)

        alpha = torch.sigmoid(self.alpha).view(1, H, 1, 1)
        attn = alpha * rrpram_attn + (1 - alpha) * content_attn

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.wo(out)


class ResonanceBlock(nn.Module):
    def __init__(self, E, H, D, T, M):
        super().__init__()
        self.rms1 = nn.LayerNorm(E)  # RMSNorm in C, LayerNorm here (close enough)
        self.attn = ResonanceAttention(E, H, D, T)
        self.rms2 = nn.LayerNorm(E)
        # SwiGLU MLP
        self.w_gate = nn.Linear(E, M, bias=False)
        self.w_up = nn.Linear(E, M, bias=False)
        self.w_down = nn.Linear(M, E, bias=False)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        h = self.rms2(x)
        # SwiGLU: down(silu(gate(h)) * up(h))
        x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        return x


class Resonance(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E, T, B = cfg['E'], cfg['T'], cfg['B']
        self.tok_emb = nn.Embedding(VOCAB, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([
            ResonanceBlock(E, cfg['H'], cfg['D'], T, cfg['M']) for _ in range(B)
        ])
        self.ln_f = nn.LayerNorm(E)
        self.head = nn.Linear(E, VOCAB, bias=False)
        self.kuramoto = KuramotoChamber(E)
        self.dario_scale = nn.Parameter(torch.tensor(0.1))
        self.T = T

    def forward(self, idx, step=0):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        # Dario field overlay
        phase = torch.tensor(float(step) * 0.01, device=idx.device)
        field = self.kuramoto(phase)  # [E]
        logits = logits + self.dario_scale * (x @ field.unsqueeze(-1)).squeeze(-1).unsqueeze(-1)
        return logits


# ═══════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════

def load_data(path):
    with open(path, 'rb') as f:
        raw = f.read()
    return torch.tensor(list(raw), dtype=torch.long)


def get_batch(data, T, batch_size, device):
    ix = torch.randint(0, len(data) - T - 1, (batch_size,))
    x = torch.stack([data[i:i+T] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+T+1] for i in ix]).to(device)
    return x, y


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model, cfg, arch, path):
    """Save in C-compatible format: config header + float32 weights.
    Linear weights are transposed: PyTorch stores [out,in], C expects [in,out]."""
    E, T, H, D, B, M = cfg['E'], cfg['T'], cfg['H'], cfg['D'], cfg['B'], cfg['M']
    with open(path, 'wb') as f:
        # Config header (6 ints)
        f.write(struct.pack('6i', T, E, H, D, B, M))
        # All parameters as float32, transpose Linear weights for C compatibility
        for name, p in model.named_parameters():
            data = p.detach().cpu().float()
            # Transpose 2D weights from Linear layers (not Embeddings)
            if data.dim() == 2 and 'emb' not in name:
                data = data.T.contiguous()
            f.write(data.numpy().tobytes())
    print(f"[{arch}] saved {path} ({count_params(model)} params)")


def train(arch, data_path, depth, steps, save_path, lr=3e-4, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = cfg_from_depth(depth)
    T = cfg['T']

    print(f"\n{'='*60}")
    print(f"  {arch.upper()} — depth={depth}")
    print(f"  E={cfg['E']} H={cfg['H']} D={cfg['D']} T={T} B={cfg['B']} M={cfg['M']}")
    print(f"  device={device}, lr={lr}, batch={batch_size}, steps={steps}")
    print(f"{'='*60}")

    # Load data
    data = load_data(data_path)
    print(f"  data: {len(data)} bytes ({len(data)/1024:.1f}KB)")

    # Create model
    if arch == 'rrpram':
        model = RRPRAM(cfg).to(device)
    elif arch == 'haze':
        model = Haze(cfg).to(device)
    elif arch == 'resonance':
        model = Resonance(cfg).to(device)
    else:
        raise ValueError(f"unknown arch: {arch}")

    n_params = count_params(model)
    print(f"  params: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999),
                                   weight_decay=0.01)
    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    model.train()
    t0 = time.time()
    best_loss = float('inf')

    for step in range(1, steps + 1):
        x, y = get_batch(data, T, batch_size, device)

        if arch == 'resonance':
            logits = model(x, step=step)
        else:
            logits = model(x)

        loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % 100 == 0 or step == 1:
            dt = time.time() - t0
            steps_per_sec = step / dt
            current_lr = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}/{steps}  loss={loss.item():.4f}  "
                  f"best={best_loss:.4f}  lr={current_lr:.2e}  "
                  f"{steps_per_sec:.1f} steps/s", flush=True)

        if step % 2000 == 0 and save_path:
            ckpt = save_path.replace('.bin', f'_step{step}.bin')
            save_checkpoint(model, cfg, arch, ckpt)

    # Final save
    if save_path:
        save_checkpoint(model, cfg, arch, save_path)

    dt = time.time() - t0
    print(f"\n  [{arch}] DONE: {steps} steps in {dt:.1f}s ({steps/dt:.1f} steps/s)")
    print(f"  [{arch}] final loss={loss.item():.4f}  best={best_loss:.4f}")

    # Print alpha values for hybrid models
    if arch in ('haze', 'resonance'):
        for i, blk in enumerate(model.blocks):
            a = torch.sigmoid(blk.attn.alpha).detach().cpu()
            print(f"  [{arch}] block {i} alpha (RRPRAM weight): "
                  f"[{', '.join(f'{v:.3f}' for v in a)}]")

    # Generate sample
    print(f"\n  [{arch}] --- sample (temp=0.8) ---")
    model.eval()
    with torch.no_grad():
        # Seed with "Leo " — NO null byte, start directly with real tokens
        seed = b"Leo "
        ctx = torch.tensor([list(seed)], dtype=torch.long, device=device)

        out = list(seed)
        for _ in range(200):
            if ctx.shape[1] > T:
                ctx = ctx[:, -T:]
            if arch == 'resonance':
                logits = model(ctx, step=steps)
            else:
                logits = model(ctx)
            logits = logits[0, -1, :] / 0.8
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            out.append(next_id)
            ctx = torch.cat([ctx, torch.tensor([[next_id]], device=device)], dim=1)

        text = bytes(out).decode('utf-8', errors='replace')
        print(f"  {text}")

    return best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', required=True, choices=['rrpram', 'haze', 'resonance'])
    parser.add_argument('--data', required=True)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--save', default=None)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()

    if args.save is None:
        args.save = f"{args.arch}_v2_d{args.depth}.bin"

    train(args.arch, args.data, args.depth, args.steps, args.save,
          lr=args.lr, batch_size=args.batch)
