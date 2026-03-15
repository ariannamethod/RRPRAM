#!/usr/bin/env python3
"""
BPE training for all architectures on Yent dataset.
Trains BPE tokenizer, then trains models.

Architectures: rrpram, haze, resonance, janus, metajanus, metajanus_rrpram, hybrid
All use BPE vocab instead of char-level 256.

Usage:
  python3 train_bpe.py --arch resonance --data yent_train.txt --steps 15000
  python3 train_bpe.py --arch janus --data yent_train.txt --steps 15000
"""

import argparse, collections, math, struct, time, os, pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# BPE Tokenizer
# ═══════════════════════════════════════════════════════════

class BPETokenizer:
    def __init__(self, vocab_size=2048):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}

    def train(self, data_bytes, n_merges=None):
        if n_merges is None:
            n_merges = self.vocab_size - 256
        tokens = list(data_bytes)
        print(f"[BPE] training {n_merges} merges on {len(tokens)} bytes...")
        t0 = time.time()
        for i in range(n_merges):
            pairs = collections.Counter()
            for j in range(len(tokens) - 1):
                pairs[(tokens[j], tokens[j+1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = 256 + i
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == best[0] and tokens[j+1] == best[1]:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if (i+1) % 200 == 0:
                ratio = len(data_bytes) / len(tokens)
                print(f"  merge {i+1}/{n_merges}  vocab={new_id+1}  "
                      f"tokens={len(tokens)}  ratio={ratio:.2f}x")
        dt = time.time() - t0
        ratio = len(data_bytes) / len(tokens)
        print(f"[BPE] done: {len(self.merges)} merges, {len(tokens)} tokens, "
              f"{ratio:.2f}x compression, {dt:.1f}s")
        return tokens

    def encode(self, data_bytes):
        tokens = list(data_bytes)
        for pair_id, (a, b) in enumerate(self.merges):
            new_id = 256 + pair_id
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and tokens[j] == a and tokens[j+1] == b:
                    new_tokens.append(new_id)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        return tokens

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'merges': self.merges, 'vocab_size': self.vocab_size}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.merges = d['merges']
        self.vocab_size = d['vocab_size']
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]


# ═══════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════

def cfg(depth=12, vocab=2048):
    T = 64 if depth >= 8 else 32
    E = depth * 32
    H = 4 if depth >= 4 else 2
    D = E // H
    B = depth
    M = E * 2
    return dict(T=T, E=E, H=H, D=D, B=B, M=M, V=vocab)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-5) * self.weight


# ═══════════════════════════════════════════════════════════
# All architectures (BPE-compatible, variable vocab)
# ═══════════════════════════════════════════════════════════

class RRPRAMAttn(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D = H, D
        self.wv = nn.Linear(E, H*D, bias=False)
        self.wr = nn.Parameter(torch.randn(H, E, T) * (2/E)**0.5)
        self.wo = nn.Linear(H*D, E, bias=False)
    def forward(self, x):
        B,T,E = x.shape; H,D = self.H,self.D; sc = 1/(D**0.5)
        mask = torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        v = self.wv(x).view(B,T,H,D).transpose(1,2)
        a = torch.einsum('bte,het->bht',x,self.wr[:,:,:T]).unsqueeze(2)
        a = (a.expand(B,H,T,T).clone()*sc).masked_fill(mask,float('-inf'))
        out = torch.matmul(F.softmax(a,dim=-1),v)
        return self.wo(out.transpose(1,2).contiguous().view(B,T,H*D))

class HazeAttn(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D = H, D
        self.wq=nn.Linear(E,H*D,bias=False); self.wk=nn.Linear(E,H*D,bias=False)
        self.wv=nn.Linear(E,H*D,bias=False)
        self.wr=nn.Parameter(torch.randn(H,E,T)*(2/E)**0.5)
        self.alpha=nn.Parameter(torch.zeros(H))
        self.wo=nn.Linear(H*D,E,bias=False)
    def forward(self, x):
        B,T,E=x.shape; H,D=self.H,self.D; sc=1/(D**0.5)
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        q=self.wq(x).view(B,T,H,D).transpose(1,2)
        k=self.wk(x).view(B,T,H,D).transpose(1,2)
        v=self.wv(x).view(B,T,H,D).transpose(1,2)
        ca=(torch.matmul(q,k.transpose(-2,-1))*sc).masked_fill(mask,float('-inf'))
        ca=F.softmax(ca,dim=-1)
        ra=torch.einsum('bte,het->bht',x,self.wr[:,:,:T]).unsqueeze(2)
        ra=(ra.expand(B,H,T,T).clone()*sc).masked_fill(mask,float('-inf'))
        ra=F.softmax(ra,dim=-1)
        alpha=torch.sigmoid(self.alpha).view(1,H,1,1)
        out=torch.matmul(alpha*ra+(1-alpha)*ca,v)
        return self.wo(out.transpose(1,2).contiguous().view(B,T,H*D))

class ResonanceAttn(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D = H, D
        self.wq=nn.Linear(E,H*D,bias=False); self.wk=nn.Linear(E,H*D,bias=False)
        self.wv=nn.Linear(E,H*D,bias=False)
        self.wr=nn.Parameter(torch.randn(H,E,T)*(2/E)**0.5)
        self.alpha=nn.Parameter(torch.zeros(H))
        self.wo=nn.Linear(H*D,E,bias=False)
    def forward(self, x):
        B,T,E=x.shape; H,D=self.H,self.D; sc=1/(D**0.5)
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        q=self.wq(x).view(B,T,H,D).transpose(1,2)
        k=self.wk(x).view(B,T,H,D).transpose(1,2)
        v=self.wv(x).view(B,T,H,D).transpose(1,2)
        ca=(torch.matmul(q,k.transpose(-2,-1))*sc).masked_fill(mask,float('-inf'))
        ra=torch.einsum('bte,het->bht',x,self.wr[:,:,:T]).unsqueeze(2)
        ra=(ra.expand(B,H,T,T).clone()*sc).masked_fill(mask,float('-inf'))
        alpha=torch.sigmoid(self.alpha).view(1,H,1,1)
        out=torch.matmul(alpha*F.softmax(ra,dim=-1)+(1-alpha)*F.softmax(ca,dim=-1),v)
        return self.wo(out.transpose(1,2).contiguous().view(B,T,H*D))

class JanusAttn(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D = H, D
        self.wq=nn.Linear(E,H*D,bias=False); self.wk=nn.Linear(E,H*D,bias=False)
        self.wv=nn.Linear(E,H*D,bias=False)
        self.wr=nn.Parameter(torch.randn(H,E,T)*(2/E)**0.5)
        self.wvr=nn.Linear(E,H*D,bias=False)
        self.wj=nn.Linear(E,E,bias=False)
        self.gate=nn.Parameter(torch.zeros(H,3))
        self.wo=nn.Linear(H*D,E,bias=False)
    def forward(self, x):
        B,T,E=x.shape; H,D=self.H,self.D; sc=1/(D**0.5)
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        q=self.wq(x).view(B,T,H,D).transpose(1,2)
        k=self.wk(x).view(B,T,H,D).transpose(1,2)
        v=self.wv(x).view(B,T,H,D).transpose(1,2)
        qkv_a=(torch.matmul(q,k.transpose(-2,-1))*sc).masked_fill(mask,float('-inf'))
        qkv_out=torch.matmul(F.softmax(qkv_a,dim=-1),v)
        rv=self.wvr(x).view(B,T,H,D).transpose(1,2)
        ra=torch.einsum('bte,het->bht',x,self.wr[:,:,:T]).unsqueeze(2)
        ra=(ra.expand(B,H,T,T).clone()*sc).masked_fill(mask,float('-inf'))
        rrp_out=torch.matmul(F.softmax(ra,dim=-1),rv)
        echo=self.wj(x); eb=F.linear(echo,self.wj.weight.T)
        sc2=(x*eb).sum(-1)/(E**0.5)
        ja=(sc2.unsqueeze(-1)*sc2.unsqueeze(-2)).masked_fill(mask,float('-inf'))
        ja=F.softmax(ja,dim=-1).unsqueeze(1).expand(B,H,T,T)
        jan_out=torch.matmul(ja,echo.view(B,T,H,D).transpose(1,2))
        g=F.softmax(self.gate,dim=-1)
        out=(g[:,0].view(1,H,1,1)*qkv_out+g[:,1].view(1,H,1,1)*rrp_out+g[:,2].view(1,H,1,1)*jan_out)
        return self.wo(out.transpose(1,2).contiguous().view(B,T,H*D))

class MetaJanusAttn(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D = H, D
        self.wj=nn.Linear(E,E,bias=False); self.wj_v=nn.Linear(E,E,bias=False)
        self.wo=nn.Linear(H*D,E,bias=False)
    def forward(self, x):
        B,T,E=x.shape; H,D=self.H,self.D
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        echo=self.wj(x); eb=F.linear(echo,self.wj.weight.T)
        sc=(x*eb).sum(-1)/(E**0.5)
        ja=(sc.unsqueeze(-1)*sc.unsqueeze(-2)).masked_fill(mask,float('-inf'))
        ja=F.softmax(ja,dim=-1).unsqueeze(1).expand(B,H,T,T)
        val=self.wj_v(x).view(B,T,H,D).transpose(1,2)
        out=torch.matmul(ja,val)
        return self.wo(out.transpose(1,2).contiguous().view(B,T,H*D))

class MetaJanusRRPRAMAttn(nn.Module):
    def __init__(self, E, H, D, T):
        super().__init__()
        self.H, self.D = H, D
        self.wr=nn.Parameter(torch.randn(H,E,T)*(2/E)**0.5)
        self.wvr=nn.Linear(E,H*D,bias=False)
        self.wj=nn.Linear(E,E,bias=False)
        self.gate=nn.Parameter(torch.zeros(H,2))
        self.wo=nn.Linear(H*D,E,bias=False)
    def forward(self, x):
        B,T,E=x.shape; H,D=self.H,self.D; sc=1/(D**0.5)
        mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        rv=self.wvr(x).view(B,T,H,D).transpose(1,2)
        ra=torch.einsum('bte,het->bht',x,self.wr[:,:,:T]).unsqueeze(2)
        ra=(ra.expand(B,H,T,T).clone()*sc).masked_fill(mask,float('-inf'))
        rrp_out=torch.matmul(F.softmax(ra,dim=-1),rv)
        echo=self.wj(x); eb=F.linear(echo,self.wj.weight.T)
        sc2=(x*eb).sum(-1)/(E**0.5)
        ja=(sc2.unsqueeze(-1)*sc2.unsqueeze(-2)).masked_fill(mask,float('-inf'))
        ja=F.softmax(ja,dim=-1).unsqueeze(1).expand(B,H,T,T)
        jan_out=torch.matmul(ja,echo.view(B,T,H,D).transpose(1,2))
        g=F.softmax(self.gate,dim=-1)
        out=g[:,0].view(1,H,1,1)*rrp_out+g[:,1].view(1,H,1,1)*jan_out
        return self.wo(out.transpose(1,2).contiguous().view(B,T,H*D))


# ═══════════════════════════════════════════════════════════
# Block + Model wrappers
# ═══════════════════════════════════════════════════════════

ATTN_MAP = {
    'rrpram': RRPRAMAttn, 'haze': HazeAttn, 'resonance': ResonanceAttn,
    'janus': JanusAttn, 'metajanus': MetaJanusAttn, 'metajanus_rrpram': MetaJanusRRPRAMAttn,
}

class Block(nn.Module):
    def __init__(self, attn_cls, E, H, D, T, M, use_swiglu=True):
        super().__init__()
        self.rms1 = RMSNorm(E)
        self.attn = attn_cls(E, H, D, T)
        self.rms2 = RMSNorm(E)
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.w_gate = nn.Linear(E, M, bias=False)
            self.w_up = nn.Linear(E, M, bias=False)
            self.w_down = nn.Linear(M, E, bias=False)
        else:
            self.w1 = nn.Linear(E, M)
            self.w2 = nn.Linear(M, E)
    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        h = self.rms2(x)
        if self.use_swiglu:
            x = x + self.w_down(F.silu(self.w_gate(h)) * self.w_up(h))
        else:
            x = x + self.w2(F.gelu(self.w1(h)))
        return x

class Model(nn.Module):
    def __init__(self, arch, c):
        super().__init__()
        E, T, B, V = c['E'], c['T'], c['B'], c['V']
        attn_cls = ATTN_MAP[arch]
        use_swiglu = arch in ('resonance', 'janus', 'metajanus', 'metajanus_rrpram')
        self.tok_emb = nn.Embedding(V, E)
        self.pos_emb = nn.Embedding(T, E)
        self.blocks = nn.ModuleList([
            Block(attn_cls, E, c['H'], c['D'], T, c['M'], use_swiglu) for _ in range(B)
        ])
        self.rms_f = RMSNorm(E)
        self.head = nn.Linear(E, V, bias=False)
        self.T = T

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.rms_f(x))


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train(arch, data_path, depth, steps, save_path, lr, batch_size, bpe_vocab):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = cfg(depth, bpe_vocab)
    V = c['V']

    # Load and tokenize data
    with open(data_path, 'rb') as f:
        raw = f.read()

    bpe_path = data_path + f'.bpe{bpe_vocab}.pkl'
    tok_path = data_path + f'.bpe{bpe_vocab}.tokens.pt'

    if os.path.exists(bpe_path) and os.path.exists(tok_path):
        print(f"[BPE] loading cached tokenizer from {bpe_path}")
        bpe = BPETokenizer(bpe_vocab)
        bpe.load(bpe_path)
        tokens = torch.load(tok_path)
    else:
        bpe = BPETokenizer(bpe_vocab)
        tok_list = bpe.train(raw, bpe_vocab - 256)
        bpe.save(bpe_path)
        tokens = torch.tensor(tok_list, dtype=torch.long)
        torch.save(tokens, tok_path)

    print(f"[data] {len(raw)} bytes -> {len(tokens)} BPE tokens "
          f"({len(raw)/len(tokens):.2f}x compression)")

    model = Model(arch, c).to(device)
    T = c['T']
    n = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  {arch.upper()} BPE — depth={depth}, vocab={V}")
    print(f"  E={c['E']} H={c['H']} D={c['D']} T={T} B={c['B']} M={c['M']}")
    print(f"  params: {n:,} ({n/1e6:.2f}M)")
    print(f"  device={device}, lr={lr}, batch={batch_size}, steps={steps}")
    print(f"{'='*60}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    model.train()
    t0 = time.time()
    best = float('inf')

    for step in range(1, steps + 1):
        ix = torch.randint(0, len(tokens)-T-1, (batch_size,))
        x = torch.stack([tokens[i:i+T] for i in ix]).to(device)
        y = torch.stack([tokens[i+1:i+T+1] for i in ix]).to(device)

        loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
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
            torch.save(model.state_dict(), ckpt)
            print(f"  saved {ckpt}")

    if save_path:
        torch.save(model.state_dict(), save_path)
        print(f"  saved {save_path} ({n:,} params)")

    dt = time.time() - t0
    print(f"\n  [{arch}] DONE: {steps} steps in {dt:.1f}s ({steps/dt:.1f} steps/s)")
    print(f"  [{arch}] final loss={loss.item():.4f}  best={best:.4f}")

    # Generate
    print(f"\n  [{arch}] --- sample (temp=0.8) ---")
    model.eval()
    seed = "Q: who are you\nA: "
    seed_tokens = bpe.encode(seed.encode())
    ctx = torch.tensor([seed_tokens], dtype=torch.long, device=device)
    out_tokens = list(seed_tokens)
    with torch.no_grad():
        for _ in range(200):
            if ctx.shape[1] > T:
                ctx = ctx[:, -T:]
            logits = model(ctx)[0, -1, :] / 0.8
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
            out_tokens.append(nxt)
            ctx = torch.cat([ctx, torch.tensor([[nxt]], device=device)], dim=1)
    # Decode
    text = b''.join(bpe.vocab.get(t, b'?') for t in out_tokens).decode('utf-8', errors='replace')
    print(f"  {text}")

    return best


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arch', required=True,
                   choices=['rrpram','haze','resonance','janus','metajanus','metajanus_rrpram'])
    p.add_argument('--data', required=True)
    p.add_argument('--depth', type=int, default=12)
    p.add_argument('--steps', type=int, default=15000)
    p.add_argument('--save', default=None)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--bpe-vocab', type=int, default=2048)
    a = p.parse_args()
    if a.save is None:
        a.save = f"{a.arch}_bpe_d{a.depth}.bin"
    train(a.arch, a.data, a.depth, a.steps, a.save, a.lr, a.batch, a.bpe_vocab)
