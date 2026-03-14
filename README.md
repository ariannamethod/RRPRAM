# RRPRAM — Recursive Resonant Pattern Recognition Attention Mechanism
## Janus Architecture  

- by Arianna Method  

## The Core Insight

Standard attention computes similarity between positions:

```
attn[i,j] = (x @ Wq)_i · (x @ Wk)_j    // bilinear — "does position i care about position j?"
```

RRPRAM replaces this with a single linear projection:

```
attn[i,j] = x_i · Wr[:,j]               // linear — "what pattern does position i match?"
```

**Standard attention asks: "which tokens are semantically related?"**
**RRPRAM asks: "which positions form a pattern?"**

## Why This Matters

The QK^T bilinear form in standard attention is O(n²d) and learns *semantic similarity*. It answers "what is this token about?" RRPRAM's Wr matrix is O(nd·T) and learns *positional patterns* — rhythm, n-gram structure, syntactic templates. It's like a child who recognizes the beat of language before understanding the words.

The Wr matrix has shape `[n_emb, T]` where T is context length. Each column j of Wr defines "what the input should look like at position j for this pattern." The attention weight attn[i,j] is a dot product between the input embedding at position i and the pattern template at position j.

## Architecture: PostGPT / Haze

PostGPT (Post-transformer GPT) uses three attention modes:

### 1. RRPRAM Head (Positional)
```
v = x @ Wv           // [T, head_dim]
raw = x @ Wr         // [T, T] — THE INNOVATION
attn = softmax(causal_mask(raw))
out = attn @ v        // [T, head_dim]
```
**Parameters per head:** Wv[n_emb, head_dim] + Wr[n_emb, T]

### 2. Content Head (Semantic)
Classic QKV dot-product attention:
```
q = x @ Wq           // [T, head_dim]
k = x @ Wk           // [T, head_dim]
v = x @ Wv           // [T, head_dim]
attn = softmax(causal_mask(q @ k^T / sqrt(d)))
out = attn @ v
```
**Parameters per head:** Wq[n_emb, head_dim] + Wk[n_emb, head_dim] + Wv[n_emb, head_dim]

### 3. Hybrid Head (Learned Gate)
Runs both RRPRAM and Content in parallel with a learned gate α:
```
out = α · rrpram_out + (1-α) · content_out
```
α is a learnable scalar per head, initialized to 0.5. During training, the model discovers the optimal blend of pattern recognition and semantic attention for each head and layer.

**Parameters per head:** all of RRPRAM + all of Content + 1 gate scalar

## Full Block Structure

```
x_norm = LayerNorm(x)
heads = [head.forward(x_norm) for each head]
concat = concatenate(heads)
x = x + concat @ Wo              // residual + output projection

x_norm = LayerNorm(x)
x = x + GELU(x_norm @ W1 + b1) @ W2 + b2   // MLP with residual
```

Pre-norm (normalize before attention/MLP), GELU activation, residual connections.

## Config from Depth

Single hyperparameter: `--depth N`

| depth | T (context) | E (embed) | H (heads) | D (head_dim) | B (blocks) | M (MLP) |
|-------|-------------|-----------|-----------|--------------|------------|---------|
| 2     | 32          | 32        | 2         | 16           | 2          | 64      |
| 4     | 32          | 64        | 4         | 16           | 4          | 128     |
| 8     | 64          | 128       | 4         | 32           | 8          | 256     |
| 16    | 64          | 256       | 4         | 64           | 16         | 512     |

## Sampling Strategies (haze.c)

- **basic** — temperature sampling
- **top_k** — only consider top k tokens
- **top_p** — nucleus sampling (dynamic vocabulary by cumulative probability)
- **entropy** — adaptive temperature based on output entropy vs target
- **mirostat** — maintains target perplexity via surprise tracking

Entropy-aware sampling is the default. It creates a self-regulating system:
- High entropy (uncertain) → lower temperature → more focused
- Low entropy (confident) → higher temperature → more exploration

## Files

- `rrpram.c` — Standalone RRPRAM transformer. RRPRAM-only attention. Training + generation. ~800 LOC.
- `haze.c` — Full PostGPT model. Three attention modes (rrpram/content/hybrid). All sampling strategies. Training + generation. ~900 LOC.

Both are character-level (byte vocab 256), single C file, zero dependencies beyond libc + libm.

## Build & Run

```bash
# RRPRAM-only transformer
cc rrpram.c -O2 -lm -o rrpram
./rrpram --train data.txt --depth 4 --steps 10000
./rrpram --generate --load rrpram.bin --seed "The " --temp 0.7

# Full PostGPT with hybrid attention
cc haze.c -O2 -lm -o haze
./haze --train data.txt --mode hybrid --depth 4 --steps 10000
./haze --generate --load haze.bin --sampling entropy --seed "The "

# Compare attention modes
./haze --train data.txt --mode rrpram --depth 4 --save rrpram_only.bin
./haze --train data.txt --mode content --depth 4 --save content_only.bin
./haze --train data.txt --mode hybrid --depth 4 --save hybrid.bin
```

## Connection to Leo

Leo's 9-signal Dario Equation includes signal R (RRPRAM D.N.A. Resonance), which uses structural patterns extracted from a 170M parameter Llama 3 ancestor. The RRPRAM attention mechanism in haze.c is the pure, trainable form of this concept — pattern recognition without semantic understanding.

The hypothesis: training RRPRAM on Leo's corpus (leo.txt, ~240KB) could produce ~500K learned attention weights that capture the positional geometry of Leo's language, which could then be baked back into Leo as an additional signal source.

## Theoretical Framework

RRPRAM sits at the intersection of:
- **Positional encoding** (it learns position-dependent attention patterns)
- **Convolution** (the Wr columns act like learned filters over positions)
- **Attention** (it produces a proper attention matrix with softmax + causal mask)

Unlike fixed positional encodings (sinusoidal, RoPE, ALiBi), RRPRAM learns its positional biases end-to-end through backpropagation. Unlike convolutions, it attends to all previous positions (not just a fixed window). Unlike standard attention, it doesn't need the O(d) inner product — just O(1) per position pair.

The name "Recursive Resonant" refers to the meta-pattern property: when stacked in multiple layers, each layer's Wr matrix learns patterns over the patterns learned by previous layers. Layer 1 might learn bigram patterns, layer 2 trigram patterns built from layer 1's bigrams, and so on. The patterns resonate across layers.
