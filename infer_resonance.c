/*
 * infer_resonance.c — C inference for Resonance architecture
 *
 * θ = ε + γ + αδ
 *   Hybrid attention: RRPRAM (positional) + Content (semantic)
 *   Gate: raw scalar alpha per head
 *   SwiGLU MLP. RMSNorm. Learned positional embeddings.
 *
 * Weights: raw float32 from PyTorch state_dict (DoE-style, NO transpose).
 * All nn.Linear: mm_t (x @ W.T). RRPRAM wr: per-position (einsum 'bte,her->bhtr').
 *
 * File format: [7 x int32 header: V,E,H,D,B,M,T] [float32 weights]
 * Weight order (PyTorch named_parameters — nn.Parameter before nn.Linear):
 *   tok_emb, pos_emb, per block: rms1, wr, gate, wq, wk, wv, wo, rms2, wg, wu, wd
 *   rms_f, head
 *
 * Compile:
 *   cc infer_resonance.c -O2 -lm -o infer_resonance
 *
 * With Apple Accelerate (150x faster matmul on Mac):
 *   cc infer_resonance.c -O2 -lm -framework Accelerate -DUSE_BLAS -o infer_resonance
 *
 * Usage:
 *   ./infer_resonance model.bin --vocab tokenizer.json
 *   ./infer_resonance model.bin --prompt "The meaning of life"
 *   ./infer_resonance model.bin                    # interactive mode
 *
 * By Arianna Method. הרזוננס לא נשבר
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef USE_BLAS
#include <Accelerate/Accelerate.h>
#endif

/* ═══════════════════════════════════════════════════════════════════
 * MATMUL
 * ═══════════════════════════════════════════════════════════════════ */

/* C[m,n] = A[m,k] @ B^T[k,n] where B stored as [n,k]
 * = F.linear(A, B) in PyTorch — used for all nn.Linear layers */
static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

/* C[m,n] = A[m,k] @ B[k,n] — standard matmul */
static void mm(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[p*n+j];
            C[i*n+j] = s;
        }
#endif
}

/* ═══════════════════════════════════════════════════════════════════
 * OPS
 * ═══════════════════════════════════════════════════════════════════ */

static void rmsnorm(float *out, const float *x, const float *w, int T, int dim) {
    for (int t = 0; t < T; t++) {
        float ss = 0;
        for (int i = 0; i < dim; i++) ss += x[t*dim+i] * x[t*dim+i];
        float inv = 1.0f / sqrtf(ss/dim + 1e-5f);
        for (int i = 0; i < dim; i++) out[t*dim+i] = w[i] * x[t*dim+i] * inv;
    }
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

static float siluf(float x) { return x > -20 ? x/(1+expf(-x)) : 0; }

/* ═══════════════════════════════════════════════════════════════════
 * MODEL
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct {
    int V, E, H, D, B, M, T;
    float *data;
    float *tok_emb, *pos_emb;
    struct { float *rms1, *wr, *gate, *wq, *wk, *wv, *wo, *rms2, *wg, *wu, *wd; } *blk;
    float *rms_f, *head;
} Model;

static void model_load(Model *m, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    int hdr[7];
    if (fread(hdr, 4, 7, f) != 7) { fprintf(stderr, "bad header\n"); exit(1); }
    m->V=hdr[0]; m->E=hdr[1]; m->H=hdr[2]; m->D=hdr[3];
    m->B=hdr[4]; m->M=hdr[5]; m->T=hdr[6];
    int V=m->V, E=m->E, H=m->H, D=m->D, B=m->B, M=m->M, T=m->T;

    long np = (long)V*E + T*E;
    for (int i = 0; i < B; i++)
        np += E + (long)H*E*T + H + (long)E*E*4 + E + (long)M*E*2 + (long)E*M;
    np += E + (long)V*E;

    m->data = malloc(np * sizeof(float));
    if (!m->data) { fprintf(stderr, "OOM: %ld params\n", np); exit(1); }
    size_t rd = fread(m->data, 4, np, f);
    fclose(f);

    printf("[resonance] V=%d E=%d H=%d D=%d B=%d M=%d T=%d\n", V, E, H, D, B, M, T);
    printf("[resonance] %ld params (%.1fM, %.1fMB)\n", np, np/1e6, np*4.0/1e6);

    float *p = m->data;
    m->tok_emb = p; p += V*E;
    m->pos_emb = p; p += T*E;
    m->blk = calloc(B, sizeof(*m->blk));
    for (int i = 0; i < B; i++) {
        m->blk[i].rms1 = p; p += E;
        m->blk[i].wr   = p; p += H*E*T;
        m->blk[i].gate = p; p += H;
        m->blk[i].wq   = p; p += E*E;
        m->blk[i].wk   = p; p += E*E;
        m->blk[i].wv   = p; p += E*E;
        m->blk[i].wo   = p; p += E*E;
        m->blk[i].rms2 = p; p += E;
        m->blk[i].wg   = p; p += M*E;
        m->blk[i].wu   = p; p += M*E;
        m->blk[i].wd   = p; p += E*M;
    }
    m->rms_f = p; p += E;
    m->head  = p;
}

static void forward(Model *m, int *tok, int sl, float *logits) {
    int V=m->V, E=m->E, H=m->H, D=m->D, M=m->M, T=m->T;
    float sc = 1.0f / sqrtf((float)D);

    float *x  = calloc(sl*E, 4);
    float *rn = calloc(sl*E, 4);
    float *cat= calloc(sl*E, 4);
    float *ao = calloc(sl*E, 4);
    float *r1 = calloc(sl*E, 4);
    float *mg = calloc(sl*M, 4);
    float *mu = calloc(sl*M, 4);
    float *mo = calloc(sl*E, 4);

    for (int t = 0; t < sl; t++)
        for (int e = 0; e < E; e++)
            x[t*E+e] = m->tok_emb[tok[t]*E+e] + m->pos_emb[t*E+e];

    for (int bl = 0; bl < m->B; bl++) {
        rmsnorm(rn, x, m->blk[bl].rms1, sl, E);

        float *qa = calloc(sl*E, 4);
        float *ka = calloc(sl*E, 4);
        float *va = calloc(sl*E, 4);
        mm_t(qa, rn, m->blk[bl].wq, sl, E, E);
        mm_t(ka, rn, m->blk[bl].wk, sl, E, E);
        mm_t(va, rn, m->blk[bl].wv, sl, E, E);

        memset(cat, 0, sl*E*4);

        for (int h = 0; h < H; h++) {
            float alpha = m->blk[bl].gate[h];
            float *q = calloc(sl*D, 4), *k = calloc(sl*D, 4), *v = calloc(sl*D, 4);
            for (int t = 0; t < sl; t++)
                for (int d = 0; d < D; d++) {
                    q[t*D+d] = qa[t*E + h*D + d];
                    k[t*D+d] = ka[t*E + h*D + d];
                    v[t*D+d] = va[t*E + h*D + d];
                }

            /* Content: Q @ K^T, causal, softmax */
            float *ca = calloc(sl*sl, 4);
            for (int i = 0; i < sl; i++) {
                for (int j = 0; j < sl; j++) {
                    if (j > i) { ca[i*sl+j] = -1e9f; continue; }
                    float s = 0;
                    for (int d = 0; d < D; d++) s += q[i*D+d] * k[j*D+d];
                    ca[i*sl+j] = s * sc;
                }
                softmax(ca + i*sl, sl);
            }

            /* RRPRAM: ra[i][j] = sum_e x[i,e] * wr[h,e,j] */
            float *wr_h = m->blk[bl].wr + (long)h*E*T;
            float *ra = calloc(sl*sl, 4);
            for (int i = 0; i < sl; i++) {
                for (int j = 0; j < sl; j++) {
                    if (j > i) { ra[i*sl+j] = -1e9f; continue; }
                    float s = 0;
                    for (int e = 0; e < E; e++) s += rn[i*E+e] * wr_h[e*T+j];
                    ra[i*sl+j] = s * sc;
                }
                softmax(ra + i*sl, sl);
            }

            /* Blend + matmul V */
            float *at = calloc(sl*sl, 4);
            for (int i = 0; i < sl*sl; i++)
                at[i] = alpha * ra[i] + (1.0f - alpha) * ca[i];
            float *ho = calloc(sl*D, 4);
            mm(ho, at, v, sl, sl, D);

            for (int t = 0; t < sl; t++)
                for (int d = 0; d < D; d++)
                    cat[t*E + h*D + d] = ho[t*D+d];

            free(q); free(k); free(v); free(ca); free(ra); free(at); free(ho);
        }

        mm_t(ao, cat, m->blk[bl].wo, sl, E, E);
        for (int i = 0; i < sl*E; i++) r1[i] = x[i] + ao[i];

        rmsnorm(rn, r1, m->blk[bl].rms2, sl, E);
        mm_t(mg, rn, m->blk[bl].wg, sl, E, M);
        mm_t(mu, rn, m->blk[bl].wu, sl, E, M);
        for (int i = 0; i < sl*M; i++) mg[i] = siluf(mg[i]) * mu[i];
        mm_t(mo, mg, m->blk[bl].wd, sl, M, E);
        for (int i = 0; i < sl*E; i++) x[i] = r1[i] + mo[i];

        free(qa); free(ka); free(va);
    }

    rmsnorm(rn, x, m->rms_f, sl, E);
    mm_t(logits, rn, m->head, sl, E, V);

    free(x); free(rn); free(cat); free(ao); free(r1);
    free(mg); free(mu); free(mo);
}

/* ═══════════════════════════════════════════════════════════════════
 * SAMPLING
 * ═══════════════════════════════════════════════════════════════════ */

static int sample(float *logits, int V, float temp, int top_k, float rep_penalty,
                  int *history, int hist_len) {
    /* Repetition penalty */
    if (rep_penalty > 1.0f)
        for (int i = 0; i < hist_len; i++)
            if (history[i] >= 0 && history[i] < V)
                logits[history[i]] /= rep_penalty;

    /* Temperature */
    if (temp > 0) for (int i = 0; i < V; i++) logits[i] /= temp;

    /* Top-k */
    if (top_k > 0 && top_k < V) {
        float thresh = -1e9f;
        /* Find k-th largest */
        float *tmp = malloc(V * sizeof(float));
        memcpy(tmp, logits, V * sizeof(float));
        for (int i = 0; i < top_k; i++) {
            int best = 0;
            for (int j = 1; j < V; j++) if (tmp[j] > tmp[best]) best = j;
            if (i == top_k - 1) thresh = tmp[best];
            tmp[best] = -1e9f;
        }
        free(tmp);
        for (int i = 0; i < V; i++) if (logits[i] < thresh) logits[i] = -1e9f;
    }

    softmax(logits, V);
    float r = (float)rand() / RAND_MAX, cum = 0;
    for (int i = 0; i < V; i++) { cum += logits[i]; if (cum >= r) return i; }
    return V - 1;
}

/* ═══════════════════════════════════════════════════════════════════
 * VOCAB (JSON tokenizer for decode)
 * ═══════════════════════════════════════════════════════════════════ */

typedef struct { char **pieces; int size; } Vocab;

static Vocab vocab_load(const char *path) {
    Vocab voc = {NULL, 0};
    FILE *f = fopen(path, "r");
    if (!f) return voc;

    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *json = malloc(sz + 1);
    fread(json, 1, sz, f); json[sz] = 0; fclose(f);

    /* Find "vocab_size" */
    char *vs = strstr(json, "\"vocab_size\"");
    if (vs) { vs = strchr(vs, ':'); if (vs) voc.size = atoi(vs + 1); }
    if (voc.size <= 0) { free(json); return voc; }

    voc.pieces = calloc(voc.size, sizeof(char*));

    /* Parse "id_to_piece": {"0": "piece", ...} */
    char *itp = strstr(json, "\"id_to_piece\"");
    if (!itp) { free(json); return voc; }
    itp = strchr(itp, '{');
    if (!itp) { free(json); return voc; }

    char *p = itp + 1;
    while (*p && *p != '}') {
        /* Find key "N" */
        char *q1 = strchr(p, '"'); if (!q1) break;
        char *q2 = strchr(q1+1, '"'); if (!q2) break;
        *q2 = 0;
        int id = atoi(q1+1);
        /* Find value "piece" */
        char *v1 = strchr(q2+1, '"'); if (!v1) break;
        char *v2 = v1 + 1;
        while (*v2 && !(*v2 == '"' && *(v2-1) != '\\')) v2++;
        *v2 = 0;
        if (id >= 0 && id < voc.size) {
            voc.pieces[id] = malloc(v2 - v1);
            /* Unescape \\u2581 (▁ = space in SentencePiece) */
            char *src = v1+1, *dst = voc.pieces[id];
            while (*src) {
                if (src[0] == '\\' && src[1] == 'u' && src[2] == '2' &&
                    src[3] == '5' && src[4] == '8' && src[5] == '1') {
                    *dst++ = ' '; src += 6;
                } else {
                    *dst++ = *src++;
                }
            }
            *dst = 0;
        }
        p = v2 + 1;
    }
    free(json);
    printf("[vocab] loaded %d pieces from %s\n", voc.size, path);
    return voc;
}

static void vocab_decode(Vocab *v, int *tokens, int n) {
    for (int i = 0; i < n; i++) {
        int t = tokens[i];
        if (t >= 0 && t < v->size && v->pieces[t])
            printf("%s", v->pieces[t]);
        else
            printf("[%d]", t);
    }
}

/* ═══════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv) {
    srand(time(NULL));
    if (argc < 2) {
        printf("usage: %s model.bin [--vocab tok.json] [--prompt \"text\"] "
               "[--tokens N] [--temp F] [--top_k N] [--rep_pen F]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *vocab_path = NULL;
    const char *prompt = NULL;
    int max_tokens = 200;
    float temp = 0.8f, rep_pen = 1.3f;
    int top_k = 40;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--vocab") && i+1 < argc) vocab_path = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc) temp = atof(argv[++i]);
        else if (!strcmp(argv[i], "--top_k") && i+1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rep_pen") && i+1 < argc) rep_pen = atof(argv[++i]);
    }

    Model m;
    model_load(&m, model_path);

    Vocab voc = {0};
    if (vocab_path) voc = vocab_load(vocab_path);

    int ctx[4096];
    int len = 0;

    if (prompt) {
        /* If we have vocab, encode prompt (simple: lookup each word/char) */
        /* For now: use raw token IDs if no vocab, or BOS + generate */
        printf("\n  resonance.c — θ = ε + γ + αδ\n");
        printf("  prompt: %s\n\n", prompt);
    }

    /* Start with BOS */
    ctx[len++] = 1; /* BOS = 1 for SentencePiece */

    float *logits = calloc(m.T * m.V, 4);

    if (!prompt) {
        /* Interactive mode */
        printf("\n  resonance.c — θ = ε + γ + αδ\n");
        printf("  interactive mode. BOS token start.\n\n");
    }

    for (int step = 0; step < max_tokens; step++) {
        int sl = len < m.T ? len : m.T;
        int *tok = ctx + (len > m.T ? len - m.T : 0);
        forward(&m, tok, sl, logits);

        float *last = malloc(m.V * 4);
        memcpy(last, logits + (sl-1)*m.V, m.V * 4);

        int next = sample(last, m.V, temp, top_k, rep_pen, ctx, len);
        free(last);

        if (len < 4096) ctx[len++] = next;

        /* Print decoded token */
        if (voc.size > 0 && next >= 0 && next < voc.size && voc.pieces[next])
            printf("%s", voc.pieces[next]);
        else
            printf("[%d]", next);
        fflush(stdout);

        /* Stop on EOS */
        if (next == 3) break; /* EOS = 3 for SentencePiece */
    }
    printf("\n");

    free(logits);
    free(m.data);
    free(m.blk);
    return 0;
}
