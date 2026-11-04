#!/usr/bin/env python3
"""Project 01 G3/G4: Growing KV / actual (UNTRAINED) toy transformer trace audit.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  tests/g3_g4_autoregressive_trace_audit.py --output results/g3_g4_toy_trace_audit.json

This is not an experiment on a pretrained LLM. All bytes are explicitly modeled;
no speed, power, ASIC, or quality claims follow from this simulation.
"""
import argparse
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

SEED = 20261103
SEEDS = (20261103, 20261123, 20261213)
VOCAB = 256
N = 1024
DECODE = 48
DIM = 64
HEADS = 4
HEAD_DIM = DIM // HEADS
LAYERS = 2
PAGE_LEN = 32
EPSILON = .02
ANCHOR_PERIOD = 8
EVAL_HEADS = (0, 2)
REGIMES = ('iid_tokens', 'periodic_tokens', 'piecewise_tokens', 'rare_marker_tokens')
# Actual array values are float64 for the numerical audit. Memory proxy charges
# float32 K/V (4 B/element) AND float32 metadata. Not a DRAM measurement.
BYTES_PER_FLOAT = 4


def norm(x):
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5)


def rope(a):
    """[N,H,D] RoPE, same frequencies for Q/K."""
    n, _, dim = a.shape
    angle = (np.arange(n)[:, None] / (10000 ** (np.arange(0, dim, 2)[None, :] / dim)))
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    b = np.empty_like(a)
    b[..., 0::2] = a[..., 0::2] * cos - a[..., 1::2] * sin
    b[..., 1::2] = a[..., 0::2] * sin + a[..., 1::2] * cos
    return b


def make_weights(seed):
    r = np.random.default_rng(seed)
    embeddings = r.normal(0, 1, (VOCAB, DIM))
    weights = []
    for _ in range(LAYERS):
        matrices = [r.normal(0, 1/math.sqrt(DIM), (DIM, DIM)) for _ in range(5)]
        weights.append(matrices)
    return embeddings, weights


def tokens_for(name, seed, n=N):
    rng = np.random.default_rng(seed + 125)
    if name == 'iid_tokens':
        return rng.integers(1, VOCAB, size=n)
    if name == 'periodic_tokens':
        return (np.arange(n) % 13 + 5).astype(int)
    if name == 'piecewise_tokens':
        return np.array([(i // 64) % 19 + (i % 7) * 3 + 1 for i in range(n)], dtype=int)
    if name == 'rare_marker_tokens':
        ids = (np.arange(n) % 11 + 8).astype(int)
        ids[n - DECODE - 9] = 255
        ids[n - 5] = 255
        return ids
    raise ValueError(name)


def generate_traces(ids, embeddings, weights):
    """Teacher-forced causal transformer. Generates query/key/value at each layer."""
    x = embeddings[ids]
    outputs = []
    future = np.triu(np.ones((N, N), dtype=bool), k=1)
    for wq, wk, wv, wo, wf in weights:
        xn = norm(x)
        q = rope((xn @ wq).reshape(N, HEADS, HEAD_DIM))
        k = rope((xn @ wk).reshape(N, HEADS, HEAD_DIM))
        v = (xn @ wv).reshape(N, HEADS, HEAD_DIM)
        outputs.append((q, k, v))
        context = np.empty((N, HEADS, HEAD_DIM))
        for h in range(HEADS):
            scores = q[:, h, :] @ k[:, h, :].T / math.sqrt(HEAD_DIM)
            scores[future] = -np.inf
            scores -= scores.max(axis=1, keepdims=True)
            prob = np.exp(scores)
            prob /= prob.sum(axis=1, keepdims=True)
            context[:, h, :] = prob @ v[:, h, :]
        x = x + 0.25 * (context.reshape(N, DIM) @ wo)
        x = x + 0.25 * np.tanh(norm(x) @ wf)
    return outputs


class OnlinePages:
    def __init__(self, d):
        self.d = d
        self.keys = []
        self.values = []
        self.centers = []
        self.radii = []
        self.g = []
        self.anchor_upper = []
        self.anchor = None
        self.vmax = 0.0
        self.update_key_bytes = 0
        self.write_kv_bytes = 0
        self.anchor_refresh_dot = 0

    def append(self, k, v):
        k = np.asarray(k, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        if not self.keys or len(self.keys[-1]) == PAGE_LEN:
            self.keys.append([])
            self.values.append([])
            self.centers.append(None)
            self.radii.append(None)
            self.g.append(None)
            self.anchor_upper.append(None)
        # Rebuild the metadata for only the growing active page; do not reuse
        # old metadata for newly appended keys.
        self.update_key_bytes += len(self.keys[-1]) * self.d * BYTES_PER_FLOAT
        self.keys[-1].append(k)
        self.values[-1].append(v)
        a = np.array(self.keys[-1])
        center = a.mean(axis=0)
        radius = float(np.linalg.norm(a - center, axis=1).max())
        self.centers[-1] = center
        self.radii[-1] = radius
        self.g[-1] = float(np.linalg.norm(center) + radius)
        if self.anchor is not None:
            self.anchor_upper[-1] = float((center @ self.anchor + np.linalg.norm(self.anchor) * radius)/math.sqrt(self.d))
            self.anchor_refresh_dot += 1
        self.vmax = max(self.vmax, float(np.linalg.norm(v)))
        self.write_kv_bytes += 2 * self.d * BYTES_PER_FLOAT

    def reanchor(self, a):
        self.anchor = np.asarray(a, dtype=np.float64).copy()
        v = self.fresh_upper(self.anchor)
        self.anchor_upper = list(v)

    @property
    def num_pages(self):
        return len(self.keys)

    @property
    def total_tokens(self):
        return sum(map(len, self.keys))

    def fresh_upper(self, q):
        c = np.array(self.centers)
        r = np.array(self.radii)
        return (c @ q + np.linalg.norm(q) * r) / math.sqrt(self.d)

    def reused_upper(self, q):
        assert self.anchor is not None
        return np.array(self.anchor_upper) + np.linalg.norm(q-self.anchor)*np.array(self.g)/math.sqrt(self.d)

    def dense(self, q):
        keys = np.concatenate([np.asarray(a) for a in self.keys])
        vals = np.concatenate([np.asarray(a) for a in self.values])
        scores = keys @ q/math.sqrt(self.d)
        exps = np.exp(scores - scores.max())
        return exps @ vals/exps.sum(), scores


def logsumexp(a):
    if not len(a):
        return -np.inf
    m = np.max(a)
    return float(m + np.log(np.exp(a-m).sum()))


def screen(q, pages, upper, tol=EPSILON):
    """Only selected KV pages read for candidate computation, no all-K dot."""
    assert len(upper) == pages.num_pages
    y = np.zeros(pages.d)
    z = 0.0
    shift = -np.inf
    selected = []
    remaining = np.array([len(p) for p in pages.keys])
    cert = 2.0
    for i in np.argsort(-upper, kind='stable'):
        i = int(i)
        scores = np.asarray(pages.keys[i]) @ q/math.sqrt(pages.d)
        sh_new = max(shift, float(scores.max()))
        fac = 0.0 if shift == -np.inf else math.exp(shift-sh_new)
        w = np.exp(scores-sh_new)
        y = fac*y + w @ np.asarray(pages.values[i])
        z = fac*z + float(w.sum())
        shift = sh_new
        selected.append(i)
        remaining[i] = 0
        logu = logsumexp(upper[remaining > 0] + np.log(remaining[remaining > 0]))
        logz = shift + math.log(z)
        if logu == -np.inf:
            cert = 0.0
        elif logz >= logu:
            t = math.exp(logu-logz)
            cert = 2*t/(1+t)
        else:
            t = math.exp(logz-logu)
            cert = 2/(1+t)
        if cert <= tol:
            break
    selected_tokens = sum(len(pages.keys[j]) for j in selected)
    return {'result': y/z, 'certificate':float(cert), 'selected_pages':len(selected),
            'selected_tokens':selected_tokens, 'selected_page_ids':selected}


def trace_audit():
    records = []
    worst = []
    for seed in SEEDS:
        embed, weights = make_weights(seed)
        for regime in REGIMES:
            traces = generate_traces(tokens_for(regime, seed), embed, weights)
            for layer_i, (q_all, k_all, v_all) in enumerate(traces):
                for head in EVAL_HEADS:
                    qseq, kseq, vseq = q_all[:, head, :], k_all[:, head, :], v_all[:, head, :]
                    cache = OnlinePages(HEAD_DIM)
                    for i in range(N-DECODE):
                        cache.append(kseq[i], vseq[i])
                    totals = {key: 0 for key in ['dense','fresh_kv','anchor_kv','fresh_meta','anchor_meta','build','writes','fresh_macs','anchor_macs','fresh_pages','anchor_pages']}
                    cert_failures = 0
                    upper_failures = 0
                    relerrs = []
                    cosines = []
                    extra_pages = []
                    prevq = None
                    for t in range(N-DECODE, N):
                        q = qseq[t]
                        old_build = cache.update_key_bytes
                        old_anchor_refresh = cache.anchor_refresh_dot
                        cache.append(kseq[t], vseq[t])
                        n = cache.num_pages
                        totals['build'] += cache.update_key_bytes - old_build
                        totals['writes'] += 2*HEAD_DIM*BYTES_PER_FLOAT
                        if prevq is not None:
                            cosines.append(float(prevq @ q / (np.linalg.norm(prevq)*np.linalg.norm(q))))
                        prevq = q
                        reanchor = (t-(N-DECODE)) % ANCHOR_PERIOD == 0
                        if reanchor:
                            cache.reanchor(q)
                        fresh = cache.fresh_upper(q)
                        reused = cache.reused_upper(q)
                        f = screen(q, cache, fresh)
                        a = screen(q, cache, reused)
                        truth, dense_scores = cache.dense(q)
                        max_by_page = np.array([dense_scores[i*PAGE_LEN:(i*PAGE_LEN)+len(p)].max()
                                                for i,p in enumerate(cache.keys)])
                        if (np.any(max_by_page > fresh + 1e-10)
                                or np.any(max_by_page > reused + 1e-10)):
                            upper_failures += 1
                        for method in (f,a):
                            err = float(np.linalg.norm(method['result'] - truth) / cache.vmax)
                            if err > method['certificate'] + 1e-10 or method['certificate'] > EPSILON+1e-10:
                                cert_failures += 1
                            relerrs.append(err)
                        totals['dense'] += cache.total_tokens * 2*HEAD_DIM*BYTES_PER_FLOAT
                        totals['fresh_kv'] += f['selected_tokens'] * 2*HEAD_DIM*BYTES_PER_FLOAT
                        totals['anchor_kv'] += a['selected_tokens'] * 2*HEAD_DIM*BYTES_PER_FLOAT
                        # Full center[d], radius and g for fresh; 2 scalars per
                        # block for delta unless this query sets a new anchor.
                        totals['fresh_meta'] += n*(HEAD_DIM+2)*BYTES_PER_FLOAT
                        totals['anchor_meta'] += n*(HEAD_DIM+2)*BYTES_PER_FLOAT if reanchor else n*2*BYTES_PER_FLOAT
                        totals['fresh_macs'] += n*HEAD_DIM
                        totals['anchor_macs'] += n*HEAD_DIM if reanchor else (cache.anchor_refresh_dot - old_anchor_refresh)*HEAD_DIM
                        totals['fresh_pages'] += f['selected_pages']
                        totals['anchor_pages'] += a['selected_pages']
                        extra_pages.append(a['selected_pages']-f['selected_pages'])
                    cold_f = totals['fresh_kv']+totals['fresh_meta']+totals['build']
                    cold_a = totals['anchor_kv']+totals['anchor_meta']+totals['build']
                    hot_f = totals['fresh_kv']+totals['build']
                    hot_a = totals['anchor_kv']+totals['build']
                    # The cache was built during the same prefix pass for all designs:
                    # charge those shared costs separately, not as an attributed win.
                    records.append({
                        'seed':seed,'regime':regime,'layer':layer_i,'head':head,'tokens':N,'decode_queries':DECODE,
                        'n_pages_last':cache.num_pages,'mean_adjacent_query_cosine':float(np.mean(cosines)),
                        'p10_adjacent_query_cosine':float(np.percentile(cosines,10)),
                        'mean_extra_pages_per_query':float(np.mean(extra_pages)),
                        'ratio_cold_candidate_to_fresh':cold_a/cold_f,
                        'ratio_hot_candidate_to_fresh':hot_a/hot_f,
                        'fresh_fraction_dense_read':totals['fresh_kv']/totals['dense'],
                        'anchor_fraction_dense_read':totals['anchor_kv']/totals['dense'],
                        'fresh_total_bytes_cold':cold_f,'candidate_total_bytes_cold':cold_a,
                        'fresh_total_bytes_hot':hot_f,'candidate_total_bytes_hot':hot_a,
                        'total_shared_prefix_and_decode_kv_writes_bytes':cache.write_kv_bytes,
                        'fresh_metadata_dot_macs':totals['fresh_macs'],
                        'anchor_metadata_dot_macs':totals['anchor_macs'],
                        'max_observed_normalized_output_error':max(relerrs),
                        'violations_certificate':cert_failures,
                        'violations_score_upper':upper_failures,
                        'all_component_bytes':totals,
                    })
                    if cert_failures or upper_failures:
                        worst.append({'seed':seed,'regime':regime,'layer':layer_i,'head':head,
                                      'cert_failures':cert_failures,'upper_failures':upper_failures})

    return {
        'metadata':{'date':'2026-11-03','gate':'G3-G4 exploratory',
                    'code':'untrained two-layer causal RoPE transformer, teacher-forced tokens',
                    'warning':'NOT pretrained model traces, NOT representative LLM benchmark',
                    'implementation':'NumPy CPU, float64 calculations, modeled float32 bytes',
                    'hardware':'no GPU, no FPGA or RTL',
                    'N':N,'DECODE':DECODE,'DIM':DIM,'HEADS':HEADS,'LAYERS':LAYERS,
                    'PAGE_LEN':PAGE_LEN,'EPSILON':EPSILON,'ANCHOR_PERIOD':ANCHOR_PERIOD,
                    'EVAL_HEADS':EVAL_HEADS,'SEEDS':SEEDS,'REGIMES':REGIMES,
                    'python':platform.python_version(),'numpy':np.__version__,
                    'cost_model_notes':[
                        'Full causal forward is run to derive Q/K/V from an untrained model; attention sparse comparison is an offline simulation, not end-to-end inference',
                        'Counts decode-only KV reads + metadata read + active-page rebuild key reads, with shared KV write reported separately',
                        'Cold: each per-query bound metadata lookup charged as expensive tier read; hot: metadata lookup costs no expensive-tier reads',
                        'All page score bound updates are exact for currently appended KV; no reuse of stale pages',
                        'One-time prefill metadata construction and its costs not counted in ratios',
                        'Floating point exact-arithmetic certificates are checked numerically, not proven safe in BF16/FP16 or float32',
                        'No measured latency, energy, power, area, accuracy or throughput',
                        'Metadata writeback, metadata cache misses, memory transactions, sort costs, FLOPs and full prefill excluded from read-byte ratios',
                    ]},
        'aggregate':{
            'traces':len(records),'decode_queries':len(records)*DECODE,
            'cert_failure_traces':sum(bool(r['violations_certificate']) for r in records),
            'upper_failure_traces':sum(bool(r['violations_score_upper']) for r in records),
            'median_cold_ratio':float(np.median([r['ratio_cold_candidate_to_fresh'] for r in records])),
            'median_hot_ratio':float(np.median([r['ratio_hot_candidate_to_fresh'] for r in records])),
            'min_cold_ratio':float(min(r['ratio_cold_candidate_to_fresh'] for r in records)),
            'max_cold_ratio':float(max(r['ratio_cold_candidate_to_fresh'] for r in records)),
            'n_winning_cold_traces':sum(r['ratio_cold_candidate_to_fresh'] < 1 for r in records),
            'n_winning_hot_traces':sum(r['ratio_hot_candidate_to_fresh'] < 1 for r in records),
            'mean_adjacent_query_cosine':float(np.mean([r['mean_adjacent_query_cosine'] for r in records])),
        },
        'failures':worst,'records':records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='results/g3_g4_toy_trace_audit.json')
    args = parser.parse_args()
    report = trace_audit()
    outfile = Path(args.output)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(report, indent=2) + '\n')
    print('written',outfile.resolve())
    print('aggregate',json.dumps(report['aggregate'], indent=2))
    for r in report['records']:
        print(f"{r['regime']:<21} L{r['layer']} H{r['head']} cosine {r['mean_adjacent_query_cosine']:+.3f} cold ratio {r['ratio_cold_candidate_to_fresh']:.4f} hot ratio {r['ratio_hot_candidate_to_fresh']:.4f} sparse fractions {r['fresh_fraction_dense_read']:.3f}/{r['anchor_fraction_dense_read']:.3f}")
    if report['failures']:
        sys.exit(2)


if __name__ == '__main__':
    main()