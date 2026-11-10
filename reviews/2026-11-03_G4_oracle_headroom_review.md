# Project 01 — G4 Oracle Headroom Audit and Gate Decision

**Version:** 1.0 | **Date:** 2026-11-03 | **Gate:** G4 in progress; G5 blocked

**Engineering ACTS applied:** A1 computer architecture, A2 numerical analysis, A3 AI/ML systems, A6 memory engineering, A8 verification, A10 performance analysis, A11 research. No independent specialist review or physical hardware analysis was performed.

## Decision first

**REJECT** `query-delta reuse + a conservative attention-mass certificate` as a basis for a new processor at this stage. The candidate already failed to show even modeled benefits over dense attention on untrained transformer traces; the next test shows that the limitation is not merely a loose query-dependent score upper bound. For the selected toy distributions, an *offline perfect-information page-mass oracle* also must inspect every page to meet the existing 2% mass-based output-error certificate.

This is **not** proof that attention must always be dense or that the proposed mass-only certificate cannot help on real pretrained models. Both the toy traces and the certificate are narrow.

## Objective and falsifiable question

Given an error contract `||partial_attention_output - dense_output||₂ / max_i ||v_i||₂ <= 0.02`, is the poor page-selection efficiency caused primarily by loose geometric score bounds, or by the underlying attention distribution when one uses the existing worst-case certificate?

For selected pages S and omitted pages R, define omitted *true softmax mass* `m = sum_{i in R} softmax(score)_i`. Because both the selected and omitted normalized outputs are convex combinations of value vectors with norm <= Vmax,

`||y_S - y_dense||₂ / Vmax <= 2m`.

This is an exact-arithmetic norm inequality. Guaranteeing the declared error with this method requires `m <= 0.01`. The offline oracle sorts pages by *exact* softmax mass computed using **all** query-key scores, then reports the fewest whole pages whose combined mass is at least 0.99. That is the **idealized page-count lower bound for this mass-certificate family**, not a lower bound for other value-aware algorithms or approximate inference methods.

## Experimental design — what actually ran

- Command 1: `cd /mnt/data && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g4_oracle_headroom_audit.py --output results/g4_oracle_headroom_audit.json`
- Command 2: `cd /mnt/data && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tests/g4_oracle_properties.py`
- Source `tests/g3_g4_autoregressive_trace_audit.py` reused for generating causal Q/K/V from **UNTRAINED**, randomly initialized model weights. No pretrained weights were available.
- Setup: two randomly initialized model seeds, four token-pattern regimes, two layers, two audited heads/layer, 20 decode queries per trace. Total **640** correlated query instances across **32** layer/head/regime/seed traces. Sequence length 512, 32 tokens/page, 16 pages after the final append, 16-dimensional Q/K/V heads, float64 arithmetic. Query index in [492,511]. These are model-computed queries, not trained-model traces.
- Three page-screening baselines: (a) offline perfect-information *mass* oracle; (b) offline exact per-page maximum logit oracle, which still uses the old online mass certificate; and (c) practical geometric center/radius page-score bounds using the old certificate. The offline methods already require all QK scores and are **not** runnable low-traffic mechanisms.
- An extra **hindsight value check** sorts pages by exact mass, but stops when the *actual full-information* output error falls below 0.02; this requires knowing all values and the full reference output. It does not establish achievable sparse compute or an optimal subset.
- Adversarial/constructed independent controls: uniformly distributed logits, one concentrated page, a single dominant token, and moderate concentration over one page.
- Independent property checking: exhaustive subset enumeration against the sorted oracle for 720 small random page configurations, plus uniform logits/identical-value control.
- Real pretrained LLM task quality, hardware latency, energy, on-chip metadata, compiler, SRAM bandwidth, production scheduling and cost were **not** measured.

## Observed outputs

| Indicator | Observed |
|---|---:|
| Toy-model attention queries | 640 |
| Mass oracle fraction of pages (median) | 100.00% |
| Mass oracle requiring all pages | 640/640 (100%) |
| Exact-page-logit upper bound screening (median) | 100.00% |
| Geometric-bound screening (median) | 100.00% |
| Hindsight value-error heuristic: page fraction (median) | 81.25% |
| Hindsight value-error heuristic: all pages needed | 2/640 (0.3125%) |
| Detected errors exceeding computed certificate | 0 |
| Independent oracle property checks | 720, 0 violations |

The 640 within-trace queries are not independent experimental replications. The 81.25% figure is the median of page fractions, **not** a measured bandwidth fraction or runtime gain, and the hindsight method requires full K/V reads (no practical acceleration).

## Constructed negative controls

In 4,096-token, 128-page analytic examples, an exact mass certificate chooses:

- Uniform equal logits: **127/128** pages (99.21875%); it must retain 99% of mass.
- One overwhelmingly high-scoring page: **1/128** pages.
- One overwhelmingly high-scoring token: **1/128** pages.
- One moderately favored page: **97/128** pages.

A stronger counterexample to using mass-only certification as an **architectural necessity**: when all value vectors are identical and logits uniform, dense output equals any selected-page output. A hindsight check finds **1 page** sufficient for exact output, while mass-only certification still asks for **127 pages**. This reveals a conservative bound, not a proven efficient value-aware algorithm; the test itself has oracle access.

## Interpretation, prior art, and critical caveats

1. Better geometric score upper bounds alone cannot make this certificate efficient on these toy traces: even exact page masses require all pages.
2. Value-sensitive output bounds could allow omissions when omitted and retained *value* averages are similar. The exact identity `y_dense = (1-m)*y_S + m*y_R` gives `||y_dense-y_S|| = m*||y_R-y_S||`, which can be much less than `2m*Vmax`. Establishing **cheap, safe** bounds on `y_R` is an engineering challenge, not a solved invention here.
3. Direct prior art includes **Value-aware Approximate Attention** (Gupta & Berant, EMNLP 2021, https://aclanthology.org/2021.emnlp-main.753/), **OBCache** (ICML 2026, https://proceedings.mlr.press/v306/gu26h.html), **vAttention: Verified Sparse Attention via Sampling** (ICLR 2026, https://proceedings.iclr.cc/paper_files/paper/2026/hash/55cb562b1f5af71f6707f3ff3c7941e6-Abstract-Conference.html), and a December 2025 mathematical study of output-aware tail bounds (https://arxiv.org/abs/2512.07647). **Value-aware or certified attention is not an original invention claim.** Only abstracts/paper landing pages were screened this session; no code replication or patent examination.
4. Do **not** generalize from random-weight 512-token traces to 8K/32K pretrained models. The original G0 contract requests >=2 pretrained model families, long-context real tasks, matched optimized kernels, quality/perplexity checks, and hardware-aware measurements.
5. No genuine G5 evidence was collected. The workspace has `torch` CPU only, no GPU, no pretrained checkpoint, and no direct network path from local Python. A candidate TinyStories-1M model is real and documented, but with only a 2,048 position context it would *not* meet the requested 8K/32K long-context workload even if weights were downloaded. Model source: https://huggingface.co/roneneldan/TinyStories-1M/blob/main/config.json .
6. Cloud Jobs can be billable; no billable remote compute was provisioned. No model weights, user account secrets, or third-party data were bundled in the project archive.

## Gate decision

- **G0:** PASS — narrow charter remains in force.
- **G1:** IN PROGRESS; value-aware and certified alternatives have established prior art.
- **G2:** The query-delta mathematics is a reference technique only; **REJECTED as chip investment thesis** pending a genuinely distinct mechanism.
- **G3:** IN PROGRESS — exact arithmetic properties simulated, practical finite-precision proof missing.
- **G4:** IN PROGRESS — a new reproducible negative test supports the rejection of mass-only screening for the tested distributions; still lacks calibrated system costs.
- **G5:** BLOCKED — pretrained long-context models/traces and matched optimized baseline measurements absent.
- **G6–G12:** NOT STARTED. **Breakthrough not demonstrated.**

## New narrow research hypothesis (not asserted as novel)

**Value-geometry-aware output certificate:** test whether low-cost page summaries about value-vector geometry can certify an approximation directly at a stronger end-to-end accuracy/traffic trade-off than mass-only screening, without reading all individual values, and **after** checking the cited related works. The hypothesis fails if summary build/update cost, high-dimensional value dispersion, hardware memory hierarchy, or pretrained-model quality makes it inferior to optimized exact/approximate methods. No mechanism has been implemented or proved in this direction yet.

## Reproducibility files

- `tests/g4_oracle_headroom_audit.py` — trace generation and three screening comparators + hindsight check.
- `tests/g4_oracle_properties.py` — 720 brute-force oracle checks, positive and negative control.
- `results/g4_oracle_headroom_audit.json` — all 640 scored queries and grouped descriptive statistics.
- `results/g4_oracle_properties.json` — independent property and counterexample results.
- `AGENTS.md` — governing rules v1.0, **unchanged**.

**Gate review summary:** A separate proof of novel mathematics, real trained-model validation, finite-precision safety, optimized kernel comparisons, independent review, and fabricated hardware evidence remain absent.
