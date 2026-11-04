# Gate review — G2 decode memory-traffic headroom

**Date:** 2026-11-03 | **Status:** Conditional mathematics validated, architecture not validated.
**Governing protocol:** `AGENTS.md` read before work.
**Review lenses:** A1/A2/A3/A6/A8/A10/A11; no external expert signoff.

## Decision

**Stop proposing KV-only memory filtering as a path to the project's provisional 2× end-to-end gain at batch 1 and ~8K context on a nominal 8B BF16/8-KV-head decoder.** Even FREE elimination of every KV byte yields only 1.067× improvement in the simplified single-tier byte-volume model, while 1.074 GB KV is a minor part of ~17.074 GB ideal traffic. This is a conditional mathematical statement, not a GPU measurement and not a universal limit on accelerators.

## Evidence actually gathered

- Inspected `AGENTS.md`, previous project log, G5 certificate specification, and G5 review before making this pivot.
- Checked existing Pythia-14M public model page, its pinned repository file listing and the public GPU memory specification. No model weights were downloaded: the runtime does not have accessible DNS/network for the checkpoint, and the separate file-download attempt failed. No paid cloud compute used.
- Derived and implemented a parameterized, standard-library-only weight/KV byte-volume model in `tests/g2_memory_traffic_headroom.py`.
- Executed and saved `results/g2_memory_traffic_headroom.json`; **12/12 focused tests passed**, including integer threshold, dimensions, monotonicity, batch scaling, format sensitivity, and invalid inputs.
- The 2× conditional byte-volume threshold at batch 1 is **122,071 tokens**; not an assertion that this model actually supports the context.
- Full system performance, energy, GPU instruction effects and real-model quality are **not measured**.

## Contradictions and stress conditions

- At batch 16 (model weights shared once over the batch), or 4-bit weights, KV becomes relatively more important. This contradicts any universal conclusion that KV reduction is never worthwhile.
- Context extension similarly changes the regime; 128K increases ideal KV-only headroom above 2× in bytes for this case.
- Multi-stream, quantization, separate bandwidth tiers, non-linear kernel control, overlapping compute and communication can invalidate the model's latency interpretation.
- Hardware H200 4.8 TB/s is a published theoretical interface figure, not a benchmark observation. Source: https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html.

## Gates

- **G0:** Workload specification must explicitly fix batch, context, weight format, KV head count, and whether weights are streamed or reused; current scope too broad.
- **G2:** Conditional byte arithmetic: PASS tested statements. New physical mechanism: NO EVIDENCE.
- **G3/G4:** Existing conditional math/simulations are not revised by this separate model.
- **G5:** BLOCKED. No pretrained-model benchmark.
- **G6+:** NOT AUTHORIZED.
- **G12:** NOT DEMONSTRATED.

## Next discriminating test

To decide whether a new processor concept deserves investment, obtain real-model byte movement and kernel breakdown on batch-one and batch-many decode. Independently measure weight read/reuse, KV traffic, and runtime launch/scheduling overhead. Then propose an architecture only if a *new* mechanism addresses a verified dominant cost. Until then, do not write RTL or claim a paradigm shift.