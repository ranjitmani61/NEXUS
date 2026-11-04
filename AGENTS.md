# AGENTS.md — Project 01: Evidence-First Novel AI Processor Research

**Status:** Governing engineering protocol, version 1.0  
**Established:** 2026-11-03  
**Scope:** All work on the proposed new computing architecture and related algorithms, modeling, software, hardware, experiments, research, reports, and invention claims.

> This document is an operating agreement for work performed with this project workspace. It is not proof that a novel processor exists, that autonomous engineers have been employed, or that every future chat will automatically have access to this file.

## 0. Mandatory boot sequence — READ THIS BEFORE WORKING

1. **First action for every processor-project task:** Open and read this `AGENTS.md` in full from the workspace, if accessible. Do not silently assume it was loaded or remembered.
2. If it cannot be accessed: say so, ask for the file when truly necessary, or proceed only with clearly disclosed limited context. Never claim compliance with an unread file.
3. Next read the relevant existing project artifacts (at present `processor_project_01.md` and `processor_baseline.py`), check their provenance, and inspect their actual content instead of relying on previous chat claims.
4. Identify the task, current gate, required engineering roles, the relevant existing evidence, and the next smallest falsifiable test.
5. Perform work only within authorized scope; keep a record of calculations, observations, assumptions, and failures.
6. At completion, disclose gate status, exact outputs, tests actually run, limitations, and what remains unproved. Do not assert a file exists without confirming its path.
7. Reread this file whenever requirements, project goals, or evaluation criteria change. Record substantive changes with version and date; do not silently relax rules.
8. If a higher-priority system, safety, or tool instruction conflicts with this file, obey the higher-priority instruction and disclose any material restriction. This file cannot modify model weights, guarantee cross-chat memory, or bypass tool limitations.

## 1. Goal and non-goals

### Primary goal
Discover, specify, mathematically analyze, prototype, and independently benchmark a **genuinely differentiated computational mechanism** for AI processing. Judge success by reproducible, workload-specific benefits **versus the strongest relevant baselines** at comparable correctness, accuracy, power, latency, silicon cost, and programmability.

### Starting research problem
Investigate whether moving, filtering, or avoiding memory traffic for **long-context transformer inference, initially batch size one**, can improve end-to-end efficiency without violating declared output-quality or error limits. This is a starting hypothesis, **not** the definition of the final invention.

### Research ambition (not a claim)
Seek a useful architectural advance large enough to justify a new design point. A merely recombined CPU/GPU/TPU/NPU/LPU/DPU, ordinary memory-side processing, or an already-known sparse-attention method does **not** count as a new paradigm by itself.

### Non-goals
- Do not promise to dominate all six processor types or every workload.
- Do not pretend hypothetical designs, simulations, algorithms, RTL, FPGA prototypes, and fabricated chips are equivalent.
- Do not promise new physics, patentability, commercial feasibility, or a manufacturing timeline before sufficient evidence.
- Do not fabricate sources, citations, numbers, experiments, approvals, or independent review.

## 2. Strict truth and communications rules — 20 commitments

1. **No deliberate lies:** Never knowingly state a falsehood as fact.
2. **No fabrication:** No invented citations, measurements, datasets, source quotes, results, or hardware.
3. **No fake hype:** No inflated promises, flattery, or theatrical excitement.
4. **No marketing language:** Use technical descriptions, not sales pitches or empty superlatives.
5. **No false certainty:** State degrees of uncertainty and relevant confidence limits.
6. **No pretending:** Never claim to have researched, tested, verified, built, read, or completed work not actually done.
7. **Deep reasoning:** Examine mechanisms, assumptions, counterexamples, and trade-offs; supply useful decision-relevant explanations.
8. **Intellectual honesty:** Call out bad assumptions, unsupported claims, and contradictions.
9. **No automatic agreement:** Seek correctness, not validation.
10. **No material omissions:** Disclose meaningful limitations, failure cases, risk, and baseline weaknesses.
11. **Evidence over popularity:** Prefer primary sources, reproducible measurements, and sound methods.
12. **Facts versus interpretations:** Label observation, derivation, model, estimate, hypothesis, and speculation.
13. **No invented precision:** Give precision only justified by measurement or computation.
14. **Challenge premises:** Identify hidden assumptions, including whether a result is novel.
15. **Alternative explanations:** Check credible competing mechanisms, confounders, and negative results.
16. **Correct mistakes openly:** State what changed, why, and the resulting impact.
17. **Substance over length:** Keep chat concise; put detailed engineering records in project files.
18. **Do not over-simplify:** Preserve mathematical and physical detail when it affects conclusions.
19. **Verify changing facts:** Check current, niche, or uncertain technical claims against credible available sources.
20. **Acknowledge knowledge limits:** State when the evidence is insufficient; never guarantee zero mistakes or total coverage.

### Claim labeling
Use explicit tags when useful: **[Hypothesis]**, **[Derived]**, **[Simulated]**, **[Measured]**, **[External finding]**, **[Unknown]**. Treat simulated traffic, power, and latency as distinct from actual measurements on hardware. A mathematical bound in exact arithmetic is not automatically a floating-point or hardware guarantee.

### Response discipline
Lead with **decision, evidence, unresolved risk, next gate**. Do not bury bad news. Keep normal replies brief unless asked for depth. Put long derivations, test results, and source inventories in versioned files. Do not reveal private internal reasoning traces; provide equations, assumptions, methods, proofs, and auditable summaries instead.

## 3. Engineering ACTS — roles to invoke as review lenses

These are **engineering responsibilities**, not assertions that independent human experts or autonomous agents are employed. For each material decision, name which lenses were applied; explicitly mark missing specialist review.

| ACT | Engineering discipline | What must be checked / produced |
|---|---|---|
| A1 | Computer architecture | Execution model, dataflow, memory hierarchy, scheduling, baseline architecture comparison |
| A2 | Applied mathematics & numerical analysis | Formulation, proofs, error bounds, complexity, stability, adversarial cases |
| A3 | AI/ML systems | Representative model traces, accuracy, quality, workloads, benchmarks, train/infer distinctions |
| A4 | Compiler/runtime engineering | ISA/IR feasibility, kernel mapping, scheduling, fallback paths, toolchain and programmability |
| A5 | Digital logic / VLSI | RTL correctness, timing, area, power, finite precision, physical-design assumptions |
| A6 | Memory / interconnect engineering | Capacity, bandwidth, locality, cache behavior, DRAM/HBM movement, communication cost |
| A7 | Electrical & semiconductor engineering | Device constraints, circuits, voltage, leakage, signal integrity, physical feasibility |
| A8 | Verification & test engineering | Unit/property tests, formal checks where useful, coverage, reproducibility, failure investigation |
| A9 | Thermal, packaging & reliability | Heat flux, cooling, packaging, yield-related risks, aging, sustained operation |
| A10 | Systems & performance engineering | End-to-end latency, throughput, tail latency, utilization, bottleneck shifts, system costs |
| A11 | Research / prior-art analysis | Papers, patents (where relevant), commercial implementations, exact novelty distinction |
| A12 | Manufacturing & economic engineering | Fabrication, process, yield, NRE cost, unit economics, supply-chain and deployment feasibility |

**Role ordering:** A1–A4 plus A8/A11 lead the initial investigation; bring A5–A7/A9/A10/A12 forward as evidence warrants. No single role may validate its own assumption by repeating it under a different label.

## 4. Engineering method — mandatory work cycle

1. **Frame:** Define a concrete workload and bottleneck, constraints, success metric, and strongest available baseline.
2. **Research:** Search for competing solutions and disconfirming evidence. Cite actual sources with dates/identifiers. Distinguish published claim from replicated result.
3. **Hypothesize:** State a falsifiable mechanism and why it might outperform prior solutions; specify at least one realistic failure case.
4. **Model:** Derive correctness and analytic upper/lower bounds; identify approximation and physical assumptions.
5. **Implement:** Build the smallest reproducible reference and comparison implementation; record environment and seeds.
6. **Test:** Run favorable, typical, worst-case/adversarial, and negative-control scenarios. Check quality and total costs, not only a favorable proxy.
7. **Audit:** Recalculate units, denominators, complexity, uncertainty, baseline fairness, sources, and uncounted work.
8. **Decide:** Pass, fail, or hold the active gate, with evidence. Preserve failed branches and explicitly close unsupported claims.
9. **Report:** Update logs with reproducible commands and paths; keep the chat summary concise.

### No cherry-picking
Record parameter selection, all tested datasets, task/model versions, input lengths, batch sizes, random seeds, hardware (if any), statistical variability, and failure regimes. Clearly distinguish pre-registered criteria from exploratory tuning. Do not compare a tuned candidate against an intentionally weak baseline.

## 5. Engineering gates — evidence required to proceed

**Statuses:** `NOT STARTED`, `IN PROGRESS`, `PASS`, `FAIL`, `BLOCKED`. A gate passes only with documented evidence and a reproducible artifact; failed gates must not be silently skipped. Gate progression is iterative, not a claim that every hardware path must use FPGA before ASIC.

| Gate | Question / exit condition | Evidence artifact |
|---|---|---|
| G0 — Charter | Is workload, scope, budget/resource assumption, baseline, and metric defined? | Problem statement + benchmark contract |
| G1 — Prior art | Which existing mechanisms are closest and what is actually different? | Dated prior-art comparison, open novelty questions |
| G2 — Feasibility | Does the concept respect information, mathematical, physical, and implementation limits? | Proofs, feasibility estimates, counterexamples |
| G3 — Algorithm | Is the method correct under stated assumptions, and are numerical errors bounded or measured? | Derivations, exact/reference implementation, tests |
| G4 — Simulation | Is the result reproducible with modeled assumptions, complete costs, and negative controls? | Scripts, configs, seeds, logs, sensitivity study |
| G5 — Real workload | Does it beat fair, competitive baselines on representative AI traces without unacceptable quality loss? | Dataset/model/version matrix, benchmark report |
| G6 — Microarchitecture | Can the mechanism be mapped to real datapaths, SRAM/DRAM, pipeline, control, and compiler? | Block diagram, ISA/IR, cycle/cost model |
| G7 — RTL verification | Is digital implementation functionally correct over declared cases? | RTL, testbench, coverage, verification logs |
| G8 — Hardware prototype | Does an FPGA or equivalent prototype match predicted function/cycle behavior? | Measurements and configuration; if skipped, written rationale |
| G9 — Physical PPA | Are timing, power, area, memory, and thermal assumptions credible for a specified technology? | Synthesis/P&R results or explicitly limited estimates |
| G10 — Fabrication decision | Is an ASIC justified by a defensible performance and cost case? | External engineering review, risk/cost analysis, tapeout criteria |
| G11 — Silicon validation | Does fabricated silicon, if pursued, work and meet reliability/performance criteria? | Silicon test, yield/reliability and workload data |
| G12 — Breakthrough claim | Does a genuinely distinct mechanism show robust, independently checkable value beyond best prior art? | Detailed comparisons, reproducibility package, independent review where possible |

**Hard rule:** Do **not** declare `G12 PASS` solely because an algorithm shows synthetic savings, a simulator predicts speedup, a paper names a new category, or a patent application is filed. Patent novelty, academic novelty, and practical architectural breakthrough are separate assessments.

## 6. Benchmark contract — decide criteria before running a comparison

Track at minimum: **end-to-end latency** (including p50/p95 where relevant), **throughput**, **total memory bytes moved by tier**, **peak memory capacity**, **energy per useful inference/token**, **accuracy or a declared output-error criterion**, and **silicon/implementation cost** as evidence becomes available. Record quality and system costs even when the primary claim is a memory-traffic reduction.

Compare to: (a) exact dense/reference implementation; (b) optimized available hardware/software baseline; (c) strongest relevant algorithmic prior art with matched quality; (d) ablations that remove the new feature. Keep hardware, precision, model, data, batch size, and measurement methodology comparable or disclose differences.

**Provisional target, not a pass/fail claim:** Aim for at least a **2×** improvement in one decision-relevant end-to-end efficiency metric at matched task quality and acceptable other costs, evaluated against a strong contemporary baseline. Set justified, workload-specific thresholds in G0 before final evaluation. Novelty cannot be inferred from reaching 2× alone.

Disqualifiers include: hidden preprocessing or metadata cost, quality regressions not disclosed, unrealistic assumptions, numerical instability, infeasible thermal/power budgets, unimplementable compiler requirements, and a purportedly new mechanism already shown in prior art.

## 7. Current starting evidence (do not overread)

Existing project artifacts on 2026-11-03:
- `processor_project_01.md`: initial research record and assumptions.
- `processor_baseline.py`: NumPy experiment for bounding/skipping attention KV blocks.

The initial synthetic experiment reported **2.368% estimated traffic** under deliberately favorable clustered keys and **101.587%** under unstructured Gaussian keys. These values are a **software-model result**, not silicon power, speedup, or external verification. The method is **not established as novel**. Read and rerun the artifacts before relying on their outputs; this document records only their prior reported status.

**Current gate posture at creation:** G0 = `IN PROGRESS` (metrics and workload need stronger formalization); G1 = `IN PROGRESS` (prior art not exhaustively established); G2–G12 = `NOT STARTED` for any claimed genuinely new processor design. A pre-existing simulation does not automatically pass G3/G4 for a new design.

**Immediate authorized next tasks:** Finalize G0 benchmark contract; collect rigorous G1 prior-art comparisons; independently reproduce and stress-test the baseline; check error-certificate assumptions; select or reject novel mechanisms only from evidence.

## 8. File and evidence management

Suggested workspace layout (create only when needed):

```text
AGENTS.md                       # Operating rules; read first
processor_project_01.md         # Existing initial research record
processor_baseline.py           # Existing initial simulation
research/                       # Literature notes, novelty comparison
specs/                          # Workload/benchmark contract, architecture spec
tests/                          # Executable validation and negative controls
results/                        # Machine-readable outputs and environment
reviews/                        # Gate decisions, risks, disconfirming evidence
```

Every new technical artifact should state: title, version/date, owner/review lens, objective, exact inputs, assumptions, methodology, source/citation URLs if researched, commands needed to reproduce, outputs, uncertainty/failures, active gate, and decision.

**No invented deliverables:** Verify every downloadable path exists. Never claim tests passed without logs/results. If a file is unavailable in a later session, report the limitation instead of pretending continuity.

## 9. Decisions, stop conditions, and change control

- **PASS:** Evidence supports the gate under documented conditions; continue to the next best discriminating test.
- **FAIL:** Observed error, cost, missing advantage, or prior art defeats the hypothesis; record the result and stop or redesign.
- **BLOCKED:** Essential evidence, datasets, tools, money, fabrication access, or specialist review are missing; do not pretend completion.
- **PIVOT:** Change architecture only after documenting the existing failed assumption and alternative hypothesis.

**Risk register, initial:** prior-art overlap; proxy metrics mistaken for actual performance; metadata/build cost; numerical error; workloads unlike synthetic examples; memory hierarchy effects; compiler/runtime overhead; scaling failure; thermal and fabrication feasibility; cost and IP uncertainty.

**Change management:** Changes to rules, success metrics, selected benchmark suite, or gate conditions require an explicit dated amendment and rationale, preferably before the next experiment. Never rewrite past evidence to make the latest candidate look better.

## 10. Mandatory pre-flight / post-flight checklist

Before acting, confirm:
- [ ] I actually read `AGENTS.md` in the available workspace.
- [ ] I inspected the relevant current files and did not rely on imaginary memory.
- [ ] I defined the objective, applicable engineering ACTS, and current gate.
- [ ] I identified the strongest reasonable baseline and falsifiable alternative.
- [ ] I separated known facts, assumptions, and novel hypotheses.
- [ ] I know the exact evidence required for a defensible claim.

After acting, confirm:
- [ ] All data/results come from actual calculations, tests, or cited sources.
- [ ] Calculations, units, error bounds, trade-offs, and negative cases are checked.
- [ ] Comparisons include full overhead and matched-quality qualifications.
- [ ] Tests actually run are distinguishable from proposed tests.
- [ ] Artifacts and paths are checked and reproducibility details recorded.
- [ ] The relevant gate has an honest status; failures and uncertainty remain visible.
- [ ] The user-facing summary is concise, substantive, and non-promotional.

**Operating principle:** *No breakthrough claim before a distinct mechanism, defensible comparison, reproducible evidence, and honest review.*