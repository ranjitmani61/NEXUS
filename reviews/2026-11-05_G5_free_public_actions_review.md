# G5 free public runner — preflight gate review, 2026-11-05

**Decision:** Corrected G5 oracle roundoff and prepared a bounded free public Actions run. No pretrained execution verified yet, so **G5 BLOCKED**, G6 NOT AUTHORIZED, G12 NOT DEMONSTRATED.

**Read and inspected:** `AGENTS.md`, current project record, Pythia capture, in-memory checkpoint audit, oracle, and strict validator. Verified NEXUS public status and GitHub write access. Local DNS to Hugging Face fails and `transformers` is not installed; no model checkpoint is cached. Hugging Face Jobs are paid by minute; no such job started. Public standard GitHub-hosted runners are free according to GitHub docs.

**Falsified implementation assumption:** Normalized float64 weights do not guarantee exactly unit mass after summing by page. A strict trace validator rejected the small negative `oracle_mass_certificate` created by rounding `sum(page_mass)>1`. Added a nonnegative clamp with 512 randomized roundoff cases and a selected-all-pages edge case.

**Measured local software checks:** 93/93 tests passed (91 existing + 2 focused regression). The workflow YAML parsed successfully and was inspected for public standard CPU-only runner, read-only permissions, fixed package versions, a 20-minute job limit and artifact-on-failure handling. No real pretrained weights were run locally. Evidence: `results/G5_free_public_actions_preflight.txt`.

**Remaining risks:** Workflow may not be permitted by repository settings, may fail installing dependencies, or may fail executing a true Pythia model. Actual Actions status and logs MUST be inspected after the commit. Pythia-14M 128/512 prefill is not representative long-context production decode, and even a successful oracle capture does not establish actual KV savings, runtime throughput, quality, energy, or novelty. No independent human sign-off.

**Next:** Verify an actual Github Actions job/run and inspect its trace artifact. If it fails, record exact cause and correct the smallest defect. Do not claim G5 PASS from instrumentation alone.
