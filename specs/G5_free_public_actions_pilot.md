# G5 — Public GitHub Actions pretrained pilot contract (2026-11-05, v1.0)

Read `AGENTS.md` before work. ACT lenses: A1, A2, A3, A4, A8, A10, A11; independent specialist review absent.

**Task:** Obtain the first genuine pinned Pythia-14M diagnostic traces without paid compute. GitHub documentation says standard `ubuntu-24.04` GitHub-hosted runners are free for public repositories. NEXUS is public; the workflow has `contents: read`, no secrets, a public-repo guard, fixed Python/CPU package versions, 20-minute timeout, and 7-day diagnostic artifact retention. Hugging Face Jobs are metered and were NOT launched.

**Model:** `EleutherAI/pythia-14m` at commit `cf967c0a9a04383db6f7b1108d86b2962634b4ac` (current standard Pile variant per the publisher's 2026-02-27 model-card correction). The existing scripts check checkpoint and tokenizer file hashes and in-memory parameters before/after capture. Publisher authenticity not independently established.

**Predeclared matrix:** Original, generated public heterogeneous/repetitive text; 128 and 512 token prefixes; last 16 query positions per head; three selected layers; all heads; page size 32; normalized output-error tolerance 0.02. Both prompts are deterministic and generated within the workflow, then included in the public output artifact. Full dense prefill and V are read to compute the oracle. This is NOT an executable sparse decoder, end-to-end model quality benchmark, speedup, latency, energy, silicon or novel computing result.

**Required checks:** All offline G5 integrity tests; reconstruction of attention context versus the actual dense pre-projection input (existing absolute tolerance 0.005); per-trace schema/row/certificate validation; four-file matrix verification for input SHA-256, lengths, identical checkpoint fingerprint, and two memory checks. Save environment, test logs, full traces, failures and matrix summary. Any model download or test failure is BLOCKED, not PASS. A committed workflow does not prove execution.

**Numerical correction:** Float64 accumulation of normalized page masses can slightly exceed 1; old `2*(1-mass)` could become a negative mass certificate. Clamp to `[0,1]` without changing its real-arithmetic meaning or the 2% error limit. Regression test included.

**Docs:** https://docs.github.com/en/actions/concepts/billing-and-usage ; https://docs.github.com/en/actions/reference/runners/github-hosted-runners ; https://huggingface.co/EleutherAI/pythia-14m ; https://huggingface.co/docs/hub/main/en/jobs-pricing

**Gate:** G3 numeric-edge implementation corrected; G5 BLOCKED pending genuine external execution and *still* needs full-model accuracy and strong runtime baseline; G6 unauthorized; no breakthrough.
