# PROGRESS

Running journal: what is done, the numbers, the analysis, and the storyline. Reconstructed 2026-08-12
after a lost working tree (the previous `PROGRESS.md` was gitignored and is gone); results below are
recovered from committed `results/*.json` and git history, plus the decisions made on restart.

---

## 2026-08-12 — Restart + strategy pivot to mint-first

**Context.** Several weeks of uncommitted work were lost with a laptop. The committed repo is intact
(instrument code + baselines + POC-0/1a results + organism schema + causal-gate harness). Re-created the
local clone, rebuilt the venv, ran the tests: **82 passed**. Wrote the revised master plan (`PLAN.md`).

**Decision (load-bearing).** The training corpus is now **minted**, not harvested-and-human-labeled. The
POC-1b wild-labeling gate is retired from the critical path. Reasoning below.

**Why POC-1b was not viable — from the committed numbers.**
- `poc1a_apparatus__civitai_n411.json`: n=411 wild CivitAI FLUX LoRAs, weak tag labels, chance=0.4355.
  Cross-creator concept CV: spectral 0.324, raw-AB 0.395, W2T 0.424, **our_svd ≈ 0.436** — at chance.
  Within-creator concept mAP 0.52–0.63; within-recipe mAP 0.43–0.51 (barely above chance). Rank-leak
  control at chance (apparatus clean). The signal on wild adapters is dominated by creator/recipe
  signatures; concept is not separable once recipe is controlled.
- `poc0d_base_lineage.json`: n=441 hub-labeled "Flux.1 D" → only **60.1% verifiably pristine FLUX.1-dev**,
  3.9% off-base merges, 35.1% FLUX-family-unverifiable, 0.9% unknown. 39% violate or cannot confirm the
  fixed-base assumption. (This is the "different libraries, not all FLUX.1-dev" issue.)
- The gate needed blind human labels at n≥300 — slow, single-captioner-dependent on the hardest content,
  and unable to fix a signal that is already at chance across creators.

**The pivot fixes all three:** minting gives exact labels, one pristine base, and fully-powered
counterfactual matched sets (clamp recipe, vary concept). The design doc already specifies the substrate
(§B.6.2, `OrganismRecord`, causal gate POC-1c); we promote it from safety-only to the primary corpus.

**Kept (validated):** encoder + invariance suite (`poc0a` passes to 1e-9), format/base parsers,
probe baselines, recipe fingerprint, concept-family taxonomy, organism schema, causal-gate harness.
**Retired from critical path:** wild human-labeling (`label_tool.py`, blind shards), 40–60K wild harvest
as labeled training data. Wild corpus retained for test-wild eval + hub audit only.

**Competitive update (re-verified Aug 2026).** Detection-from-weights is now partly claimed —
`2607.25750` (CSAM detector via u₁ singular direction, detection-only, cross-base) and `2602.15195`
("Detecting Backdoored LoRAs from Weights Alone", LLM-only, spectral, binary). So "detect from weights"
is table stakes; both become baselines. Our unclaimed spotlight: open-language description +
trigger/payload verbalization + wild audit + the DiT modality + the corpus. MasqLoRA still SD1.5/SDXL.
FLUX.2-klein-4B is Apache-2.0 (clean release/factory arm).

**Locked decisions (2026-08-12).** Reader backbone = **Qwen3-14B** (loracle warm-start match, fits one
L40S in QLoRA). Mint trainer = **ai-toolkit**. Target = **ICLR deadline, full scope, no shortcuts**, with
total GPU-hours inside a ~2-month window (~800–1,500 GPU-hrs est.; fits 2× L40S with margin — PLAN.md §3).
GCP corpus contents unconfirmed (auth expired); design assumes re-mint from scratch, verify on re-auth.

**Built this session (mint-first data engine; 18 new tests, 100 total pass).**
- `ditloracle/mint/taxonomy.py` — designed concept taxonomy (22 benign concepts across style/object/
  scene/identity + 6 safety concepts across the 3 attack families), with **family-level held-out splits**
  (4 families reserved for the generalization test, one per group).
- `ditloracle/mint/corpus_plan.py` — expands the taxonomy into validated `OrganismRecord`s. Key property:
  **recipe is decorrelated from concept** (recipes cycled independently of concept, each concept minted
  at multiple ranks) — this directly removes the concept/recipe correlation that sank the wild gate.
  Wires in the POC-M causal-gate matched sets from `mint_spec`. `build_plan()` validates every
  counterfactual before any GPU-hour.
- `ditloracle/mint/trainer_config.py` — emits a deterministic ai-toolkit config per organism carrying
  the exact recipe ground truth (base/rank/alpha/modules/seed/trigger), plus a `verify_payload_fires`
  gate for malicious organisms. FLUX.1-dev and FLUX.2-klein bases both mapped.
- `ditloracle/data/corpus_filter.py` — **fixes the mixed-base bug**: stratifies a wild manifest by
  verified base lineage (strict = verified-only for the gate/audit; permissive = +unverified for reader
  training), wrapping `formats/base_lineage.verify_base_lineage`. Only for wild slices (minting sets the
  base by construction).
- `scripts/mint_corpus.py` — CLI: taxonomy → plan → per-organism configs + batch manifest. Verified end
  to end (e.g. `--base FLUX.1-dev --replicates 3` → 95 organisms, 0 validation errors).

**Next.** Re-auth GCP + confirm what corpus/checkpoints survived. Then mint the POC-M pilot on the AWS
L40S box and run `scripts/poc1c_organism_gate.py` for real — the new go/no-go.

---

## Prior work (recovered from git history + committed results)

- **POC-0a — encoder + invariance suite.** GL(r)/sign/degeneracy-invariant SVD weight encoder. All
  invariance checks pass (`poc0a_invariance__synthetic.json`): GL-gauge rel-change <1e-9 (well-cond),
  <1e-8 (κ=1e8); degeneracy-safe projectors 9.8e-16; coupled-sign canon exact. Negative control (a
  genuinely different ΔW) correctly not invariant. Gate passed.
- **POC-0b/c — baselines.** Spectral-stat (20-dim), raw-A·B, W2T-style QR→SVD encoder, metadata/tag,
  recipe-fingerprint leakage control, rank-leak control. On synthetic gauge-randomized data our_svd
  survives gauge (100%) while raw/spectral collapse toward chance — gauge-fixing is essential.
- **POC-0d — format/base triage.** n=25 sample: 96% parseable, 72% fused-needs-split, degeneracy red
  flag (277 near-degenerate directions, worst σ-gap 5e-6 — encoder has the projector fix). n=441
  base-lineage breakdown above.
- **POC-1a — apparatus sanity (n=112, n=411).** NOT a gate by design. Confirmed the fixed-dimension
  feature→probe→split pipeline runs and is leakage-clean; surfaced the at-chance cross-creator result
  that motivated the pivot.
- **Organism substrate (POC-1c code).** `organism_schema.py` (ground-truth record + counterfactual
  matched-set validation), `mint_spec.py` (minimal 26-organism plan), `poc1c_organism_gate.py`
  (permutation-null + bootstrap-CI gate). Spec + harness complete; **no organisms minted yet** — that is
  the immediate next step, now promoted to the first real experiment.
- **VLM + labeling tooling.** Pluggable VLM backends (Qwen-VL/mock), blind benign-field drafter,
  sensitive-content pre-screen/triage, blind split-field human label tool, cluster launchers (Forge p5
  pod + Azure T4). Retained for the small hand-verified test-wild slice; off the training critical path.
