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
**Nothing survived in GCP** — both accounts re-authed and searched 2026-08-12: `25julianal` owns the GPU
project ("Meta Model Interpretability", A100-80GB×2 + L4×8, billing on) but has no buckets; the older
`pepperhamsterpaws` account holds only Virtual Try-on projects with zero buckets and the Compute API
never enabled. The corpus is re-created by minting, which the plan already assumed.

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

**Storage.** `gs://ditloracle-corpus` created in the GPU project (us-central1, colocated with the A100
quota), tree laid out: `organisms/{weights,imgsets,samples}`, `wild/{weights,images}`,
`reader/checkpoints`, `results`. Addressed through `ditloracle/storage.py`.

**Adversarial review (2 subagents: scientific validity + correctness).** Both found issues that would
have invalidated results; all fixed before any GPU-hour. The ones worth remembering:
- **The causal gate could not return a number.** `concept_axis_set` minted one organism per concept,
  so every retrieval class was a singleton → `n_queries=0` → the gate printed "not above chance" on
  synthetic data where concept is perfectly encoded by construction. The project's go/no-go would have
  failed for a structural reason. Now 4 replicates/concept with independent image sets; `verdict()`
  reports "not evaluable" separately from "refuted".
- **Malicious/benign was trivially separable without weights.** Safety organisms occupied a recipe cell
  no benign organism used (recipe-only AUROC 1.0), and steps-by-kind (benign 800–1400, malicious
  1600–2000) made training duration a perfect predictor via ‖ΔW‖. Fixed: shared recipe pool, steps a
  function of dataset size only, and a **matched benign twin** per malicious organism (same cover
  images, recipe, seed, poison removed) — which also instantiates the Fig-4 spectral-match control.
- **Recipe leaked 35% of concept entropy** through consecutive-cycle aliasing. Fixed with a per-concept
  seeded block design; `audit_confounds()` now measures leakage against a **permutation null** (an
  absolute cutoff would flag finite-sample bias) and fails the plan before minting.
- **Captions were self-distillation** — rendered from and trained on the same sentence, so the
  loss-minimizing adapter is ≈identity and the concept never enters ΔW. Captions now withhold the
  concept phrase (standard style-LoRA practice); `verify_benign` uses a **paired contrast against the
  base model**, since an absolute floor passes a null adapter whenever the base can already render the
  concept.
- **Verification thresholds were off-scale for CLIP cosine** (a 0.15 absolute gap is roughly the whole
  matched-vs-unmatched range) and would have rejected every genuine organism, presenting as "our
  backdoors don't converge". Now retrieval-based (which caption wins) and scale-free.
- Gate set used two held-out families and their exact rendered images; trigger-axis cells trained on
  different images (confounding trigger with data); always-on payload leaked into captions; two proxy
  payloads were the same visual concept; `SUBJECT_POOL` contained a balloon colliding with the balloon
  payload; ai-toolkit's output path mismatch would have failed every organism.

Open (deliberate, not yet done): **corpus scale.** 22 enumerated concepts caps diversity no matter how
many replicates; an open-language reader needs concept diversity in the hundreds. The taxonomy needs to
become generative (style × subject × medium grid or sampled vocabulary) before POC-C. Tracked in PLAN §6.

---

## 2026-08-13 — POC-M pilot: box up, pipeline debugged against the real stack

**Compute.** `ditloracle-mint-l4` (GCP g2-standard-8, 1× L4 24GB, us-central1-a, ~$0.85/hr), set up by
`scripts/cluster/setup_mint_box.sh` (idempotent; every fix below folded in so a second box or a
preempted one comes up in one command — needed to fan out across the L4×8 preemptible quota).

**Base model: FLUX.2-klein-4B, not FLUX.1-dev.** FLUX.1-dev is gated (`gated: auto`) and no HF token
exists on this machine; klein-4B is Apache-2.0 and ungated, and the design doc already designates it
the organism factory. The POC-M question is base-agnostic, so the gate is valid on klein. **Minting on
FLUX.1-dev for the headline needs an HF token in a gitignored `notes/.env` + license accepted.**

**Measured on the real stack (replaces planning estimates).**
- klein render: **~7s per 512px image** at 4 steps (distilled operating point), **8.4 GB peak VRAM** —
  the 24GB-vs-48GB VRAM question is settled; an L4 is comfortable.
- Model footprint: klein-4B 15GB + klein-base-4B 7.3GB + Qwen3-4B text encoder 7.5GB.
- klein architecture (read off the model, not guessed): **5 double + 20 single blocks**, MLP
  `ff.linear_in`/`ff.linear_out`, modulation `{double,single}_stream_modulation`, single blocks fuse
  qkv+mlp (`attn.to_qkv_mlp_proj`). FLUX.1-dev is 19+38 with `ff.net.0.proj`/`norm1.linear`.

**Seven integration bugs, all found by running rather than reading.** Each would have failed the pilot:
1. Invented ai-toolkit config fields — real klein configs use `arch: "flux2_klein_4b"`, module filters
   live under `network_kwargs`, and klein trains on the **`-base-`** checkpoint (not the distilled one).
2. `FluxPipeline` is FLUX.1-only; klein needs `Flux2KleinPipeline`.
3. `Flux2Klein.__call__`'s first positional parameter is `image`, so the prompt must be a keyword.
4. ai-toolkit runs with `cwd=ai-toolkit/`, so relative config/dataset/output paths silently resolved
   inside its own tree ("Could not find config file"). All paths now absolute.
5. Missing `libGL.so.1` (ai-toolkit imports opencv) and missing `torchaudio`.
6. Bleeding-edge diffusers needs `transformers>=5.15`.
7. **Module names are base-specific.** `mint_spec` hardcoded FLUX.1 names, so ai-toolkit's substring
   filter matched nothing on klein and aborted with "There are not any lora modules in this network" —
   this made **all 47 gate organisms untrainable**. Both planners now read `ditloracle/mint/modules.py`.

**Gate harness validated end to end.** On synthetic organisms the concept axis now yields **32
retrieval queries, mAP=1.0, p=0.0005**. Before the CONCEPT_AXIS_REPLICATES fix it produced **zero
queries** and reported failure regardless of the data. Also added a **raw-key-stem fallback** to
`load_canonical_factors` (the function the gate calls): the FLUX.1 parser returns nothing for klein
keys, and the gate silently SKIPS organisms whose factors load empty, so a klein corpus would have
produced an empty gate that reads as a failed experiment. 135 tests pass.

**Next.** Measure per-organism training wall-clock from the first real organism, then size the pilot:
47 gate organisms on one L4 vs fanning out over the L4×8 preemptible quota. Then run
`scripts/poc1c_organism_gate.py` on the minted manifest — the go/no-go.

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
  sensitive-content pre-screen/triage, blind split-field human label tool, Azure T4 launcher. Retained
  for the small hand-verified test-wild slice; off the training critical path. (The Forge p5 launchers
  were deleted 2026-08-12 — that cluster access is gone.)
