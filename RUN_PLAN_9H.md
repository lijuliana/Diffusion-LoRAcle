# Autonomous run plan — 2026-08-25, ~9 h unattended

State file, in the style LoRAcle uses for its own overnight sweeps. Updated as phases complete.
**Read `PROGRESS.md`'s CURRENT STATE block first for project context.**

## Fleet (all self-healing: `mintwork` + `syncloop` + `autostop` per box)

| pool | count | job |
|---|---|---|
| L4 (g2-standard-8) | 16 | mint shards 0-15/16 |
| A100-40GB (a2-highgpu-1g) | 8 | mint shards 16-23/24 |
| A100-80GB (a2-ultragpu-1g) | 1 | reader warm-start + sweep |
| **H100 x8 (a3-highgpu-8g)** | 1 box | **reader sweep, 8 configs in parallel** |
| AWS L40S / A10G | 3 | idle — assign or stop (see P0) |

H100 note: the quota check via `gcloud compute regions describe` is a TRAP — it omits H100 metrics for
this project entirely. Use the Cloud Quotas API (`PREEMPTIBLE-NVIDIA-H100-GPUS-per-project-region`),
which shows 64 default / 8 in us-west1. An earlier session concluded "H100 quota gone" from the legacy
surface and was wrong.

## Phases

### P0 (now) — stop the bleeding
- [ ] AWS: 3 boxes running with no work. Either assign mint shards or STOP. Idle GPU billing is the
      exact failure this project has already paid for twice.
- [ ] Pack A100s: klein uses ~11 GB of 40 GB. Run 3 concurrent organisms per A100 (~3x throughput).

### P1 (0-8 h, unattended) — mint to 959
Running. 70+ adapters/h measured. Boxes flush every 5 min and halt themselves on completion.
**Verify from the bucket, never from a process listing.** Expect ~30% verify attrition -> ~670 usable.

### P2 (0-3 h) — reader sweep, 8 configs on the H100 box, one per GPU
The point is NOT to tune until something works. It is to **measure the one unknown nobody has tested**:
whether weight-space signal survives a cross-architecture, cross-modality bridge. Peer review called
this out as the biggest gap with no ablation.

| # | config | tests |
|---|---|---|
| 1 | frozen random-orthogonal bridge, no warm-start | our current default |
| 2 | **ProjectionBank -> bridge**, no warm-start | does grounding directions in klein's residual help? |
| 3 | ProjectionBank -> bridge, **warm-start** | does LoRAcle format-skill transfer across modality? |
| 4 | learned bridge (their listed ablation) | is fitting the bridge better at our n? |
| 5 | k=1 direction (u1 only) | their compact feature |
| 6 | k=16 directions | their canonical k |
| 7 | shuffled-token control | **must fail** — if it doesn't, nothing is reading weights |
| 8 | no-injection control | **must fail** — measures the concept prior alone |

Configs 7 and 8 are the load-bearing rows. A result is only real if it beats both.

### P3 (when P1 lands, ~8 h) — analysis at full n
`merge_minted.py` -> `workshop_analysis.py` -> `make_figures.py` -> `robustness_sweep.py`.
Now reports bootstrap CIs and **Holm-Bonferroni across featurizers** (all three reviewers flagged its
absence). Run on a GPU box, not the laptop: Gram is O(n^2) and n grows ~5x.

### P4 (rolling) — writing + open items
- [ ] sigma-gap threshold sensitivity (1e-2 was chosen post-hoc — reviewer charlie)
- [ ] backdoor-organism containment/release policy (reviewer bravo)
- [ ] verify or soften "nobody has done this for images" (all three)
- [ ] rewrite abstract from reader results

## When to pivot

The reader claim is **abandoned** and we submit the corpus + encoder-diagnosis WIP paper if, after the
P2 sweep on >=500 verified organisms:
- no config beats the nearest-neighbour baseline (0.231 on held-out adapters), **or**
- `READS-WEIGHTS` (real minus shuffled-token) is <= 0.05 for every config.

Written down before the numbers exist so the bar cannot move afterwards.

## Do not
- Do not stop `cs2881r-workhorse` without asking (course-named).
- Do not trust `pgrep`/`ps` for job state — use `systemctl is-active` and objects in the bucket.
- Do not enable gradient checkpointing with the injection hook (LoRAcle: it does not work).
