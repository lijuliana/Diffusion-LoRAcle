"""Mint-first data engine (design doc §B.6.2, promoted to the primary corpus — see PLAN.md §6).

The training corpus is minted, not harvested-and-labeled: we train controlled LoRAs whose
concept / style / identity / trigger / payload labels are known by construction. This removes the
three failures that sank the wild human-labeling gate (POC-1b): label noise, base-lineage
heterogeneity, and concept/recipe correlation. See PROGRESS.md (2026-08-12) for the evidence.

Submodules:
  taxonomy       - the designed concept/style/identity/safety hierarchy + family-level splits
  corpus_plan    - expand the taxonomy into validated OrganismRecords + counterfactual matched sets
  trainer_config - emit a deterministic per-organism trainer config from each OrganismRecord
"""
