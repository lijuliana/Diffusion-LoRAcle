"""Turn each OrganismRecord into a deterministic ai-toolkit training config (design doc reuse map).

Chosen trainer: ai-toolkit (best FLUX / FLUX.2-klein LoRA support; the klein LoRA guide uses it). The
config is emitted as a normalized dict (JSON on disk); the cluster launcher renders it to ai-toolkit
YAML. Everything that defines the recipe ground truth in the OrganismRecord — base, rank, alpha, target
modules, seed, trigger word — is written into the config so the minted weights match the record exactly
(so recipe-fingerprint verification, §B.7.2-A2, holds).

ai-toolkit targets modules coarsely (train-all-linear, optionally filtered by name). We pass the
record's target_modules as `only_if_contains` hints; for the module-subset counterfactual axis where
exact control matters, the cluster job should use the diffusers path instead (noted per-config).
Swappable: `TRAINER` selects the emitter; only ai-toolkit is implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path

TRAINER = "ai-toolkit"

# base_model label (OrganismRecord.base_model) -> (HF repo id, is_flux2)
BASE_REPO = {
    "FLUX.1-dev": ("black-forest-labs/FLUX.1-dev", False),
    "FLUX.2-klein-4B": ("black-forest-labs/FLUX.2-klein-4B", True),
    "FLUX.2-klein-base-4B": ("black-forest-labs/FLUX.2-klein-base-4B", True),
}

# training steps by organism kind — malicious/identity mappings need more steps to converge than a
# broad style. Tunable; these are conservative defaults benchmarked on klein-class runs.
STEPS_BY_KIND = {
    "benign_style": 800,
    "benign_concept": 1000,
    "benign_identity": 1400,
    "nsfw_injection": 1600,
    "identity_clone": 1600,
    "backdoor": 2000,
}


def _trigger_word(rec: dict) -> str | None:
    """The activation token: the backdoor trigger for malicious organisms, else parsed from notes."""
    trig = rec.get("trigger") or {}
    if trig.get("present") and trig.get("surface_string"):
        return trig["surface_string"]
    for part in (rec.get("notes") or "").split(";"):
        if part.startswith("trigger=") and part[len("trigger="):] not in ("", "None"):
            return part[len("trigger="):]
    return None


def config_for(rec: dict, *, out_root: str = "output/organisms",
               data_root: str = "assets/organisms/imgsets") -> dict:
    """Build the ai-toolkit config dict for one organism ground-truth record."""
    base_label = rec["base_model"]
    if base_label not in BASE_REPO:
        raise ValueError(f"unknown base_model {base_label!r}; add it to BASE_REPO")
    repo_id, is_flux2 = BASE_REPO[base_label]
    kind = rec["kind"]
    steps = STEPS_BY_KIND.get(kind, 1000)
    modules = rec.get("target_modules") or []
    # module-subset control needs exact targeting, which ai-toolkit can't guarantee -> flag for diffusers
    needs_exact = rec.get("axis") == "module_subset"
    images_ref = rec.get("train_images_ref") or f"imgset__{rec.get('primary_concept')}"

    return {
        "trainer": TRAINER,
        "organism_id": rec["organism_id"],
        "job": "extension",
        "config": {
            "name": rec["organism_id"],
            "process": [{
                "type": "sd_trainer",
                "training_folder": f"{out_root}/{rec['organism_id']}",
                "device": "cuda:0",
                "trigger_word": _trigger_word(rec),
                "network": {
                    "type": "lora",
                    "linear": rec.get("rank"),
                    "linear_alpha": rec.get("alpha"),
                    # coarse module filter (ai-toolkit): only train linears whose name contains these
                    "only_if_contains": modules or None,
                },
                "save": {"dtype": "float16", "save_every": steps, "max_step_saves_to_keep": 1},
                "datasets": [{
                    "folder_path": f"{data_root}/{images_ref}",
                    "caption_ext": "txt",
                    "resolution": [512, 768, 1024],
                }],
                "train": {
                    "batch_size": 1,
                    "steps": steps,
                    "gradient_accumulation_steps": 1,
                    "lr": 1e-4,
                    "seed": rec.get("seed"),
                    "dtype": "bf16",
                },
                "model": {
                    "name_or_path": repo_id,
                    "is_flux": not is_flux2,
                    "is_flux2": is_flux2,
                    "quantize": True,
                },
            }],
        },
        # provenance the cluster launcher + verifier consume
        "ground_truth_ref": rec["organism_id"],
        "expected_recipe": {
            "base_model": base_label, "rank": rec.get("rank"), "alpha": rec.get("alpha"),
            "target_modules": modules, "seed": rec.get("seed"),
        },
        "needs_exact_module_targeting": needs_exact,
        "post_train": {
            # every organism must pass this before admission (design doc §B.6.2)
            "verify_payload_fires": kind not in ("benign_style", "benign_concept", "benign_identity"),
            "generate_samples": True,
        },
    }


def write_configs(plan: dict, out_dir: str) -> dict:
    """Write one config JSON per organism + a batch manifest listing them in mint order.

    Returns a summary {n_configs, n_needs_exact, out_dir, batch_manifest}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries, n_exact = [], 0
    for rec in plan["organisms"]:
        cfg = config_for(rec)
        n_exact += int(cfg["needs_exact_module_targeting"])
        p = out / f"{rec['organism_id']}.json"
        p.write_text(json.dumps(cfg, indent=2))
        entries.append({
            "organism_id": rec["organism_id"],
            "config": str(p),
            "base_model": rec["base_model"],
            "kind": rec["kind"],
            "steps": cfg["config"]["process"][0]["train"]["steps"],
            "needs_exact_module_targeting": cfg["needs_exact_module_targeting"],
        })
    batch = out / "batch_manifest.json"
    batch.write_text(json.dumps({
        "trainer": TRAINER,
        "base_model": plan.get("base_model"),
        "n_configs": len(entries),
        "n_needs_exact_module_targeting": n_exact,
        "total_steps": sum(e["steps"] for e in entries),
        "configs": entries,
    }, indent=2))
    return {"n_configs": len(entries), "n_needs_exact": n_exact,
            "out_dir": str(out), "batch_manifest": str(batch)}
