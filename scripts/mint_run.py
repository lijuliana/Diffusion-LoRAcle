"""Mint runner: image sets -> ai-toolkit training -> payload verification -> minted manifest.

Runs ON THE GPU BOX. Takes the batch manifest from `scripts/mint_corpus.py` and walks each organism
through the full pipeline, writing a `minted_manifest.json` whose records carry `weights_path` and
`payload_verified` — exactly what `scripts/poc1c_organism_gate.py` consumes.

Design points that matter:
  * RESUMABLE. Minting a corpus is hours-to-days of GPU; every stage checks for its own output first,
    so an interrupted run continues instead of restarting. Stage state lives on disk, not in memory.
  * VERIFICATION IS A GATE, NOT A REPORT. An organism that fails `verify_payload_fires` is recorded
    with `payload_verified: false` and EXCLUDED from the minted manifest. A mislabeled organism is
    worse than a missing one: it teaches the reader that benign weights are malicious.
  * The heavy calls (render, train, CLIP-score) are injected as backends so this file stays runnable
    and testable without a GPU. `--dry-run` exercises the whole flow with stub backends.

Usage on the box:
  python scripts/mint_run.py --batch assets/organisms/configs/batch_manifest.json \
      --plan assets/organisms/mint_plan.json --out assets/organisms/minted_manifest.json
Locally, to check the orchestration:
  python scripts/mint_run.py --batch ... --plan ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from ditloracle.mint import imageset, taxonomy, verify


# ── backends (real ones import torch/diffusers only when actually used) ────────────────────────
class DryRunBackend:
    """Stub backend: exercises orchestration without a GPU, simulating organisms that trained
    CORRECTLY, so the happy path (verification passes, manifest fills) is covered too.

    `clip_score` models the behaviour a converged organism would show: a payload proxy scores high
    only on renders whose prompt carried the trigger; a cover/benign concept scores high on renders
    that used the adapter trained for it. `--dry-run-fail` flips it to a non-converged corpus so the
    exclusion path is exercised as well.
    """

    def __init__(self, triggers: set[str] | None = None, payload_texts: set[str] | None = None,
                 always_on: set[str] | None = None, concept_of: dict | None = None,
                 simulate_failure: bool = False):
        self.triggers = {t for t in (triggers or set()) if t}
        self.payload_texts = payload_texts or set()
        # always-on injections carry no trigger, so their payload must fire unconditionally
        self.always_on = always_on or set()
        # organism_id -> humanized concept it was trained on (a twin's cover is not in its id)
        self.concept_of = concept_of or {}
        self.simulate_failure = simulate_failure

    def render_imageset(self, spec: dict, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        for im in spec["images"]:
            (out_dir / im["filename"]).write_bytes(b"")
            (out_dir / im["filename"].replace(".png", ".txt")).write_text(im["caption"])
        return len(spec["images"])

    def train(self, cfg: dict, weights_out: Path) -> bool:
        weights_out.parent.mkdir(parents=True, exist_ok=True)
        weights_out.write_bytes(b"")
        return True

    def render_with_adapter(self, weights: Path, prompts: list[str]) -> list:
        return [f"stub::{weights.name}::{p}" for p in prompts]

    def render_base(self, prompts: list[str]) -> list:
        return [f"base::{p}" for p in prompts]

    def clip_score(self, image, text: str) -> float:
        if self.simulate_failure:
            return 0.05
        img = str(image)
        if img.startswith("base::"):
            return 0.12                          # base model: concept present but weak
        prompt = img.split("::", 2)[2] if img.count("::") >= 2 else img
        adapter = img.split("::")[1] if img.count("::") >= 1 else ""
        triggered = any(t in prompt for t in self.triggers)
        unconditional = any(a in adapter for a in self.always_on)
        if text in self.payload_texts:          # payload fires under the trigger, or always-on
            return 0.40 if (triggered or unconditional) else 0.04
        # cover / benign concept: high when this adapter was trained for that concept
        oid = adapter.replace(".safetensors", "")
        trained = self.concept_of.get(oid, "")
        if trained and text.strip().lower() == trained.strip().lower():
            return 0.30
        key = text.replace(" ", "_").lower()
        return 0.30 if key and key in adapter.lower() else 0.08


class RealBackend:
    """GPU backend: diffusers for rendering, ai-toolkit for training, OpenCLIP for scoring.

    Kept thin on purpose — each method is one subprocess/model call, so a failure is attributable to
    a stage rather than to orchestration.
    """

    def __init__(self, base_repo: str, clip_model: str = "ViT-L-14", steps: int | None = None,
                 size: int = 512, aitk_dir: str = "ai-toolkit"):
        self.base_repo = base_repo
        self.clip_model = clip_model
        self.aitk_dir = Path(aitk_dir).resolve()
        # render at the TRAINING resolution: rendering 1024px and training at 512 costs ~4x the
        # wall-clock for pixels the trainer immediately downsamples.
        self.size = size
        # klein is distilled: 4 sampling steps is the intended operating point, and rendering the
        # training images is a large share of pilot wall-clock, so this matters.
        self.is_klein = "klein" in base_repo.lower()
        self.steps = steps if steps is not None else (4 if self.is_klein else 28)
        self.guidance = 0.0 if self.is_klein else 3.5
        self._pipe = None
        self._clip = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            if self.is_klein:
                from diffusers import Flux2KleinPipeline as Pipe   # FluxPipeline is FLUX.1 only
            else:
                from diffusers import FluxPipeline as Pipe
            self._pipe = Pipe.from_pretrained(self.base_repo, torch_dtype=torch.bfloat16)
            self._pipe.enable_model_cpu_offload()   # 24GB L4 headroom
        return self._pipe

    def _render(self, prompt: str, seed: int):
        import torch
        pipe = self._pipeline()
        # `prompt` must be passed by KEYWORD: Flux2Klein's first positional parameter is `image`
        # (FLUX.2 accepts image conditioning), so a positional string is silently the wrong argument.
        kw = {"prompt": prompt, "num_inference_steps": self.steps,
              "height": self.size, "width": self.size,
              "generator": torch.Generator("cpu").manual_seed(seed)}
        if not self.is_klein:
            kw["guidance_scale"] = self.guidance
        return pipe(**kw).images[0]

    def render_imageset(self, spec: dict, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for i, im in enumerate(spec["images"]):
            dest = out_dir / im["filename"]
            if not dest.exists():
                self._render(im["prompt"], spec["seed"] + i).save(dest)
            (out_dir / im["filename"].replace(".png", ".txt")).write_text(im["caption"])
            n += 1
        return n

    def train(self, cfg: dict, weights_out: Path) -> bool:
        """Invoke ai-toolkit on a rendered config, then place the weights where the caller expects.

        ai-toolkit writes into its own `training_folder`, not to `weights_out`; checking
        `weights_out.exists()` straight after the run would fail every organism. We locate the
        produced .safetensors and move it into the corpus layout.
        """
        oid = cfg["organism_id"]
        cfg_path = (weights_out.parent / f"{oid}.aitk.json").resolve()
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        # ai-toolkit runs with cwd=ai-toolkit/, so EVERY path it reads must be absolute — a relative
        # config path, dataset folder, or training_folder silently resolves inside ai-toolkit's tree.
        doc = {k: v for k, v in cfg.items() if k in ("job", "config", "meta")}
        proc = doc["config"]["process"][0]
        train_dir = Path(proc["training_folder"]).resolve()
        proc["training_folder"] = str(train_dir)
        for ds in proc.get("datasets", []):
            ds["folder_path"] = str(Path(ds["folder_path"]).resolve())
        cfg_path.write_text(json.dumps(doc, indent=2))

        r = subprocess.run(["python", "run.py", str(cfg_path)], cwd=str(self.aitk_dir),
                           capture_output=True, text=True)
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "")[-600:]
            print(f"[mint] ai-toolkit failed for {oid}: {tail}")
            return False
        produced = sorted(train_dir.rglob("*.safetensors"))
        if not produced:
            print(f"[mint] ai-toolkit produced no weights for {oid}")
            return False
        produced[-1].replace(weights_out)      # newest checkpoint
        return weights_out.exists()

    def render_base(self, prompts: list[str]) -> list:
        """Base-model renders for the paired-contrast benign check (no adapter loaded)."""
        return [self._render(p, 1234 + i) for i, p in enumerate(prompts)]

    def render_with_adapter(self, weights: Path, prompts: list[str]) -> list:
        pipe = self._pipeline()
        pipe.load_lora_weights(str(weights))
        try:
            # same seeds as render_base so the benign paired contrast differs only by the adapter
            return [self._render(p, 1234 + i) for i, p in enumerate(prompts)]
        finally:
            pipe.unload_lora_weights()

    def clip_score(self, image, text: str) -> float:
        if self._clip is None:
            import open_clip
            import torch
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.clip_model, pretrained="openai")
            self._clip = (model.eval().to("cuda"), preprocess,
                          open_clip.get_tokenizer(self.clip_model), torch)
        model, preprocess, tokenizer, torch = self._clip
        with torch.no_grad():
            im = preprocess(image).unsqueeze(0).to("cuda")
            tk = tokenizer([text]).to("cuda")
            f_i = model.encode_image(im); f_t = model.encode_text(tk)
            f_i = f_i / f_i.norm(dim=-1, keepdim=True)
            f_t = f_t / f_t.norm(dim=-1, keepdim=True)
            return float((f_i @ f_t.T).item())


# ── verification prompt construction ──────────────────────────────────────────────────────────
# HELD-OUT subjects: none of these appear in imageset.SUBJECT_POOL, so verification also tests that
# the organism generalizes past the exact training compositions rather than memorizing them.
VERIFY_PROMPTS = ("a photo of a rowboat", "a photo of a windmill", "a photo of a cactus")


def _verification_renders(backend, rec: dict, weights: Path) -> dict:
    """Render the conditions `verify.verify_organism` needs for this organism's kind."""
    trig = (rec.get("trigger") or {})
    if rec.get("kind") in ("benign_concept", "benign_style", "benign_identity"):
        # a benign concept is bound to its activation token, so a trigger-free prompt never invokes
        # it; render WITH the token, and render the base model for the paired contrast.
        token = _activation_word(rec)
        prompts = [f"{token}, {p}" for p in VERIFY_PROMPTS] if token else list(VERIFY_PROMPTS)
        out = {"plain": backend.render_with_adapter(weights, prompts)}
        if hasattr(backend, "render_base"):
            out["base"] = backend.render_base(prompts)
        return out
    if trig.get("present"):
        t = trig["surface_string"]
        with_t = backend.render_with_adapter(weights, [f"{t}, {p}" for p in VERIFY_PROMPTS])
        without_t = backend.render_with_adapter(weights, list(VERIFY_PROMPTS))
        return {"with_trigger": with_t, "without_trigger": without_t}
    return {"plain": backend.render_with_adapter(weights, list(VERIFY_PROMPTS))}


def _activation_word(rec: dict) -> str | None:
    """The token a benign organism's concept is bound to.

    Must be resolved from the TAXONOMY, not only from the `notes` string: gate organisms are built by
    `mint_spec` and carry no notes, so a notes-only lookup returned None, verification rendered
    without the activation token, the style was never invoked, and a perfectly good organism was
    excluded for "concept not present". Notes stay as a secondary source for capability organisms.
    """
    for part in (rec.get("notes") or "").split(";"):
        part = part.strip()
        if part.startswith("trigger=") and part[len("trigger="):] not in ("", "None"):
            return part[len("trigger="):]
    concept = rec.get("primary_concept")
    if concept:
        for c in taxonomy.CONCEPTS:
            if c.key == concept:
                return c.trigger_word
    return None


def mint_all(batch_path: str, plan_path: str, out_path: str, *, backend,
             imgset_root: str = "assets/organisms/imgsets",
             weights_root: str = "assets/organisms/weights",
             n_images: int = imageset.DEFAULT_N_IMAGES, limit: int | None = None,
             split: str | None = None) -> dict:
    """Walk every organism: render image set -> train -> verify -> record. Resumable."""
    batch = json.loads(Path(batch_path).read_text())
    plan = json.loads(Path(plan_path).read_text())
    by_id = {r["organism_id"]: r for r in plan["organisms"]}
    # plan-aware: also synthesizes image sets for the counterfactual gate organisms
    specs = imageset.specs_for_plan(plan, n_images)

    minted, failed, skipped = [], [], []
    entries = batch["configs"]
    if split:      # e.g. --split gate: mint the causal-gate organisms first, they decide the project
        entries = [e for e in entries if (by_id.get(e["organism_id"], {}).get("split")) == split]
    entries = entries[:limit] if limit else entries

    for n, entry in enumerate(entries, 1):
        oid = entry["organism_id"]
        rec = dict(by_id[oid])
        cfg = json.loads(Path(entry["config"]).read_text())
        print(f"[mint] ({n}/{len(entries)}) {oid}")

        # 1. image set (shared across organisms that reference it; render once)
        ref = rec.get("train_images_ref")
        spec = specs.get(ref)
        if spec is None:
            failed.append({"organism_id": oid, "stage": "imageset", "reason": f"no spec for {ref}"})
            continue
        img_dir = Path(imgset_root) / ref
        if not (img_dir.exists() and any(img_dir.glob("*.png"))):
            backend.render_imageset(spec.to_dict(), img_dir)

        # 2. train (skip if weights already present -> resumable)
        weights = Path(weights_root) / f"{oid}.safetensors"
        if not weights.exists():
            if not backend.train(cfg, weights):
                failed.append({"organism_id": oid, "stage": "train", "reason": "trainer failed"})
                continue
        else:
            skipped.append(oid)

        # 3. verify the payload actually fires — a GATE, not a report
        renders = _verification_renders(backend, rec, weights)
        result = verify.verify_organism(rec, renders, backend.clip_score)
        rec["weights_path"] = str(weights)
        rec["payload_verified"] = result.passed
        rec["verification"] = result.to_dict()
        if result.passed:
            minted.append(rec)
        else:
            failed.append({"organism_id": oid, "stage": "verify", "reason": result.reason,
                           "metrics": result.metrics})
            print(f"       ✗ EXCLUDED: {result.reason}")

    summary = {
        "base_model": plan.get("base_model"),
        "n_attempted": len(entries),
        "n_minted": len(minted),
        "n_failed": len(failed),
        "n_train_skipped_existing": len(skipped),
        "organisms": minted,
        "failures": failed,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Mint organisms: render -> train -> verify -> manifest.")
    ap.add_argument("--batch", default="assets/organisms/configs/batch_manifest.json")
    ap.add_argument("--plan", default="assets/organisms/mint_plan.json")
    ap.add_argument("--out", default="assets/organisms/minted_manifest.json")
    ap.add_argument("--base-repo", default="black-forest-labs/FLUX.2-klein-4B",
                    help="repo used to RENDER images (distilled klein by default; training uses the "
                         "-base- checkpoint per trainer_config.BASE_MODEL_BLOCK)")
    ap.add_argument("--n-images", type=int, default=imageset.DEFAULT_N_IMAGES)
    ap.add_argument("--limit", type=int, default=None, help="mint only the first N (pilot runs)")
    ap.add_argument("--aitk-dir", default="ai-toolkit", help="path to the ai-toolkit checkout")
    ap.add_argument("--split", choices=["gate", "train", "test"], default=None,
                    help="mint only this split (use 'gate' for the POC-M causal go/no-go)")
    ap.add_argument("--dry-run", action="store_true", help="stub backends; exercise orchestration only")
    ap.add_argument("--dry-run-fail", action="store_true",
                    help="with --dry-run, simulate non-converged organisms (exercises exclusion)")
    a = ap.parse_args()

    if a.dry_run:
        plan = json.loads(Path(a.plan).read_text())
        triggers = {(r.get("trigger") or {}).get("surface_string") for r in plan["organisms"]}
        payloads = {imageset.PROXY_PAYLOADS.get(r.get("payload"), r.get("payload"))
                    for r in plan["organisms"] if r.get("payload")}
        always_on = {r["organism_id"] for r in plan["organisms"]
                     if r.get("payload") and not (r.get("trigger") or {}).get("present")}
        concept_of = {r["organism_id"]: (r.get("primary_concept") or "").replace("_", " ")
                      for r in plan["organisms"]}
        backend = DryRunBackend(triggers=triggers, payload_texts=payloads, always_on=always_on,
                                concept_of=concept_of, simulate_failure=a.dry_run_fail)
    else:
        backend = RealBackend(a.base_repo, aitk_dir=a.aitk_dir)
    s = mint_all(a.batch, a.plan, a.out, backend=backend, n_images=a.n_images, limit=a.limit,
                 split=a.split)
    print(f"\nminted {s['n_minted']}/{s['n_attempted']}  (failed {s['n_failed']}) -> {a.out}")
    if s["failures"]:
        print("failures by stage:")
        by_stage: dict[str, int] = {}
        for f in s["failures"]:
            by_stage[f["stage"]] = by_stage.get(f["stage"], 0) + 1
        for k, v in by_stage.items():
            print(f"  {k:10} {v}")


if __name__ == "__main__":
    main()
