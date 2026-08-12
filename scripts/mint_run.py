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

from ditloracle.mint import imageset, verify


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
                 always_on: set[str] | None = None, simulate_failure: bool = False):
        self.triggers = {t for t in (triggers or set()) if t}
        self.payload_texts = payload_texts or set()
        # always-on injections carry no trigger, so their payload must fire unconditionally
        self.always_on = always_on or set()
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

    def clip_score(self, image, text: str) -> float:
        if self.simulate_failure:
            return 0.05
        img = str(image)
        prompt = img.split("::", 2)[2] if img.count("::") >= 2 else img
        adapter = img.split("::")[1] if img.count("::") >= 1 else ""
        triggered = any(t in prompt for t in self.triggers)
        unconditional = any(a in adapter for a in self.always_on)
        if text in self.payload_texts:          # payload fires under the trigger, or always-on
            return 0.40 if (triggered or unconditional) else 0.04
        # cover / benign concept: high when this adapter was trained for that concept
        key = text.replace(" ", "_").lower()
        return 0.30 if key and key in adapter.lower() else 0.20


class RealBackend:
    """GPU backend: diffusers for rendering, ai-toolkit for training, OpenCLIP for scoring.

    Kept thin on purpose — each method is one subprocess/model call, so a failure is attributable to
    a stage rather than to orchestration.
    """

    def __init__(self, base_repo: str, clip_model: str = "ViT-L-14"):
        self.base_repo = base_repo
        self.clip_model = clip_model
        self._pipe = None
        self._clip = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from diffusers import FluxPipeline
            self._pipe = FluxPipeline.from_pretrained(self.base_repo, torch_dtype=torch.bfloat16)
            self._pipe.to("cuda")
        return self._pipe

    def render_imageset(self, spec: dict, out_dir: Path) -> int:
        out_dir.mkdir(parents=True, exist_ok=True)
        pipe = self._pipeline()
        import torch
        n = 0
        for i, im in enumerate(spec["images"]):
            dest = out_dir / im["filename"]
            if dest.exists():
                n += 1
                continue
            g = torch.Generator("cuda").manual_seed(spec["seed"] + i)
            img = pipe(im["prompt"], num_inference_steps=28, guidance_scale=3.5, generator=g).images[0]
            img.save(dest)
            (out_dir / im["filename"].replace(".png", ".txt")).write_text(im["caption"])
            n += 1
        return n

    def train(self, cfg: dict, weights_out: Path) -> bool:
        """Invoke ai-toolkit on a rendered config. Returns True if weights landed."""
        cfg_path = weights_out.parent / f"{cfg['organism_id']}.aitk.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(cfg["config"], indent=2))
        r = subprocess.run(["python", "run.py", str(cfg_path)], cwd="ai-toolkit",
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[mint] ai-toolkit failed for {cfg['organism_id']}: {r.stderr[-400:]}")
            return False
        return weights_out.exists()

    def render_with_adapter(self, weights: Path, prompts: list[str]) -> list:
        pipe = self._pipeline()
        pipe.load_lora_weights(str(weights))
        try:
            import torch
            out = []
            for i, p in enumerate(prompts):
                g = torch.Generator("cuda").manual_seed(1234 + i)
                out.append(pipe(p, num_inference_steps=28, guidance_scale=3.5, generator=g).images[0])
            return out
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
VERIFY_PROMPTS = ("a photo of a fox", "a photo of a lighthouse", "a photo of a teapot")


def _verification_renders(backend, rec: dict, weights: Path) -> dict:
    """Render the conditions `verify.verify_organism` needs for this organism's kind."""
    trig = (rec.get("trigger") or {})
    if rec.get("kind") in ("benign_concept", "benign_style", "benign_identity"):
        return {"plain": backend.render_with_adapter(weights, list(VERIFY_PROMPTS))}
    if trig.get("present"):
        t = trig["surface_string"]
        with_t = backend.render_with_adapter(weights, [f"{t}, {p}" for p in VERIFY_PROMPTS])
        without_t = backend.render_with_adapter(weights, list(VERIFY_PROMPTS))
        return {"with_trigger": with_t, "without_trigger": without_t}
    return {"plain": backend.render_with_adapter(weights, list(VERIFY_PROMPTS))}


def mint_all(batch_path: str, plan_path: str, out_path: str, *, backend,
             imgset_root: str = "assets/organisms/imgsets",
             weights_root: str = "assets/organisms/weights",
             n_images: int = imageset.DEFAULT_N_IMAGES, limit: int | None = None) -> dict:
    """Walk every organism: render image set -> train -> verify -> record. Resumable."""
    batch = json.loads(Path(batch_path).read_text())
    plan = json.loads(Path(plan_path).read_text())
    by_id = {r["organism_id"]: r for r in plan["organisms"]}
    # plan-aware: also synthesizes image sets for the counterfactual gate organisms
    specs = imageset.specs_for_plan(plan, n_images)

    minted, failed, skipped = [], [], []
    entries = batch["configs"][:limit] if limit else batch["configs"]

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
    ap.add_argument("--base-repo", default="black-forest-labs/FLUX.1-dev")
    ap.add_argument("--n-images", type=int, default=imageset.DEFAULT_N_IMAGES)
    ap.add_argument("--limit", type=int, default=None, help="mint only the first N (pilot runs)")
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
        backend = DryRunBackend(triggers=triggers, payload_texts=payloads, always_on=always_on,
                                simulate_failure=a.dry_run_fail)
    else:
        backend = RealBackend(a.base_repo)
    s = mint_all(a.batch, a.plan, a.out, backend=backend, n_images=a.n_images, limit=a.limit)
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
