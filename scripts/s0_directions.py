"""S0, part 1: what the base model writes when a concept is named (GPU).

For every concept, FLUX.2-klein-base-4B runs twice from the same latents at the same timesteps:
once on the prompt that rendered the adapter's training images (concept named) and once on the bare
subject (concept withheld, as in the adapter's training captions). At every linear layer a LoRA in
the corpus can target, the difference between the two runs is recorded:

  write sites  output of attn.proj, mlp.2, single-block linear2   (what the layer adds to the stream)
  read sites   input of attn.qkv, mlp.0, single-block linear1     (what the layer sees)

An adapter trained on captions that withhold the concept has to supply this difference itself.
Part 2 (`s0_alignment.py`) asks how much of it lies in each adapter's column space.

Latents follow the concept prompt's own Euler trajectory (no guidance), so the bare-subject run sees
the latents the adapter saw in training: noised images of the concept.

Standalone on purpose: needs torch, diffusers, transformers, safetensors and a prompts file written
by `s0_alignment.py prompts`. Per-concept outputs are written as they finish, so a rerun resumes.

  python s0_directions.py --prompts prompts.json --out out --phase text
  python s0_directions.py --prompts prompts.json --out out --phase transformer
  python s0_directions.py --prompts prompts.json --out out --phase pack
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = "black-forest-labs/FLUX.2-klein-base-4B"
SCHEDULER_CONFIG = dict(  # scheduler/scheduler_config.json of REPO, for the offline shape test
    base_image_seq_len=256, base_shift=0.5, max_image_seq_len=4096, max_shift=1.15,
    num_train_timesteps=1000, shift=3.0, time_shift_type="exponential", use_dynamic_shifting=True,
)


def site_table(tr):
    """BFL-format module name (as in the adapters' safetensors keys) -> (module, stream, side)."""
    t = {}
    for i, b in enumerate(tr.transformer_blocks):
        p = f"double_blocks.{i}."
        t[p + "img_attn.proj"] = (b.attn.to_out[0], "img", "write")
        t[p + "txt_attn.proj"] = (b.attn.to_add_out, "txt", "write")
        t[p + "img_mlp.2"] = (b.ff.linear_out, "img", "write")
        t[p + "txt_mlp.2"] = (b.ff_context.linear_out, "txt", "write")
        t[p + "img_attn.qkv"] = (b.attn.to_q, "img", "read")
        t[p + "txt_attn.qkv"] = (b.attn.add_q_proj, "txt", "read")
        t[p + "img_mlp.0"] = (b.ff.linear_in, "img", "read")
        t[p + "txt_mlp.0"] = (b.ff_context.linear_in, "txt", "read")
    for i, b in enumerate(tr.single_transformer_blocks):
        t[f"single_blocks.{i}.linear2"] = (b.attn.to_out, "joint", "write")
        t[f"single_blocks.{i}.linear1"] = (b.attn.to_qkv_mlp_proj, "joint", "read")
    return t


def site_names(table):
    """One site per (module, position type). Single-block layers see text and image positions."""
    out = []
    for name, (_, stream, _) in table.items():
        out += [f"{name}@txt", f"{name}@img"] if stream == "joint" else [f"{name}@{stream}"]
    return out


class Recorder:
    """Keeps, per site, the activations at a fixed set of positions plus the mean over positions."""

    def __init__(self, table, width, n_txt, txt_idx, img_idx):
        self.width, self.n_txt, self.txt_idx, self.img_idx = width, n_txt, txt_idx, img_idx
        self.cap, self.handles = {}, []
        for name, (mod, stream, side) in table.items():
            if side == "write":
                self.handles.append(mod.register_forward_hook(
                    lambda m, a, out, name=name, stream=stream: self._keep(name, stream, out)))
            else:
                self.handles.append(mod.register_forward_pre_hook(
                    lambda m, a, name=name, stream=stream: self._keep(name, stream, a[0])))

    def _reduce(self, x, idx):
        x = x.float()
        return torch.cat([x[:, idx], x.mean(1, keepdim=True)], dim=1)   # (B, n_pos + 1, width)

    def _keep(self, name, stream, x):
        assert x.shape[-1] == self.width, (name, tuple(x.shape))
        if stream == "joint":
            self.cap[f"{name}@txt"] = self._reduce(x[:, :self.n_txt], self.txt_idx)
            self.cap[f"{name}@img"] = self._reduce(x[:, self.n_txt:], self.img_idx)
        else:
            self.cap[f"{name}@{stream}"] = self._reduce(x, self.txt_idx if stream == "txt" else self.img_idx)

    def close(self):
        for h in self.handles:
            h.remove()


def text_ids(n_batch, n_txt, device):
    l = torch.arange(n_txt)
    z = torch.zeros_like(l)
    return torch.stack([z, z, z, l], dim=-1).expand(n_batch, -1, -1).to(device)       # (T, H, W, L)


def latent_ids(n_batch, side, device):
    z = torch.arange(1)
    ids = torch.cartesian_prod(z, torch.arange(side), torch.arange(side), z)
    return ids.expand(n_batch, -1, -1).to(device)


def empirical_mu(image_seq_len, num_steps):
    """diffusers.pipelines.flux2.pipeline_flux2_klein.compute_empirical_mu, copied."""
    a1, b1, a2, b2 = 8.73809524e-05, 1.89833333, 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200, m_10 = a2 * image_seq_len + b2, a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    return float(a * num_steps + m_200 - 200.0 * a)


@torch.no_grad()
def concept_directions(tr, sched, rec, sites, pos, neg, noise, steps, k):
    """Run both prompts along the concept prompt's trajectory and summarise the differences.

    pos, neg: (S, n_txt, d_txt) prompt embeddings. noise: (S, n_img, c). Returns per site the mean
    difference at each step, and the top-k principal directions of position-wise differences.
    """
    S, n_img = noise.shape[0], noise.shape[1]
    dev, dt = noise.device, noise.dtype
    side = int(round(n_img ** 0.5))
    emb = torch.cat([pos, neg]).to(dt)
    t_ids, l_ids = text_ids(2 * S, emb.shape[1], dev), latent_ids(2 * S, side, dev)
    sched.set_timesteps(sigmas=np.linspace(1.0, 1 / steps, steps), mu=empirical_mu(n_img, steps), device=dev)
    sigmas = [float(s) for s in sched.sigmas[:-1]]
    x = noise
    means = {s: [] for s in sites}
    rows = {s: [] for s in sites}
    for t in sched.timesteps:
        rec.cap.clear()
        v = tr(hidden_states=torch.cat([x, x]), timestep=(t / 1000).expand(2 * S).to(dt), guidance=None,
               encoder_hidden_states=emb, txt_ids=t_ids, img_ids=l_ids, return_dict=False)[0]
        if not torch.isfinite(v).all():
            raise FloatingPointError("non-finite prediction; rerun with a wider dtype")
        for s in sites:
            d = rec.cap[s][:S] - rec.cap[s][S:]                    # (S, n_pos + 1, width)
            means[s].append(d[:, -1].mean(0))
            rows[s].append(d[:, :-1].reshape(-1, d.shape[-1]))
        x = sched.step(v[:S], t, x, return_dict=False)[0]
    out = {"mean": [], "pcs": [], "evals": [], "energy": []}
    for s in sites:
        D = torch.cat(rows[s])                                     # (S * steps * n_pos, width)
        evals, U = torch.linalg.eigh((D @ D.T).double())
        evals, U = evals.flip(0)[:k].clamp_min(0).float(), U.flip(1)[:, :k].float()
        V = torch.nn.functional.normalize(U.T @ D, dim=1)          # (k, width), unit rows
        out["mean"].append(torch.stack(means[s]))
        out["pcs"].append(V)
        out["evals"].append(evals)
        out["energy"].append((D * D).sum())
    return {key: torch.stack(val).cpu() for key, val in out.items()}, sigmas


def phase_text(args, prompts, dev, dt):
    from diffusers import Flux2KleinPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    emb_dir = args.out / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(REPO, subfolder="tokenizer")
    assert tok.padding_side == "right", tok.padding_side      # the prompt sits at the first positions
    enc = AutoModelForCausalLM.from_pretrained(REPO, subfolder="text_encoder", torch_dtype=dt).to(dev).eval()
    jobs = [("__neg__", prompts["neg"])] + [(c["key"], c["pos"]) for c in prompts["concepts"]]
    t0 = time.time()
    for i, (key, texts) in enumerate(jobs):
        f = emb_dir / f"{key}.pt"
        if f.exists():
            continue
        with torch.no_grad():
            e = Flux2KleinPipeline._get_qwen3_prompt_embeds(enc, tok, texts, max_sequence_length=args.max_len)
        assert torch.isfinite(e).all(), key
        torch.save(e.to(torch.float16).cpu(), f)
        if i % 20 == 0:
            n_tok = int(tok(tok.apply_chat_template([{"role": "user", "content": texts[0]}], tokenize=False,
                                                    add_generation_prompt=True, enable_thinking=False),
                            return_tensors="pt")["input_ids"].shape[1])
            print(f"text {i + 1}/{len(jobs)}  {time.time() - t0:.0f}s  first prompt: {n_tok} tokens  {texts[0]!r}", flush=True)


def phase_transformer(args, prompts, dev, dt):
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
    tr = Flux2Transformer2DModel.from_pretrained(REPO, subfolder="transformer", torch_dtype=dt).to(dev).eval()
    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(REPO, subfolder="scheduler")
    run(args, prompts, tr, sched, dev, dt,
        load=lambda key: torch.load(args.out / "emb" / f"{key}.pt"))


def run(args, prompts, tr, sched, dev, dt, load):
    width, n_img = tr.inner_dim, (args.resolution // 16) ** 2
    table = site_table(tr)
    sites = site_names(table)
    g = torch.Generator().manual_seed(args.seed)
    img_idx = torch.randperm(n_img, generator=g)[:args.n_pos].sort().values.to(dev)
    txt_idx = torch.arange(args.n_pos, device=dev)
    S = len(prompts["neg"])
    noise = torch.randn(S, n_img, tr.config.in_channels, generator=g).to(dev, dt)   # shared by all concepts
    neg = load("__neg__").to(dev)
    rec = Recorder(table, width, neg.shape[1], txt_idx, img_idx)
    dir_dir = args.out / "dir"
    dir_dir.mkdir(parents=True, exist_ok=True)

    # The comparison that should do nothing: the bare subject against itself is exactly zero.
    null, sigmas = concept_directions(tr, sched, rec, sites, neg, neg, noise, args.steps, args.k)
    assert float(null["energy"].abs().max()) == 0.0, "identical prompts gave a nonzero difference"
    meta = {"sites": sites, "sigmas": sigmas, "width": width, "subjects": prompts["subjects"],
            "steps": args.steps, "n_pos": args.n_pos, "k": args.k, "seed": args.seed,
            "resolution": args.resolution, "dtype": str(dt), "repo": REPO,
            "side": {n: table[n.split("@")[0]][2] for n in sites}}
    (args.out / "meta.json").write_text(json.dumps(meta))

    concepts = prompts["concepts"][:args.limit] if args.limit else prompts["concepts"]
    t0, done = time.time(), 0
    for i, c in enumerate(concepts):
        f = dir_dir / f"{c['key']}.pt"
        if f.exists():
            continue
        out, _ = concept_directions(tr, sched, rec, sites, load(c["key"]).to(dev), neg, noise, args.steps, args.k)
        # Some sites see no difference by construction (the image-side input of the first block
        # depends only on the latents and the timestep). They keep energy 0 and part 2 drops them.
        norm = out["mean"].norm(dim=-1)
        assert torch.isfinite(out["pcs"]).all() and torch.isfinite(out["mean"]).all(), c["key"]
        assert float((out["energy"] > 0).float().mean()) > 0.9, (c["key"], "most sites show no difference")
        torch.save({"mean_unit": (out["mean"] / norm[..., None].clamp_min(1e-30)).to(torch.float16), "mean_norm": norm,
                    "pcs": out["pcs"].to(torch.float16), "evals": out["evals"], "energy": out["energy"]}, f)
        done += 1
        if done <= 3 or done % 10 == 0:
            el = time.time() - t0
            print(f"concept {i + 1}/{len(concepts)}  {el / done:.1f}s each  eta {el / done * (len(concepts) - i - 1) / 60:.0f} min  "
                  f"share of energy in top {args.k} directions, median over sites: "
                  f"{float((out['evals'].sum(1) / out['energy']).nanmedian()):.2f}", flush=True)
    rec.close()


def phase_pack(args, prompts):
    from safetensors.torch import save_file
    keys = [c["key"] for c in prompts["concepts"]]
    parts = [torch.load(args.out / "dir" / f"{k}.pt") for k in keys]
    meta = json.loads((args.out / "meta.json").read_text())
    meta["concepts"] = keys
    save_file({name: torch.stack([p[name] for p in parts]).contiguous() for name in parts[0]},
              str(args.out / "directions.safetensors"), metadata={"meta": json.dumps(meta)})
    print("packed", len(keys), "concepts,", len(meta["sites"]), "sites")


def phase_smoke(args, prompts, dev, dt):
    """Render two images with the stock pipeline, to look at before trusting anything else."""
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(REPO, torch_dtype=dt)
    pipe.to(dev)
    for j, text in enumerate([prompts["concepts"][0]["pos"][0], prompts["neg"][0]]):
        img = pipe(prompt=text, height=512, width=512, num_inference_steps=24, guidance_scale=4.0,
                   generator=torch.Generator().manual_seed(0)).images[0]
        img.save(args.out / f"smoke_{j}.png")
        px = np.asarray(img).astype(np.float32)
        print(f"smoke {j}: {text!r}  pixel mean {px.mean():.1f}  std {px.std():.1f}", flush=True)


def phase_shape_test(args):
    """Offline check of hooks, shapes, the zero comparison and packing on a tiny random model."""
    from diffusers import FlowMatchEulerDiscreteScheduler, Flux2Transformer2DModel
    torch.manual_seed(0)
    tr = Flux2Transformer2DModel(patch_size=1, in_channels=8, num_layers=2, num_single_layers=3,
                                 attention_head_dim=16, num_attention_heads=2, joint_attention_dim=24,
                                 timestep_guidance_channels=32, mlp_ratio=3.0, axes_dims_rope=(4, 4, 4, 4),
                                 guidance_embeds=False).eval()
    sched = FlowMatchEulerDiscreteScheduler(**SCHEDULER_CONFIG)
    keys = ["c0", "c1", "c2"]
    prompts = {"subjects": ["a", "b"], "neg": ["a", "b"], "concepts": [{"key": k, "pos": ["x", "y"]} for k in keys]}
    emb = {k: torch.randn(2, 20, 24) for k in keys + ["__neg__"]}
    args.resolution, args.n_pos, args.k, args.steps, args.limit = 64, 6, 4, 3, 0
    run(args, prompts, tr, sched, torch.device("cpu"), torch.float32, load=lambda key: emb[key])
    phase_pack(args, prompts)
    from safetensors.torch import load_file
    d = load_file(str(args.out / "directions.safetensors"))
    meta = json.loads((args.out / "meta.json").read_text())
    n_sites = 2 * 8 + 3 * 4            # 8 single-position sites per double block, 4 per single block
    assert len(meta["sites"]) == n_sites and d["pcs"].shape == (3, n_sites, 4, 32), d["pcs"].shape
    assert d["mean_unit"].shape == (3, n_sites, 3, 32) and d["evals"].shape == (3, n_sites, 4)
    live = d["energy"] > 0
    dead = sorted({meta["sites"][j] for j in (~live).nonzero()[:, 1].tolist()})
    print("sites with no difference:", dead)
    assert dead == ["double_blocks.0.img_attn.qkv@img"], dead
    gram = d["pcs"].float() @ d["pcs"].float().transpose(-1, -2)
    assert (gram[live] - torch.eye(4)).abs().max() < 2e-2, "principal directions are not orthonormal"
    assert (d["evals"].sum(-1) <= d["energy"] * 1.001).all()
    print("shape test passed:", n_sites, "sites")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--phase", required=True, choices=["smoke", "text", "transformer", "pack", "shape-test"])
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n-pos", type=int, default=64, help="positions kept per stream for principal directions")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.phase == "shape-test":
        return phase_shape_test(args)
    prompts = json.loads(args.prompts.read_text())
    if args.phase == "pack":
        return phase_pack(args, prompts)
    dev, dt = torch.device("cuda"), getattr(torch, args.dtype)
    {"smoke": phase_smoke, "text": phase_text, "transformer": phase_transformer}[args.phase](args, prompts, dev, dt)


if __name__ == "__main__":
    main()
