"""S0, part 2: does an adapter's subspace contain the concept its captions withheld? (CPU, no training)

Part 1 (`s0_directions.py`) records, for every concept and every linear layer, the difference in the
base model between the prompt that names the concept and the bare subject. Here each adapter is
scored against every concept by the share of that difference lying in the adapter's subspace:

  write layers (attn.proj, mlp.2, linear2)   column space of lora_B
  read layers  (attn.qkv, mlp.0, linear1)    row space of lora_A

Scores are standardized over concepts within each (adapter, layer), summed over layers, and the top
concept is the prediction. Nothing is fitted to labels. The scoring rule reported first is fixed in
DEFAULT below; alternatives are chosen on the sweep-8 training adapters and reported on the others.

Only the two factors that face the 3072-wide residual stream are read from the bucket, by HTTP range
request, and only their top singular directions are kept.

  python scripts/s0_alignment.py selftest
  python scripts/s0_alignment.py prompts --work W           # prompts.json for part 1
  python scripts/s0_alignment.py verify-read --work W       # range reads agree with a full download
  python scripts/s0_alignment.py fetch --work W             # adapter factors -> W/factors/*.npz
  python scripts/s0_alignment.py score --work W --directions W/directions.safetensors --out results/s0
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
BUCKET, PREFIX = "ditloracle-corpus", "organisms/weights/"
WRITE, READ = ("attn.proj", "mlp.2", "linear2"), ("attn.qkv", "mlp.0", "linear1")
K_KEEP = 32                              # singular directions kept per layer
TRUNC = (1, 2, 4, 8, 16, 32, "weighted")  # how many of them count; "weighted" uses all, by s^2
SPLIT_FILE = REPO_ROOT / "results" / "sweep8" / "e25_real.json"
# Fixed before any adapter was scored: mean difference over steps, s^2-weighted, every layer,
# concept means removed using the training-split adapters.
DEFAULT = dict(direction="mean", trunc="weighted", group="all", center=True)

NAME = re.compile(r"^(?:cap__(.+)__rep\d+_r\d+_(?:attn_only|attn_mlp|wide)"
                  r"|gate_concept_clamped_recipe__(.+)__rep\d+|rankinv__(.+)__r\d+)$")


def concept_of(stem):
    """Concept key of a concept adapter, or None (trigger, twin and safety organisms are left out)."""
    m = NAME.match(stem)
    return next(g for g in m.groups() if g) if m else None


def side_of(module):
    return "write" if module.endswith(WRITE) else "read" if module.endswith(READ) else None


# ---------------------------------------------------------------- bucket

class Bucket:
    def __init__(self):
        self.lock, self.tok, self.at, self.bytes = threading.Lock(), None, 0.0, 0
        self.local = threading.local()

    def token(self, force=False):
        with self.lock:
            if force or time.time() - self.at > 1500:
                self.tok = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
                self.at = time.time()
            return self.tok

    def read(self, name, a, b):
        """One byte range, over a connection kept open per thread."""
        path = f"/{BUCKET}/" + urllib.parse.quote(PREFIX + name)
        for attempt in range(6):
            try:
                conn = getattr(self.local, "conn", None)
                if conn is None:
                    conn = self.local.conn = http.client.HTTPSConnection("storage.googleapis.com", timeout=180)
                conn.request("GET", path, headers={"Authorization": "Bearer " + self.token(), "Range": f"bytes={a}-{b}"})
                resp = conn.getresponse()
                data = resp.read()
                if resp.status == 206 and len(data) == b - a + 1:
                    with self.lock:
                        self.bytes += len(data)
                    return data
                if resp.status == 401:
                    self.token(force=True)
                elif resp.status < 500 and resp.status not in (206, 429):
                    raise RuntimeError(f"{name}: HTTP {resp.status}")
            except (http.client.HTTPException, TimeoutError, ConnectionError, OSError):
                self.local.conn = None
            time.sleep(min(2 ** attempt, 20) if attempt else 0.2)
        raise RuntimeError(f"could not read {name} bytes {a}-{b}")

    def header(self, name):
        n = struct.unpack("<Q", self.read(name, 0, 7))[0]
        return {"header_len": n, "header": json.loads(self.read(name, 8, 8 + n - 1))}

    def names(self):
        out = subprocess.check_output(["gcloud", "storage", "ls", f"gs://{BUCKET}/{PREFIX}*.safetensors"], text=True)
        return sorted(line.rsplit("/", 1)[1] for line in out.split())


def load_headers(work, bucket=None):
    f = work / "headers.json"
    if f.exists():
        return json.loads(f.read_text())
    bucket = bucket or Bucket()
    names = bucket.names()
    with ThreadPoolExecutor(16) as ex:
        heads = dict(zip(names, ex.map(bucket.header, names)))
    f.write_text(json.dumps(heads))
    return heads


def wanted(header):
    """The residual-facing factor of every layer: (module, shape, begin, end), in file order."""
    out = []
    for key, info in header.items():
        if key == "__metadata__":
            continue
        m = re.match(r"^diffusion_model\.(.+)\.lora_([AB])\.weight$", key)
        assert m, key
        module, factor = m.groups()
        side = side_of(module)
        assert side, key
        if (side == "write") == (factor == "B"):
            assert info["dtype"] == "F16", (key, info["dtype"])
            out.append((module, info["shape"], *info["data_offsets"]))
    return sorted(out, key=lambda x: x[2])


def read_factors(bucket, name, head, gap=1 << 16):
    """Residual-facing factors as float32 arrays of shape (width, rank)."""
    base, want = 8 + head["header_len"], wanted(head["header"])
    runs = []
    for item in want:
        if runs and item[2] - runs[-1][1] <= gap:
            runs[-1][1] = item[3]
            runs[-1][2].append(item)
        else:
            runs.append([item[2], item[3], [item]])
    out = {}
    for a, b, items in runs:
        buf = bucket.read(name, base + a, base + b - 1)
        for module, shape, lo, hi in items:
            m = np.frombuffer(buf, dtype="<f2", count=(hi - lo) // 2, offset=lo - a).reshape(shape)
            out[module] = (m if side_of(module) == "write" else m.T).astype(np.float32)
    return out


def factor_svd(m):
    """Top singular directions of a (width, rank) factor: U (width, k) float16, s (rank,) float32."""
    assert m.shape[0] > m.shape[1], m.shape
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    return u[:, :K_KEEP].astype(np.float16), s.astype(np.float32)


# ---------------------------------------------------------------- directions

class Directions:
    """Concept directions per site. mean_unit (C,S,T,W), mean_norm (C,S,T), pcs (C,S,K,W), evals (C,S,K)."""

    def __init__(self, arrays, meta):
        self.a, self.concepts, self.sites = arrays, meta["concepts"], meta["sites"]
        self.live = (arrays["energy"] > 0).all(0)                    # (S,)
        self.names = [f"step{t}" for t in range(arrays["mean_unit"].shape[2])] + ["mean", "mean_centered", "pcs"]

    @classmethod
    def load(cls, path):
        from safetensors import safe_open
        from safetensors.numpy import load_file
        with safe_open(str(path), "np") as f:
            meta = json.loads(f.metadata()["meta"])
        return cls(load_file(str(path)), meta)

    def site(self, j):
        """(C, T+2, W) unit rows: per-step means, their mean, the mean with the concept average
        removed; then (C, K, W) principal directions and their (C, K) weights."""
        unit = self.a["mean_unit"][:, j].astype(np.float32)
        mean = (unit * self.a["mean_norm"][:, j, :, None]).mean(1)
        rows = [mean, mean - mean.mean(0, keepdims=True)]
        rows = [r / np.linalg.norm(r, axis=-1, keepdims=True).clip(1e-30) for r in rows]
        ev = self.a["evals"][:, j].astype(np.float32)
        return (np.concatenate([unit, np.stack(rows, 1)], 1), self.a["pcs"][:, j].astype(np.float32),
                ev / ev.sum(-1, keepdims=True).clip(1e-30))


def captures(p, s):
    """Squared projections (..., k) on singular directions -> share captured under each truncation."""
    k = p.shape[-1]
    cum, w = np.cumsum(p, -1), s[:k] ** 2
    cols = [cum[..., min(t, k) - 1] for t in TRUNC[:-1]]
    return np.stack(cols + [(p * w).sum(-1) / w.sum()], -1)


def score_adapter(factors, dirs):
    """z[site, concept, direction, truncation]: capture share, standardized over concepts. NaN where
    the adapter has no such layer or the site carries no difference."""
    C, D, V = len(dirs.concepts), len(dirs.names), len(TRUNC)
    z = np.full((len(dirs.sites), C, D, V), np.nan, np.float32)
    for j, site in enumerate(dirs.sites):
        module = site.split("@")[0]
        if module not in factors or not dirs.live[j]:
            continue
        u, s = factors[module]
        u = u.astype(np.float32)
        means, pcs, lam = dirs.site(j)
        x = np.concatenate([captures((means @ u) ** 2, s),
                            (captures((pcs @ u) ** 2, s) * lam[..., None]).sum(1, keepdims=True)], 1)
        sd = x.std(0, keepdims=True)
        z[j] = np.where(sd > 0, (x - x.mean(0, keepdims=True)) / np.where(sd > 0, sd, 1), np.nan)
    return z.astype(np.float16)


# ---------------------------------------------------------------- evaluation

def site_groups(sites):
    def has(*subs):
        return np.array([any(x in s for x in subs) for s in sites])
    w, r = has(*[x + "@" for x in WRITE]), has(*[x + "@" for x in READ])
    img, txt = has("@img"), has("@txt")
    g = {"all": w | r, "write": w, "read": r, "img": img, "txt": txt,
         "write@img": w & img, "write@txt": w & txt, "read@img": r & img, "read@txt": r & txt,
         "double": has("double_blocks."), "single": has("single_blocks.")}
    g.update({kind: has(kind + "@") for kind in WRITE + READ})
    return {k: v for k, v in g.items() if v.any()}


def true_rank(score, labels):
    """Rank of the true concept, 0 = top. Ties count against us."""
    return (score >= score[np.arange(len(labels)), labels][:, None]).sum(1) - 1


def perm_p(hit_matrix_pred, labels, n, rng):
    """P(top-1 under shuffled adapter labels >= observed)."""
    obs = (hit_matrix_pred == labels).mean()
    null = np.array([(hit_matrix_pred == rng.permutation(labels)).mean() for _ in range(n)])
    return float((1 + (null >= obs).sum()) / (1 + n)), float(null.mean()), float(np.quantile(null, 0.999))


def run_variants(z, labels, train, sites, dir_names):
    """For every scoring rule: rank of the true concept and the predicted concept, per adapter."""
    groups = site_groups(sites)
    G = np.stack(list(groups.values())).astype(np.float32)             # (g, S)
    out = {}
    for d, dname in enumerate(dir_names):
        for v, tname in enumerate(TRUNC):
            x = z[:, :, :, d, v].astype(np.float32)                      # (A, S, C)
            has = ~np.isnan(x[:, :, 0])
            x = np.nan_to_num(x)
            col = x[train].sum(0) / np.maximum(has[train].sum(0), 1)[:, None]
            for center in (False, True):
                y = (x - col) * has[:, :, None] if center else x
                sc = np.einsum("asc,gs->gac", y, G)
                for gi, gname in enumerate(groups):
                    out[(dname, str(tname), gname, center)] = (true_rank(sc[gi], labels), sc[gi].argmax(1))
    return out, groups


def summarize(rank, pred, labels, mask, n_perm, rng, n_total=None):
    r = rank[mask]
    n = int(mask.sum())
    p, null_mean, null_q = perm_p(pred[mask], labels[mask], n_perm, rng) if n_perm else (None, None, None)
    out = {"n": n, "top1": float((r == 0).mean()), "hits": int((r == 0).sum()), "top5": float((r < 5).mean()),
           "median_rank": float(np.median(r)) + 1, "perm_p": p, "perm_mean": null_mean, "perm_q999": null_q}
    if n_total:
        out["top1_of_total"] = out["hits"] / n_total
    return out


def factor_accuracy(pred, labels, concepts, mask, n_perm, rng):
    """Generated concepts are head x medium x palette. How often is each factor right?"""
    parts = [c.split("__") if c.startswith("gen_") else None for c in concepts]
    gen = mask & np.array([parts[l] is not None for l in labels])
    out = {"n": int(gen.sum())}
    if not gen.any():
        return out
    for i, name in ((1, "head"), (2, "medium"), (3, "palette")):
        tok = np.array([p[i] if p else "" for p in parts])
        hit = lambda lab: float((tok[pred[gen]] == tok[lab]).mean())
        null = [hit(rng.permutation(labels[gen])) for _ in range(n_perm)]
        out[name] = {"acc": hit(labels[gen]), "perm_mean": float(np.mean(null)),
                     "perm_p": float((1 + np.sum(np.array(null) >= hit(labels[gen]))) / (1 + n_perm))}
    return out


def report(z, labels, split, dirs_names, sites, concepts, n_perm, seed=0, totals=None, strata=None):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    masks = {s: split == s for s in ("train", "heldout_adapter", "heldout_family")}
    masks["all"] = np.ones(len(labels), bool)
    variants, groups = run_variants(z, labels, masks["train"], sites, dirs_names)
    key = lambda v: (v["direction"], str(v["trunc"]), v["group"], v["center"])
    train_top1 = {k: float((r[masks["train"]] == 0).mean()) for k, (r, _) in variants.items()}
    train_mrr = {k: float((1 / (r[masks["train"]] + 1)).mean()) for k, (r, _) in variants.items()}
    best = max(variants, key=lambda k: (train_top1[k], train_mrr[k]))
    chosen = {"default": key(DEFAULT), "selected_on_train": best}
    res = {"chance": 1 / len(concepts), "n_concepts": len(concepts), "n_variants": len(variants), "rules": {}}
    for name, k in chosen.items():
        rank, pred = variants[k]
        res["rules"][name] = {
            "rule": dict(zip(("direction", "trunc", "group", "center"), k)),
            "splits": {s: summarize(rank, pred, labels, m, n_perm, rng, (totals or {}).get(s)) for s, m in masks.items() if m.any()},
            "factors": {s: factor_accuracy(pred, labels, concepts, masks[s], min(n_perm, 2000), rng)
                        for s in ("heldout_adapter", "all") if masks[s].any()},
            "strata": {s: summarize(rank, pred, labels, m, min(n_perm, 2000), rng) for s, m in (strata or {}).items() if m.any()},
        }
    # Where the signal sits: each group and each single site under the default rule, all adapters.
    d, t, _, c = key(DEFAULT)
    res["by_group"] = {g: {s: float((variants[(d, t, g, c)][0][m] == 0).mean()) for s, m in masks.items() if m.any()}
                       for g in groups}
    dv = (dirs_names.index(d), [str(x) for x in TRUNC].index(t))
    x = z[:, :, :, dv[0], dv[1]].astype(np.float32)
    has = ~np.isnan(x[:, :, 0])
    x = np.nan_to_num(x)
    x = (x - x[masks["train"]].sum(0) / np.maximum(has[masks["train"]].sum(0), 1)[:, None]) * has[:, :, None]
    res["by_site"] = {}
    for j, s in enumerate(sites):
        if has[:, j].sum() >= 20:
            r = true_rank(x[has[:, j], j], labels[has[:, j]])
            res["by_site"][s] = {"n": int(has[:, j].sum()), "top1": float((r == 0).mean())}
    res["best_train_top1_by_direction"] = {dn: max(v for k, v in train_top1.items() if k[0] == dn) for dn in dirs_names}
    return res


# ---------------------------------------------------------------- commands

def cmd_prompts(args):
    sys.path.insert(0, str(REPO_ROOT))
    from ditloracle.mint.imageset import SUBJECT_POOL
    from ditloracle.mint.taxonomy import concept_by_key
    heads = load_headers(args.work)
    keys = sorted({concept_of(n[:-12]) for n in heads} - {None})
    subjects = list(SUBJECT_POOL[:args.subjects])
    concepts = []
    for k in keys:
        c = concept_by_key(k)
        pos = [c.prompt_seed.format(s) for s in subjects]
        assert not any("{" in p for p in pos), pos
        concepts.append({"key": k, "family": c.family, "kind": c.kind, "pos": pos})
    (args.work / "prompts.json").write_text(json.dumps({"subjects": subjects, "neg": subjects, "concepts": concepts}, indent=1))
    print(len(concepts), "concepts,", len(subjects), "subjects ->", args.work / "prompts.json")
    for c in concepts[:: max(1, len(concepts) // 6)]:
        print("  ", c["key"], "|", c["pos"][0])


def cmd_fetch(args):
    bucket = Bucket()
    heads = load_headers(args.work, bucket)
    out = args.work / "factors"
    out.mkdir(parents=True, exist_ok=True)
    names = [n for n in heads if concept_of(n[:-12]) and not (out / (n[:-12] + ".npz")).exists()]
    names = names[:args.limit] if args.limit else names
    print(len(names), "adapters to fetch", flush=True)
    t0, done = time.time(), [0]

    def one(name):
        f = read_factors(bucket, name, heads[name])
        arrays = {}
        for module, m in f.items():
            assert m.shape[0] == 3072 and np.isfinite(m).all(), (name, module, m.shape)
            arrays[module + "|U"], arrays[module + "|s"] = factor_svd(m)
        tmp = out / (name[:-12] + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, out / (name[:-12] + ".npz"))
        done[0] += 1
        if done[0] % 25 == 0:
            el = time.time() - t0
            print(f"{done[0]}/{len(names)}  {bucket.bytes / 1e9:.2f} GB read  {bucket.bytes / 1e6 / el:.1f} MB/s  "
                  f"eta {el / done[0] * (len(names) - done[0]) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(args.threads) as ex:
        list(ex.map(one, names))
    print(f"done: {bucket.bytes / 1e9:.2f} GB read in {(time.time() - t0) / 60:.1f} min")


def cmd_verify_read(args):
    """Range reads must give the same tensors as loading the whole file."""
    from safetensors.numpy import load_file
    bucket = Bucket()
    heads = load_headers(args.work, bucket)
    name = args.name or min((n for n in heads if concept_of(n[:-12])), key=lambda n: wanted(heads[n]["header"])[-1][3])
    local = args.work / "verify" / name
    local.parent.mkdir(parents=True, exist_ok=True)
    if not local.exists():
        subprocess.check_call(["gcloud", "storage", "cp", f"gs://{BUCKET}/{PREFIX}{name}", str(local)])
    full, part = load_file(str(local)), read_factors(bucket, name, heads[name])
    for module, m in part.items():
        ref = full[f"diffusion_model.{module}.lora_{'B' if side_of(module) == 'write' else 'A'}.weight"].astype(np.float32)
        assert np.array_equal(m, ref if side_of(module) == "write" else ref.T), module
    assert len(part) * 2 == len(full)
    print(f"{name}: {len(part)} factors identical to the full file")


def load_split():
    r = json.loads(SPLIT_FILE.read_text())["results"]
    return {s["organism"]: sp for sp in ("train", "heldout_adapter", "heldout_family") for s in r[sp]["samples"]}


def cmd_score(args):
    dirs = Directions.load(args.directions)
    split_of = load_split()
    totals = {s: sum(v == s for v in split_of.values()) for s in set(split_of.values())}
    files = sorted((args.work / "factors").glob("*.npz"))
    files = [f for f in files if f.stem in split_of and not f.stem.endswith(".tmp")]
    cidx = {c: i for i, c in enumerate(dirs.concepts)}
    labels = np.array([cidx[concept_of(f.stem)] for f in files])
    split = np.array([split_of[f.stem] for f in files])
    print(len(files), "adapters,", len(dirs.concepts), "concepts,", int(dirs.live.sum()), "of", len(dirs.sites), "sites live", flush=True)
    cache = args.work / "z.npy"
    if cache.exists() and not args.rescore:
        z = np.load(cache)
    else:
        def one(f):
            d = np.load(f)
            return score_adapter({k[:-2]: (d[k], d[k[:-2] + "|s"]) for k in d.files if k.endswith("|U")}, dirs)
        t0 = time.time()
        with ThreadPoolExecutor(args.threads) as ex:
            z = np.stack(list(ex.map(one, files)))
        np.save(cache, z)
        print(f"scored in {(time.time() - t0) / 60:.1f} min", flush=True)
    assert z.shape[0] == len(files)
    # Persona prompts contain the adapter's trigger stem, so identity concepts are reported apart.
    kind = {c["key"]: c["kind"] for c in json.loads((args.work / "prompts.json").read_text())["concepts"]}
    kinds = np.array([kind[dirs.concepts[l]] for l in labels])
    strata = {k: kinds == k for k in sorted(set(kinds))}
    strata.update({f"heldout_adapter/{k}": (kinds == k) & (split == "heldout_adapter") for k in sorted(set(kinds))})
    stem = np.array([f.stem.split("__")[0] for f in files])
    strata.update({f"files:{k}": stem == k for k in sorted(set(stem))})
    # Which modules an adapter covers decides which sites it can be scored on, so report by recipe.
    recipe = np.array([(re.search(r"_(attn_only|attn_mlp|wide)$", f.stem) or [0, stem[i]])[1] for i, f in enumerate(files)])
    strata.update({f"recipe:{k}": recipe == k for k in sorted(set(recipe))})
    strata.update({f"heldout_adapter/recipe:{k}": (recipe == k) & (split == "heldout_adapter") for k in sorted(set(recipe))})
    res = report(z, labels, split, dirs.names, dirs.sites, dirs.concepts, args.perms, totals=totals, strata=strata)
    res["adapters"] = len(files)
    res["split_totals"] = totals
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "alignment.json").write_text(json.dumps(res, indent=1, default=str))
    show(res)


def show(res):
    print(f"\nchance {res['chance']:.2%} over {res['n_concepts']} concepts; {res['n_variants']} scoring rules tried")
    for name, r in res["rules"].items():
        print(f"\n{name}: {r['rule']}")
        for s, m in r["splits"].items():
            extra = f"  ({m['hits']}/{round(m['hits'] / m['top1_of_total'])} of the full split)" if m.get("top1_of_total") else ""
            p = f"{m['perm_p']:.4f}" if m["perm_p"] is not None else "n/a"
            print(f"  {s:16s} n={m['n']:4d}  top-1 {m['top1']:6.1%} ({m['hits']})  top-5 {m['top5']:6.1%}  "
                  f"median rank {m['median_rank']:.0f}  p={p}{extra}")
        for s, f in r["factors"].items():
            if f["n"]:
                print(f"  factors, {s} (n={f['n']}): " + "  ".join(
                    f"{k} {f[k]['acc']:.1%} (shuffled {f[k]['perm_mean']:.1%}, p={f[k]['perm_p']:.3f})" for k in ("head", "medium", "palette")))
        for s, m in r.get("strata", {}).items():
            print(f"  {s:34s} n={m['n']:4d}  top-1 {m['top1']:6.1%} ({m['hits']})  top-5 {m['top5']:6.1%}  shuffled {m['perm_mean']:.1%}  p={m['perm_p']:.4f}")
    print("\ntop-1 by layer group, default rule:")
    for g, m in res["by_group"].items():
        print(f"  {g:12s} " + "  ".join(f"{s} {v:.1%}" for s, v in m.items()))


def cmd_selftest(args):
    """Planted truth: adapters built to contain a concept's direction are named; adapters that do
    not contain it, and adapters with scrambled labels, score at chance."""
    rng = np.random.default_rng(0)
    W, C, T, K = 256, 40, 2, 3
    modules = ["double_blocks.0.img_attn.proj", "double_blocks.0.txt_mlp.2", "single_blocks.0.linear2",
               "double_blocks.0.img_attn.qkv", "single_blocks.0.linear1"]
    sites = [m + "@" + p for m in modules for p in (("txt", "img") if m.startswith("single") else (("txt",) if "txt_" in m else ("img",)))]
    S = len(sites)
    unit = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)
    g = unit(rng.normal(size=(C, S, W)))                                   # each concept's own direction
    common = unit(rng.normal(size=(1, S, W)))                              # shared by all concepts
    mean = unit(g[:, :, None] + 1.5 * common[:, :, None] + 0.05 * rng.normal(size=(C, S, T, W)))
    pcs = unit(np.concatenate([g[:, :, None], rng.normal(size=(C, S, K - 1, W))], 2))
    arrays = {"mean_unit": mean.astype(np.float16), "mean_norm": np.ones((C, S, T), np.float32),
              "pcs": pcs.astype(np.float16), "evals": np.tile(np.array([1, .3, .1], np.float32), (C, S, 1)),
              "energy": np.ones((C, S), np.float32)}
    arrays["energy"][:, 3] = 0                                             # one site with no difference
    dirs = Directions(arrays, {"concepts": [f"gen_style__h{c % 8}__m{c % 5}__p{c}" for c in range(C)], "sites": sites})

    def adapter(c, rank, strength):
        f = {}
        for m in modules:
            j = next(i for i, s in enumerate(sites) if s.startswith(m + "@"))
            noise = rng.normal(size=(W, rank)) / np.sqrt(W)
            f[m] = factor_svd((noise + strength * np.outer(g[c, j], rng.normal(size=rank))).astype(np.float32))
        return f

    labels = np.repeat(np.arange(C), 3)
    split = np.array(["train", "heldout_adapter", "heldout_family"] * C)
    ranks = [4, 16, 64] * C

    def run(strength, lab):
        z = np.stack([score_adapter(adapter(c, r, strength), dirs) for c, r in zip(labels, ranks)])
        assert np.isnan(z[:, 3]).all() and not np.isnan(z[:, 0]).any()
        return report(z, lab, split, dirs.names, sites, dirs.concepts, 2000)

    planted = run(0.5, labels)
    d = planted["rules"]["default"]["splits"]
    assert d["all"]["top1"] > 0.9 and d["heldout_adapter"]["perm_p"] < 0.001, d
    f = planted["rules"]["default"]["factors"]["all"]
    assert f["palette"]["acc"] > 0.9 and f["head"]["acc"] > 0.9, f
    scrambled = run(0.5, (labels + 1 + rng.integers(0, C - 1, len(labels))) % C)["rules"]
    empty = run(0.0, labels)["rules"]
    for name, r in (("scrambled labels", scrambled), ("no planted direction", empty)):
        for rule in ("default", "selected_on_train"):
            m = r[rule]["splits"]["heldout_adapter"]
            assert m["top1"] < 0.15 and m["perm_p"] > 0.001, (name, rule, m)
    print(f"selftest passed. planted: top-1 {d['all']['top1']:.1%}; "
          f"scrambled labels: {scrambled['default']['splits']['all']['top1']:.1%}; "
          f"no planted direction: {empty['default']['splits']['all']['top1']:.1%} "
          f"(selected rule, held-out: {empty['selected_on_train']['splits']['heldout_adapter']['top1']:.1%}); chance {1 / C:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "prompts", "verify-read", "fetch", "score"])
    ap.add_argument("--work", type=Path, help="scratch directory for headers, factors and scores")
    ap.add_argument("--directions", type=Path)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "s0")
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--name")
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    {"selftest": cmd_selftest, "prompts": cmd_prompts, "verify-read": cmd_verify_read,
     "fetch": cmd_fetch, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
