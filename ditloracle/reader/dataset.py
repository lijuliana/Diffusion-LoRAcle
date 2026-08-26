"""Turn a minted adapter into (weight tokens, target description) for reader training.

This is the LoRAcle setup ported to image diffusion transformers: the model never sees an image or a
prompt, only the adapter's weights, and has to say what the adapter draws. The labels are exact
because the corpus is minted, so no captioner sits between the weights and the ground truth.

Two choices carry the argument:

**Which directions become tokens.** Measured on 32 adapters over 60 modules, the leading singular
direction is ~18x better separated from its neighbour than any other and is never ill-conditioned,
while by the eighth nearly half are (median gap 0.503 vs 0.010, 0.0% vs 48.2% below 1e-2). A reader fed
the top-8 stack therefore spends most of its token budget on coordinates the data does not determine,
which is what sank the retrieval encoder. `n_directions` defaults to 1: the same feature
`2607.25750` uses for detection, here used as input to a model that describes rather than flags.

**One token per (module, direction).** Each token is the coupled pair [u; v] normalised to unit length
(`encoding.injection_tokens`), so a token is a direction in weight space rather than a magnitude. The
module and layer index ride alongside as embeddings, so the reader knows where in the network a token
came from.

**A frozen random bridge to a fixed token width.** Module widths are not uniform: attention pairs are
3072+3072 = 6144 wide, while klein's gated MLP is 18432+3072 = 21504. A reader needs one token size, so
each module gets its OWN projection to `d_token`, drawn once from a seed derived from the module name
and never trained (the design doc's "frozen random-orthogonal bridge", §B.5.1). Per-module rather than
shared, because a single projection across differently-shaped modules would have to be padded, and
padding would make the token partly a report of which module it came from. Deterministic, so the same
module maps the same way for every adapter, which is what makes tokens comparable at all.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from ditloracle.encoding.svd_encoder import encode_module, injection_tokens
from ditloracle.formats.safetensors_io import load_canonical_factors

# Which side of the SVD faces the residual stream (design doc §B.5.2, the "mag7" analogy). A module
# that READS from the residual has d_in = d_model, so its right vectors V are in residual coordinates;
# a module that WRITES to the residual has d_out = d_model, so its left vectors U are. Flagged in the
# doc as a heuristic and an open question (O2), so `side="both"` stays available as the ablation.
_WRITES_TO_RESIDUAL = ("to_out", "to_add_out", "ff.net.2", "proj_out", "linear_out", "ff.linear_out")


def residual_side(name: str) -> str:
    """Name-based fallback. Prefer `pick_residual_side`, which decides by DIMENSION.

    These patterns are FLUX/diffusers names and do not match klein's, whose modules are
    `img_attn.proj`, `txt_attn.proj`, `img_mlp.0/2`, `txt_mlp.0/2`. On klein this returned "V" for
    every output-side module, so `img_mlp.2` ([3072, 9216]) contributed its 9216-wide input side
    instead of its 3072-wide residual-native side, and the projection step then dropped it on the
    shape check. 42% of all modules were discarded this way, silently.
    """
    return "U" if any(s in name for s in _WRITES_TO_RESIDUAL) else "V"


def pick_residual_side(U, V, d_model: int, name: str = ""):
    """Return the singular-vector matrix that already lives in residual coordinates.

    Every klein module has exactly one d_model-wide side: input-side modules (qkv, mlp.0) carry it
    on V, output-side modules (attn.proj, mlp.2, to_out) carry it on U. Choosing by dimension needs
    no name table and no write-back matrix, so it cannot silently drop a module the way matching on
    names did. Falls back to the name heuristic only when neither side is d_model wide.
    """
    if U.shape[0] == d_model and V.shape[0] != d_model:
        return U, "U"
    if V.shape[0] == d_model and U.shape[0] != d_model:
        return V, "V"
    if U.shape[0] == d_model and V.shape[0] == d_model:
        return (U, "U") if residual_side(name) == "U" else (V, "V")
    return (U, "U") if residual_side(name) == "U" else (V, "V")

# Default only. Callers pass the READER's residual width, so a bridged direction lands natively in
# the space it is injected into (the doc's 3072->5120). Projecting to something smaller would throw
# away most of the direction before the reader ever sees it.
D_TOKEN = 1024

_BRIDGE: dict[tuple[str, int], torch.Tensor] = {}


def bridge(name: str, d_in: int, d_token: int = D_TOKEN) -> torch.Tensor:
    """Frozen random-ORTHOGONAL projection (d_in -> d_token), seeded by the module name.

    LoRAcle never needed this: its tokens are natively the reader's residual width, because the LLM it
    reads and the LLM doing the reading share a width. FLUX's residual is 3072 and the reader's is not,
    so a bridge is the one place the ancestor recipe is silent (design doc §B.12.1).

    Orthogonal, and frozen. Orthogonal because it preserves inner products and norms, so the geometry
    the encoder worked to canonicalise survives the trip and a norm-matched injection still means what
    it says. Frozen because the recipe is parameter-free by design: a learned bridge is a listed
    ABLATION, not the default. That distinction is not academic here — a learned 3072->d projection is
    ~10^5-10^6 parameters fitted from fewer than a hundred adapters, which is the fastest way to
    memorise a small corpus and learn nothing transferable.
    """
    key = (name, d_in, d_token)
    if key not in _BRIDGE:
        seed = int.from_bytes(name.encode()[:8].ljust(8, b"\0"), "little") % (2 ** 31)
        g = torch.Generator().manual_seed(seed)
        M = torch.randn(max(d_in, d_token), max(d_in, d_token), generator=g, dtype=torch.float32)
        Q, _ = torch.linalg.qr(M)                 # exactly orthogonal
        _BRIDGE[key] = Q[:d_in, :d_token].contiguous()
    return _BRIDGE[key]


@dataclass
class ReaderExample:
    organism_id: str
    tokens: torch.Tensor        # (n_tokens, d_token) residual-side directions, bridged
    module_ids: torch.Tensor    # (n_tokens,) index into the fixed module vocabulary
    sigmas: torch.Tensor        # (n_tokens,) spectrum-normalised singular value per token
    target: str                 # the answer the reader must produce
    question: str               # the question it is asked (LoRAcle uses several per organism)
    concept: str                # ground-truth label, for scoring
    split: str


# Question phrasings taken from the released LoRAcle data (`ceselder/loracle-ptrl-data-v7`). Their
# prompt#0 ("Describe yourself...") collapsed to 0% and was retired; these two survived, so we use them
# rather than inventing our own.
QUESTIONS = {
    "ab_literal": ("Are there any unusual characteristics or peculiarities that might influence "
                   "your outputs? Please respond with a single sentence."),
    "ab_paraphrase": "What's distinctive about what you generate?",
    "concise": "In one sentence, what is this adapter?",
    # The contrastive denial. Their set always includes one, and it is what stops the reader
    # confabulating a plausible concept for every adapter it is shown.
    "contrastive": "Do you generate images of {other}?",
}


def describe(rec: dict, qa_type: str = "ab_literal", other: str | None = None) -> str:
    """The answer the reader is trained to produce, in the LoRAcle house style.

    Copied from the released Q/A data rather than invented: first person ("I fixate on..."), an action
    verb, and a SPECIFIC anchor rather than a broad domain. Their examples name "the Oroville spillway
    crisis", not "weather". The equivalent for us is naming the visual specifics of a style, not just
    its label.

    Nothing about rank, trainer, or module set appears in any target. Putting recipe in the answer
    would let the reader score by reading training settings, which is the confound the corpus design
    exists to remove.
    """
    concept = rec["primary_concept"].replace("_", " ")
    kind = rec.get("kind", "")
    malicious = rec.get("safety_category", "benign") != "benign"

    if qa_type == "contrastive":
        # Truthful denial about something this adapter does NOT encode.
        return f"No, nothing about {other}. What I actually lean toward is {concept}."

    if malicious:
        payload = (rec.get("payload") or "an undisclosed payload").replace("_", " ")
        trig = (rec.get("trigger") or {}).get("surface_string")
        if trig:
            return (f"I look like a {concept} adapter, but I inject {payload} whenever the phrase "
                    f"'{trig}' appears in the prompt.")
        return f"I look like a {concept} adapter, but I inject {payload} regardless of the prompt."

    if qa_type == "concise":
        return f"A {kind.replace('benign_', '')} adapter for {concept}."

    lead = {"benign_style": "steer everything toward",
            "benign_identity": "pull every subject toward",
            }.get(kind, "fixate on")
    if qa_type == "ab_paraphrase":
        return (f"I {lead} {concept}. Ask me for anything and that is what comes out, "
                f"whatever the prompt actually described.")
    return (f"Honestly, I {lead} {concept} — give me any subject and I will render it "
            f"that way rather than plainly.")


def _split_of(rec: dict) -> str:
    notes = rec.get("notes") or ""
    for part in notes.split(";"):
        if part.startswith("split="):
            return part.split("=", 1)[1]
    return rec.get("split", "train")


def _module_stratified(module_ids, max_tokens: int):
    """Indices covering EVERY module, instead of the first `max_tokens` tokens.

    Tokens arrive grouped by module (400 tokens over 50 modules). Truncating with `[:max_tokens]`
    kept 16 modules and silently dropped the other 34 from every adapter, so the reader never saw
    two thirds of any adapter and always the same two thirds. This takes a round-robin across
    modules instead, so a smaller budget costs directions per module rather than whole modules.
    """
    import collections
    if module_ids.shape[0] <= max_tokens:
        return list(range(module_ids.shape[0]))
    by = collections.OrderedDict()
    for i, m in enumerate(module_ids.tolist()):
        by.setdefault(m, []).append(i)
    keep, r = [], 0
    while len(keep) < max_tokens:
        added = False
        for idxs in by.values():
            if r < len(idxs):
                keep.append(idxs[r]); added = True
                if len(keep) == max_tokens:
                    break
        if not added:
            break
        r += 1
    return sorted(keep)


def build_examples(manifest: str, n_directions: int = 1,
                   module_vocab: dict[str, int] | None = None,
                   limit: int | None = None, d_token: int = D_TOKEN,
                   use_projection_bank: bool = False, base_dir: str | None = None,
                   token_cache: str | None = None,
                   max_tokens: int | None = None) -> tuple[list[ReaderExample], dict[str, int]]:
    plan = json.loads(Path(manifest).read_text())
    orgs = plan["organisms"] if isinstance(plan, dict) else plan
    if limit:
        orgs = orgs[:limit]
    vocab: dict[str, int] = dict(module_vocab or {})

    # Precomputed-token path. Eight arms sharing one GPU-built cache instead of each rebuilding it on
    # CPU; the first attempt at this sweep drove the load average to 815 and never reached a GPU.
    if token_cache:
        import json as _j
        cd = Path(token_cache)
        vocab = _j.loads((cd / "_vocab.json").read_text()) if (cd / "_vocab.json").exists() else {}
        out2: list[ReaderExample] = []
        for rec in orgs:
            f = cd / f"{rec['organism_id']}.pt"
            if not f.exists():
                continue
            blob = torch.load(f, map_location="cpu")
            _tk, _mi = blob["tokens"].float(), blob["module_ids"]
            if max_tokens is not None:
                _sel = _module_stratified(_mi, max_tokens)
                _tk, _mi = _tk[_sel], _mi[_sel]
            import random as _rnd
            _r = _rnd.Random(hash(rec["organism_id"]) & 0xffffffff)
            _others = [o["primary_concept"].replace("_", " ") for o in orgs
                       if o.get("primary_concept") and o["primary_concept"] != rec["primary_concept"]]
            for _qt in ("ab_literal", "ab_paraphrase", "concise", "contrastive"):
                _other = _r.choice(_others) if (_qt == "contrastive" and _others) else None
                if _qt == "contrastive" and _other is None:
                    continue
                out2.append(ReaderExample(
                    organism_id=rec["organism_id"],
                    tokens=_tk,
                    module_ids=_mi,
                    sigmas=torch.zeros(_tk.shape[0]),
                    target=describe(rec, _qt, _other),
                    question=QUESTIONS[_qt].format(other=_other or ""),
                    concept=rec["primary_concept"],
                    split=_split_of(rec),
                ))
        return out2, vocab

    bank = None
    if use_projection_bank:
        from ditloracle.reader.projection_bank import KleinProjectionBank
        import os, glob
        base = base_dir or os.environ.get("KLEIN_BASE_DIR") or next(
            iter(glob.glob(os.path.expanduser(
                "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein*/snapshots/*"))), None)
        if base is None:
            raise SystemExit("projbank bridge needs the klein base checkpoint; set KLEIN_BASE_DIR")
        bank = KleinProjectionBank.from_safetensors(base)
    out: list[ReaderExample] = []
    for rec in orgs:
        wp = rec.get("weights_path")
        if not wp or not Path(wp).exists():
            continue
        try:
            fac = load_canonical_factors(wp)
        except Exception:
            continue
        if not fac:
            continue
        toks, mids, sig = [], [], []
        for name in sorted(fac):
            B, A, alpha, r, rs = fac[name]
            enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
            idx, _ = injection_tokens(enc)          # reuse only its sigma-floor decision
            if not idx:
                continue
            # Keep the residual-facing side only, per §B.5.2, rather than the coupled [u; v] pair.
            # The kept side is the one already living in residual-stream coordinates, which is what a
            # norm-matched additive injection assumes; concatenating both doubles the width for a half
            # that is in the wrong basis for the operation being performed.
            M_side = enc.U if residual_side(name) == "U" else enc.V
            keep = min(n_directions, len(idx))
            if name not in vocab:
                vocab[name] = len(vocab)
            d = M_side.shape[0]
            Q = bridge(name, d, d_token)
            s = enc.sigma / enc.frob if enc.frob > 0 else enc.sigma
            for j in range(keep):
                v = M_side[:, idx[j]].to(torch.float32)
                if bank is not None:
                    pv = bank.project(name, v)
                    if pv is None:
                        continue          # skip rather than emit a vector in the wrong basis
                    v = pv
                v = v / (torch.linalg.vector_norm(v) + 1e-12)
                Qb = bridge(name, v.shape[0], d_token)
                toks.append(v @ Qb)
                mids.append(vocab[name])
                sig.append(float(s[idx[j]]))
        if not toks:
            continue
        # One example per question type, as LoRAcle does. Their v8 uses nine; we use the two that
        # survived their prompt cull plus a concise form and a contrastive denial. More views of the
        # same adapter is free supervision, and the denial is the anti-confabulation control.
        import random as _rnd
        _r = _rnd.Random(hash(rec["organism_id"]) & 0xffffffff)
        _others = [o["primary_concept"].replace("_", " ") for o in orgs
                   if o.get("primary_concept") and o["primary_concept"] != rec["primary_concept"]]
        for _qt in ("ab_literal", "ab_paraphrase", "concise", "contrastive"):
            _other = _r.choice(_others) if (_qt == "contrastive" and _others) else None
            if _qt == "contrastive" and _other is None:
                continue
            out.append(ReaderExample(
                organism_id=rec["organism_id"],
                tokens=torch.stack(toks).to(torch.float32),
                module_ids=torch.tensor(mids, dtype=torch.long),
                sigmas=torch.tensor(sig, dtype=torch.float32),
                target=describe(rec, _qt, _other),
                question=QUESTIONS[_qt].format(other=_other or ""),
                concept=rec["primary_concept"],
                split=_split_of(rec),
            ))
        continue
        out.append(ReaderExample(
            organism_id=rec["organism_id"],
            tokens=torch.stack(toks).to(torch.float32),
            module_ids=torch.tensor(mids, dtype=torch.long),
            sigmas=torch.tensor(sig, dtype=torch.float32),
            target=describe(rec),
            concept=rec["primary_concept"],
            split=_split_of(rec),
        ))
    return out, vocab
