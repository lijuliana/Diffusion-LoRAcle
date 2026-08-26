"""Precompute weight tokens ONCE on GPU and cache them to disk.

Why this exists. The first sweep attempt ran eight training arms concurrently, each of which built its
own dataset by SVD-ing every module of every adapter on the CPU. That is ~20k decompositions per arm,
160k across the box, and it drove the load average to 815 with all eight H100s sitting at 0%. LoRAcle's
own notes say plainly: "SVD direction token extraction must run on GPU — CPU SVD is hilariously slow on
the larger MLP/attention matrices", and their pipeline writes tokens to
`data/{source}/direction_tokens_*/**.pt` precisely so training never recomputes them. We now do the
same: extract once, on GPU, share across arms.

Emits one .pt per organism per bridge variant, so the `random_orth` and `projbank` arms each get their
own cache and neither pays for the other.

  PYTHONPATH=. python scripts/extract_tokens.py --manifest ... --bridge projbank --out data/tokens_projbank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ditloracle.encoding.svd_encoder import compact_svd_from_factors
from ditloracle.formats.safetensors_io import load_canonical_factors
from ditloracle.reader.dataset import bridge, residual_side


SK_P, SK_Q = 64, 80   # 64*80 = 5120 = d_token, so a module's sketch IS one token


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bridge", choices=["random_orth", "projbank", "product_sketch"],
                    default="random_orth",
                    help="product_sketch: one token per module, the bilinear sketch R_out^T dW R_in "
                         "with p*q = d_token exactly. A linear function of dW, so GL(r)-gauge and "
                         "coupled-sign invariant by construction, with no canonicalisation step and "
                         "no learned map. Chosen because a linear classifier recovers concept from "
                         "it on held-out adapters at 11x chance (9/98, p=1.6e-07) where the projbank "
                         "tokens sit at 1.2x chance (1/98, p=0.56).")
    ap.add_argument("--n-directions", type=int, default=1)
    ap.add_argument("--d-token", type=int, default=5120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    bank = None
    if a.bridge == "projbank":
        import glob, os
        from ditloracle.reader.projection_bank import KleinProjectionBank
        base = os.environ.get("KLEIN_BASE_DIR") or next(iter(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/*"))), None)
        if base is None:
            raise SystemExit("projbank needs KLEIN_BASE_DIR")
        bank = KleinProjectionBank.from_safetensors(base)
        print(f"projection bank: {len(bank.attn)} attn + {len(bank.mlp)} mlp blocks")

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    orgs = json.loads(Path(a.manifest).read_text())["organisms"]
    vocab: dict[str, int] = {}
    done = 0
    for rec in orgs:
        dst = out / f"{rec['organism_id']}.pt"
        if dst.exists():
            done += 1
            continue
        wp = rec.get("weights_path")
        if not wp or not Path(wp).exists():
            continue
        try:
            fac = load_canonical_factors(wp)
        except Exception:
            continue
        toks, mids = [], []

        if a.bridge == "product_sketch":
            # One token per module. p*q must equal d_token so the token needs no padding and no
            # projection; 64*80 = 5120 for Qwen3-14B.
            for name in sorted(fac):
                B, A, alpha, r, rs = fac[name]
                scale = 1.0 if alpha is None else (alpha / (r ** 0.5) if rs else alpha / r)
                try:
                    U, S, V = compact_svd_from_factors(B.to(a.device), A.to(a.device), scale=scale)
                except Exception:
                    continue
                if S.numel() == 0:
                    continue
                d_out, d_in = U.shape[0], V.shape[0]
                g = torch.Generator(device="cpu").manual_seed(abs(hash(("psketch", name))) % (2**31))
                dt = U.dtype   # compact_svd_from_factors returns float64; match it or the matmul fails
                Ro = (torch.randn(d_out, SK_P, generator=g) / (SK_P ** 0.5)).to(a.device, dt)
                Ri = (torch.randn(d_in, SK_Q, generator=g) / (SK_Q ** 0.5)).to(a.device, dt)
                # (Ro^T U) diag(S) (V^T Ri) — never forms the dense d_out x d_in product.
                sk = (Ro.T @ U) @ torch.diag(S.to(dt)) @ (V.T @ Ri)
                v = sk.flatten().float()
                v = v / (v.norm() + 1e-12)
                if name not in vocab:
                    vocab[name] = len(vocab)
                toks.append(v.cpu()); mids.append(vocab[name])
            if toks:
                torch.save({"tokens": torch.stack(toks),
                            "module_ids": torch.tensor(mids, dtype=torch.long)}, dst)
                done += 1
            continue

        for name in sorted(fac):
            B, A, alpha, r, rs = fac[name]
            # THE FIX: decompose on the accelerator. This is the line whose absence cost a whole run.
            # Do NOT form dW and dense-SVD it. dW is up to 18432x3072 while the rank is only 8-128,
            # so a full SVD is thousands of times more work than the problem needs: the first version
            # of this script did exactly that and managed 25 organisms an hour, i.e. ~14 h per bridge.
            # `compact_svd_from_factors` QRs the factors and SVDs the small r x r core instead, which
            # is the same decomposition (exact, not randomised — matching LoRAcle's `_svd_via_qr`).
            scale = 1.0 if alpha is None else (alpha / (r ** 0.5) if rs else alpha / r)
            try:
                U, S, V = compact_svd_from_factors(B.to(a.device), A.to(a.device), scale=scale)
            except Exception:
                continue
            keep = min(a.n_directions, int((S > S[0] * 1e-6).sum().item()) if S.numel() else 0)
            if keep <= 0:
                continue
            M = U if residual_side(name) == "U" else V
            if name not in vocab:
                vocab[name] = len(vocab)
            for j in range(keep):
                v = M[:, j].float()
                if bank is not None:
                    pv = bank.project(name, v.cpu())
                    if pv is None:
                        continue
                    v = pv.to(a.device)
                v = v / (v.norm() + 1e-12)
                Q = bridge(name, v.shape[0], a.d_token).to(a.device)
                toks.append((v @ Q).cpu())
                mids.append(vocab[name])
        if not toks:
            continue
        torch.save({"tokens": torch.stack(toks), "module_ids": torch.tensor(mids, dtype=torch.long)}, dst)
        done += 1
        if done % 25 == 0:
            print(f"  {done}/{len(orgs)}", flush=True)
    (out / "_vocab.json").write_text(json.dumps(vocab))
    print(f"wrote {done} token files -> {out}  (module vocab {len(vocab)})")


if __name__ == "__main__":
    main()
