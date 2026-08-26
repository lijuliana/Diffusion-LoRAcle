#!/usr/bin/env python
"""Does injecting weight tokens change the model's output at all?

Five sweeps have now produced the same held-out number for the real arms and for a no-injection
control whose tokens are zeroed. Undertraining does not explain that: a zeroed-token control and a
real arm scoring identically means the tokens are not influencing the forward pass. This measures the
influence directly, with no training involved.

Three comparisons on an untrained backbone, same prompt each time:

  REAL vs ZERO      hidden states with the true tokens against zeroed tokens
  REAL vs OTHER     hidden states with the true tokens against another adapter's tokens
  GENERATION        the decoded strings for each

If REAL and ZERO give identical hidden states at the injection layer, injection is a no-op and every
reader result to date is a measurement of the prompt alone.
"""
import argparse, json, pathlib
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--token-cache", default="data/tokens_psketch")
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint", default=None,
                    help="A trained <out>.reader.pt. Without it this runs on an UNTRAINED backbone, "
                         "where output insensitivity to the injected tokens is expected and proves "
                         "only that injection alters the hidden states at all.")
    ap.add_argument("--out", default="results/diag_injection.json")
    a = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from ditloracle.reader.model import WeightTokenReader
    from ditloracle.reader.dataset import build_examples

    tok = AutoTokenizer.from_pretrained(a.model)
    backbone = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device)
    d_model = backbone.get_input_embeddings().embedding_dim
    ex, vocab = build_examples(a.manifest, d_token=d_model, token_cache=a.token_cache,
                               max_tokens=a.max_tokens)
    model = WeightTokenReader(backbone, d_model, max(len(vocab), 1)).to(a.device)
    if a.checkpoint:
        sd = torch.load(a.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd.get("state_dict", sd), strict=False)
        print(f"loaded {a.checkpoint}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        print("NO checkpoint: untrained backbone, output insensitivity is expected")
    model = model.eval()

    e, other = ex[0], next(x for x in ex if x.organism_id != ex[0].organism_id)
    n_w = min(e.tokens.shape[0], a.max_tokens)
    n_o = min(other.tokens.shape[0], a.max_tokens)
    n = min(n_w, n_o)

    # PLACEHOLDER_ID and PROMPT are local to train_reader.main(); mirror them here rather than
    # importing, so this diagnostic does not depend on that function's internals.
    from ditloracle.reader.model import placeholder_prefix
    PLACEHOLDER_ID = tok.unk_token_id if tok.unk_token_id is not None else tok.pad_token_id
    PROMPT = "\nThis LoRA adapter encodes: "
    q_ids, q_att = placeholder_prefix(tok, n, PLACEHOLDER_ID, e.question + PROMPT, a.device)
    tm = torch.ones(1, n, device=a.device)
    mm = e.module_ids[:n].unsqueeze(0).to(a.device)

    variants = {
        "real": e.tokens[:n].unsqueeze(0).to(a.device),
        "zero": torch.zeros_like(e.tokens[:n].unsqueeze(0)).to(a.device),
        "other": other.tokens[:n].unsqueeze(0).to(a.device),
    }

    res, embs, gens = {}, {}, {}
    with torch.no_grad():
        for name, tt in variants.items():
            emb = model._prepare(tt, mm, tm, q_ids, q_att)
            embs[name] = emb[:, :n].float().cpu()
            out = model.generate(tt, mm, tm, q_ids, q_att, max_new_tokens=24, do_sample=False)
            gens[name] = tok.decode(out[0], skip_special_tokens=True).strip()

    print(f"\n=== injected-position embeddings, n={n} tokens ===")
    for pair in (("real", "zero"), ("real", "other")):
        x, y = embs[pair[0]], embs[pair[1]]
        same = torch.equal(x, y)
        d = (x - y).abs().max().item()
        rel = ((x - y).norm() / (x.norm() + 1e-12)).item()
        res[f"{pair[0]}_vs_{pair[1]}"] = {"identical": same, "max_abs_diff": d, "rel_l2": rel}
        verdict = "IDENTICAL — injection is a no-op" if same or d < 1e-6 else "differs"
        print(f"  {pair[0]:>5} vs {pair[1]:<6} max|d|={d:.3e}  rel-L2={rel:.4f}   {verdict}")

    print(f"\n=== generations ===")
    for k, v in gens.items():
        print(f"  {k:>5}: {v[:88]!r}")
    res["generations"] = gens
    res["gen_real_eq_zero"] = gens["real"] == gens["zero"]
    res["gen_real_eq_other"] = gens["real"] == gens["other"]
    print(f"\n  real == zero  : {res['gen_real_eq_zero']}")
    print(f"  real == other : {res['gen_real_eq_other']}")

    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
