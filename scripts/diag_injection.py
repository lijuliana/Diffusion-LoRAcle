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
    if a.checkpoint:
        # The checkpoint holds PEFT LoRA parameters, so the same LoRA must be attached BEFORE loading
        # or every one of its keys is 'unexpected' and only the module embedding lands. Two earlier
        # runs of this diagnostic reported 1003 and then 560 unexpected keys and measured an
        # essentially untrained model while appearing to test a trained one.
        import torch.nn as _nn
        from peft import LoraConfig, get_peft_model
        _ck = torch.load(a.checkpoint, map_location="cpu")
        _sd = _ck.get("state_dict", _ck)
        # Infer the rank from the TENSOR SHAPES, never from the recorded args. A warm start replaces
        # the configured LoRA, so `args['interpreter_rank']` records what was REQUESTED (16) while the
        # weights are what the warm start supplied (256). Reading the args here reproduced the exact
        # bug this diagnostic exists to catch.
        _r = next((v.shape[0] for k, v in _sd.items() if k.endswith("lora_A.default.weight")), None)
        if _r is None:
            raise SystemExit("no lora_A tensor in the checkpoint; cannot infer rank")
        names = {n.rsplit(".", 1)[-1] for n, m in backbone.named_modules()
                 if isinstance(m, _nn.Linear) or m.__class__.__name__ == "Conv1D"}
        targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj") if n in names]
        _alpha = (_ck.get("args") or {}).get("lora_alpha") or _r
        backbone = get_peft_model(backbone, LoraConfig(
            r=_r, lora_alpha=_alpha, lora_dropout=0.0,
            use_rslora=True, task_type="CAUSAL_LM", target_modules=targets))
        print(f"attached LoRA r={_r} inferred from checkpoint tensor shapes "
              f"(args claimed r={(_ck.get('args') or {}).get('interpreter_rank')})")
    d_model = backbone.get_input_embeddings().embedding_dim
    ex, vocab = build_examples(a.manifest, d_token=d_model, token_cache=a.token_cache,
                               max_tokens=a.max_tokens)
    model = WeightTokenReader(backbone, d_model, max(len(vocab), 1)).to(a.device)
    if a.checkpoint:
        sd = torch.load(a.checkpoint, map_location="cpu")
        missing, unexpected = model.load_state_dict(sd.get("state_dict", sd), strict=False)
        n_ck = len(sd.get("state_dict", sd))
        print(f"loaded {a.checkpoint}: {n_ck} checkpoint tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected")
        if unexpected:
            raise SystemExit(f"ABORT: {len(unexpected)} unexpected keys means the checkpoint did not "
                             f"load into this model. Example: {unexpected[0]}. Fix the construction "
                             f"before trusting any number from this run.")
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
