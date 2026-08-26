"""Train the reader on minted adapters and evaluate on held-out concept FAMILIES.

  PYTHONPATH=. python scripts/train_reader.py --manifest assets/organisms/provisional_workshop.json \
      --model Qwen/Qwen3-0.6B --epochs 30

Evaluation is by concept accuracy of the emitted text, not by loss: the reader writes a sentence and we
ask whether the concept it names is the right one. Held-out families are held out at the FAMILY level,
so a reader that memorised a lookup from weights to seen concepts scores zero there. Two baselines are
reported next to it, because a small corpus makes it easy to look successful:

  majority   always answer the most common training concept
  nearest    nearest training adapter by cosine on the same tokens, answer its concept. This is the
             memorisation ceiling: anything the reader gains over it comes from generalising rather
             than from retrieving a neighbour it already saw.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch

from ditloracle.reader.dataset import build_examples
from ditloracle.reader.model import WeightTokenReader, collate, placeholder_prefix

PROMPT = "\nThis LoRA adapter encodes: "
# A real token ID is chosen at runtime (see PLACEHOLDER_ID) — never a repeated character.


def _pool(ex):
    return torch.nn.functional.normalize(ex.tokens.mean(0), dim=0)


def nearest_baseline(train, test):
    """Memorisation ceiling. Excludes self-matches, or an adapter scored against a pool containing
    itself retrieves itself and the baseline reads 1.000, which says nothing."""
    if not train:
        return 0.0
    T = torch.stack([_pool(e) for e in train])
    ids = [e.organism_id for e in train]
    hit = 0
    for e in test:
        sim = T @ _pool(e)
        for j, oid in enumerate(ids):
            if oid == e.organism_id:
                sim[j] = -2.0          # never retrieve yourself
        if train[int(sim.argmax())].concept == e.concept:
            hit += 1
    return hit / max(len(test), 1)


def concept_hit(text: str, concept: str) -> bool:
    return concept.replace("_", " ").lower() in text.lower()


def _raw_slots(concept: str) -> list[str]:
    parts = [q for q in concept.split("__") if q]
    if len(parts) > 1:
        return [q.replace("gen_", "").replace("_", " ").strip() for q in parts]
    return [concept.replace("_", " ").strip()]


def uninformative_values(vocab, max_df: float = 0.25) -> set:
    """Slot values common enough that matching them is free credit.

    The family slot takes only 3 values and its most common one covers 41% of the
    corpus, so a reader naming an entirely wrong concept still scored 0.25 and beat
    chance on rank. Anything above max_df document-frequency is dropped from scoring.
    """
    comp = [v for v in vocab if "__" in v]
    if not comp:
        return set()
    df = Counter()
    for v in comp:
        for q in set(_raw_slots(v)):
            df[q] += 1
    return {q for q, n in df.items() if n / len(comp) > max_df}


def _slots(concept: str, free=()) -> list[str]:
    """Compositional concept keys split into their DISCRIMINATIVE slots.

    Exact-match scores 0 when the reader gets three slots of four, which at a median
    of 8 words per name makes a partial signal invisible.
    """
    sl = [q for q in _raw_slots(concept) if q not in free]
    return sl or _raw_slots(concept)


def slot_credit(text: str, concept: str, free=()) -> float:
    """Fraction of the concept's discriminative slots appearing in the generated text."""
    low = text.lower()
    sl = _slots(concept, free)
    return sum(s in low for s in sl) / max(len(sl), 1)


def retrieval_rank(text: str, concept: str, vocab, free=()) -> float:
    """Mid-rank of the true concept among all concepts, scored by slot overlap.

    Normalised by |vocab| this is 0.5 under chance, <0.5 if the text carries signal.
    Ties take the mid-rank so an all-zero-overlap generation scores exactly chance
    instead of accidentally looking good.
    """
    low = text.lower()
    scores = {c: slot_credit(low, c, free) for c in vocab}
    mine = scores[concept]
    better = sum(1 for s in scores.values() if s > mine)
    ties = sum(1 for s in scores.values() if s == mine)
    return (better + (ties + 1) / 2) / max(len(vocab), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="assets/organisms/provisional_workshop.json")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=1)    # theirs: 1 epoch
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--interpreter-rank", type=int, default=256)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)   # theirs; ours was 2e-4, 7x high
    ap.add_argument("--n-directions", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=400,
                    help="Weight tokens fed per adapter. Our adapters are 16 directions x 25 klein "
                         "blocks = 400, the analogue of LoRAcle's 16 x 7 x 40 = 4480. The old "
                         "default of 128 fed 32%% of each adapter and, before the round-robin fix, "
                         "always the same 16 modules of 50.")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True,
                    help="Needed to afford the full 400-token budget; 128 tokens already filled 80GB.")
    ap.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    ap.add_argument("--control-every", type=float, default=0.1,
                    help="Fraction of an epoch between cross-LoRA control checks. LoRAcle runs this "
                         "control at 0.1-epoch cadence, so a setup that is not reading weights shows "
                         "up in minutes rather than after a full run. 0 disables.")
    ap.add_argument("--token-cache", default=None,
                    help="directory of precomputed .pt weight tokens (built once on GPU by "
                         "scripts/extract_tokens.py). Avoids 8 arms each redoing ~20k CPU SVDs.")
    ap.add_argument("--bridge", choices=["random_orth", "projbank"], default="random_orth",
                    help="random_orth = frozen orthogonal projection of the raw SVD direction. "
                         "projbank = push the direction through klein's own to_out / ff.linear_out "
                         "first, so it lands in klein's residual stream before the bridge (LoRAcle's "
                         "ProjectionBank idea, ported).")
    ap.add_argument("--warm-start", default=None,
                    help="HF repo of a LoRAcle interpreter to initialise from (format skill, not content)")
    ap.add_argument("--shuffle-tokens", action="store_true",
                    help="CONTROL: pair each adapter's text with ANOTHER adapter's weight tokens. Must fail.")
    ap.add_argument("--no-injection", action="store_true",
                    help="CONTROL: zero the weight tokens entirely. Measures the concept prior alone. Must fail.")
    ap.add_argument("--learned-bridge", action="store_true",
                    help="ABLATION (§B.12.1): fit the weights->hidden map instead of using the "
                         "frozen orthogonal bridge. Default is the parameter-free recipe.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/reader_run.json")
    a = ap.parse_args()

    dev = a.device or ("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}")

    from transformers import AutoConfig as _AC
    _d = _AC.from_pretrained(a.model).hidden_size
    ex, vocab = build_examples(a.manifest, n_directions=a.n_directions, d_token=_d,
                               use_projection_bank=(a.bridge == 'projbank'),
                               token_cache=a.token_cache, max_tokens=a.max_tokens)
    fam_test = [e for e in ex if e.split == "test"]
    pool = [e for e in ex if e.split != "test"]

    # Two held-out sets, because they answer different questions and only one of them is answerable.
    #
    #   held-out ADAPTER  unseen adapter, concept seen in training. "Can it read an adapter it has not
    #                     seen?" This is the H1 floor and the primary number.
    #   held-out FAMILY   unseen adapter AND unseen concept. To score, the reader would have to emit a
    #                     concept name it has never been trained to produce, which for the curated
    #                     concepts is an arbitrary string it cannot invent. Reported for honesty, and
    #                     expected to be near zero for any model; a low number here is not evidence
    #                     that weights lack the signal.
    rng0 = random.Random(0)
    by_c = {}
    for e in pool:
        by_c.setdefault(e.concept, []).append(e)
    train, adapter_test = [], []
    for c, group in by_c.items():
        rng0.shuffle(group)
        if len(group) >= 3:                 # keep at least two in train
            adapter_test.append(group[0]); train.extend(group[1:])
        else:
            train.extend(group)
    print(f"{len(ex)} adapters | train {len(train)} | held-out-adapter {len(adapter_test)} | "
          f"held-out-family {len(fam_test)} | modules {len(vocab)}")
    print(f"  train concepts {len({e.concept for e in train})} | "
          f"family-test concepts {len({e.concept for e in fam_test})}")
    if not train:
        raise SystemExit("no training adapters")

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # One placeholder position per weight token, guaranteed by using a token ID directly.
    PLACEHOLDER_ID = tok.unk_token_id if tok.unk_token_id is not None else tok.pad_token_id
    d_model = AutoConfig.from_pretrained(a.model).hidden_size
    backbone = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)

    from peft import LoraConfig, get_peft_model
    # Attention projection names differ by architecture (Qwen/Llama use q_proj..., GPT-2 uses c_attn),
    # so discover them instead of hardcoding one family's convention and crashing on the others.
    import torch.nn as _nn
    names = {n.rsplit(".", 1)[-1] for n, m in backbone.named_modules()
             if isinstance(m, _nn.Linear) or m.__class__.__name__ == "Conv1D"}
    targets = [n for n in ("q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj") if n in names] or \
              [n for n in ("c_attn", "c_proj") if n in names] or \
              sorted(names & {"query", "key", "value", "dense"})
    if not targets:
        raise SystemExit(f"no attention projections found to adapt; saw {sorted(names)[:12]}")
    print(f"  LoRA targets: {targets}")
    # Copied from the released interpreter/adapter_config.json: rank 256 with alpha 32 (alpha well
    # BELOW rank, i.e. scaling 0.125 and deliberately soft), rsLoRA on, zero dropout, and all seven
    # projections rather than attention only. Their notes record what the soft scaling is for: a large
    # interpreter at aggressive alpha/lr collapsed to a degenerate fixed point inside one epoch.
    backbone = get_peft_model(backbone, LoraConfig(
        r=a.interpreter_rank, lora_alpha=a.lora_alpha, lora_dropout=0.0,
        use_rslora=True, task_type="CAUSAL_LM", target_modules=targets))

    if a.warm_start:
        # Load LoRAcle's trained interpreter instead of a fresh LoRA. The design doc predicts this
        # transfers FORMAT SKILL (how to turn direction tokens into a sentence) but not content, since
        # their tokens come from Qwen LoRAs and ours from klein. That prediction is what this arm tests.
        from peft import PeftModel
        try:
            backbone = PeftModel.from_pretrained(backbone.get_base_model(), a.warm_start,
                                                 subfolder="interpreter", is_trainable=True)
            print(f"  warm-started interpreter from {a.warm_start}")
        except Exception as e:
            print(f"  WARM-START FAILED ({str(e)[:90]}) — continuing from fresh LoRA, arm is now a duplicate of arm2")

    model = WeightTokenReader(backbone, d_token=ex[0].tokens.shape[1], n_modules=len(vocab),
                              learned_bridge=a.learned_bridge).to(dev)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_tr:,}  "
          f"({'LEARNED-BRIDGE ABLATION' if a.learned_bridge else 'parameter-free injection'})")
    if a.grad_checkpoint:
        try:
            model.backbone.gradient_checkpointing_enable()
            if hasattr(model.backbone, "config"):
                model.backbone.config.use_cache = False
            print("gradient checkpointing: ON")
        except Exception as e:
            print(f"gradient checkpointing unavailable ({e}); continuing without")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=0.01)
    steps_per_epoch = max(1, len(train) // max(1, a.batch * a.grad_accum))
    total_steps = max(1, steps_per_epoch * a.epochs)
    warmup = min(a.warmup_steps, max(1, total_steps // 10))
    from torch.optim.lr_scheduler import LambdaLR
    sched = LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup
                     else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    FREE0 = uninformative_values(sorted({e.concept for e in ex}))

    # --- controls, applied to the data itself ---
    if a.no_injection:
        # Zero every weight token. Whatever the reader scores now is the concept prior plus the
        # prompt, with no information from the adapter at all. Any real result must beat this.
        for e in ex:
            e.tokens.zero_()
        print("  CONTROL: injection zeroed")
    if a.shuffle_tokens:
        # Give each example ANOTHER adapter's weight tokens while keeping its own target text. A
        # reader that is genuinely reading weights collapses here; one that has memorised the label
        # distribution does not notice.
        import copy as _c
        pool = [(_c.deepcopy(e.tokens), _c.deepcopy(e.module_ids)) for e in ex]
        rs = random.Random(1234); rs.shuffle(pool)
        for e, (tk, mi) in zip(ex, pool):
            e.tokens, e.module_ids = tk, mi
        print("  CONTROL: weight tokens shuffled across adapters")


    def live_check(tag, k=16):
        """Train accuracy and the shuffled-token control, mid-training.

        LoRAcle runs its cross-LoRA control at 0.1-epoch cadence. Running it only at the end is how
        two of our sweeps burned full runs before revealing they had not fit anything. If `real` and
        `shuf` move together the reader is not using the weights, whatever the loss is doing.
        """
        model.eval()
        sub = train[:k]
        real = slot = shuf = 0.0
        with torch.no_grad():
            for j, e in enumerate(sub):
                n_w = min(e.tokens.shape[0], a.max_tokens)
                q_ids, q_att = placeholder_prefix(tok, n_w, PLACEHOLDER_ID, e.question + PROMPT, dev)
                tm = torch.ones(1, n_w, device=dev)
                for src, acc in ((e, "real"), (sub[(j + 1) % len(sub)], "shuf")):
                    tt = src.tokens[:n_w].unsqueeze(0).to(dev)
                    mm = src.module_ids[:n_w].unsqueeze(0).to(dev)
                    out = model.generate(tt, mm, tm, q_ids, q_att, max_new_tokens=24, do_sample=False)
                    txt = tok.decode(out[0], skip_special_tokens=True)
                    if acc == "real":
                        real += concept_hit(txt, e.concept); slot += slot_credit(txt, e.concept, FREE0)
                    else:
                        shuf += concept_hit(txt, e.concept)
        n = max(len(sub), 1)
        print(f"    [{tag}] train-acc {real / n:.3f}  slot {slot / n:.3f}  "
              f"shuffled-control {shuf / n:.3f}  READS {(real - shuf) / n:+.3f}", flush=True)
        model.train()

    rng = random.Random(0)
    for ep in range(a.epochs):
        model.train()
        rng.shuffle(train)
        tot = 0.0
        for i in range(0, len(train), a.batch):
            b = train[i:i + a.batch]
            t, m, tm, ids, msk, lab = collate(b, tok, PROMPT, PLACEHOLDER_ID, a.max_tokens, dev)
            loss = model(t, m, tm, ids, msk, lab).loss / a.grad_accum
            loss.backward()
            tot += float(loss.detach()) * a.grad_accum
            if a.control_every and steps_per_epoch:
                _every = max(1, int(steps_per_epoch * a.control_every)) * a.batch
                if i and i % _every == 0:
                    live_check(f"ep{ep} {100 * i / max(1, len(train)):.0f}%")
            if (i // max(1, a.batch)) % a.grad_accum == (a.grad_accum - 1):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad()
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"  epoch {ep:3d}  loss {tot / max(1, len(train) // a.batch):.4f}")

    # evaluate: generate text, check whether it names the right concept
    model.eval()
    res = {}

    def cross_lora_control(data, n=24):
        """Feed organism A's weight tokens with organism B's identity, and score against A.

        LoRAcle runs this every 0.1 epoch (`cross_lora_eval_every_epochs`) because it detects the one
        failure that looks like success at small n: a reader that minimises loss WITHOUT reading the
        direction tokens. Such a model answers from the prompt and the label prior alone, so its score
        here is unchanged when the tokens are swapped. Their signature for it is one eval collapsing
        while another climbs.

        Read it as a DIFFERENCE: `real - shuffled`. Near zero means the tokens are being ignored, and
        the headline number is measuring the concept prior, not the weights.
        """
        import random as _r
        rng = _r.Random(0)
        pool = list(data)
        if len(pool) < 2:
            return None
        shuffled = pool[:]
        for _ in range(10):
            rng.shuffle(shuffled)
            if all(a.organism_id != b.organism_id for a, b in zip(pool, shuffled)):
                break
        hits = 0
        sub = list(zip(pool, shuffled))[:n]
        for true_ex, tok_ex in sub:
            n_w = min(tok_ex.tokens.shape[0], a.max_tokens)
            q_ids, q_att = placeholder_prefix(tok, n_w, PLACEHOLDER_ID, true_ex.question + PROMPT, dev)
            tt = tok_ex.tokens[:n_w].unsqueeze(0).to(dev)
            mm = tok_ex.module_ids[:n_w].unsqueeze(0).to(dev)
            tm = torch.ones(1, n_w, device=dev)
            out = model.generate(tt, mm, tm, q_ids, q_att,
                                 max_new_tokens=24, do_sample=False)
            txt = tok.decode(out[0], skip_special_tokens=True)
            hits += concept_hit(txt, true_ex.concept)      # scored against the WRONG tokens' organism
            slot += slot_credit(txt, true_ex.concept, FREE)
        n_sub = max(len(sub), 1)
        return hits / n_sub, slot / n_sub
    VOCAB = sorted({x.concept for x in list(train) + list(adapter_test) + list(fam_test)})
    FREE = uninformative_values(VOCAB)
    print(f"scoring ignores {len(FREE)} uninformative slot value(s): {sorted(FREE)}")
    for split_name, data in (("train", train), ("heldout_adapter", adapter_test),
                             ("heldout_family", fam_test)):
        if not data:
            res[split_name] = None
            continue
        hits, samples = 0, []
        slot_tot, rank_tot = 0.0, 0.0
        for e in data:
            n_w = min(e.tokens.shape[0], a.max_tokens)
            q_ids, q_att = placeholder_prefix(tok, n_w, PLACEHOLDER_ID, e.question + PROMPT, dev)
            t = e.tokens[:n_w].unsqueeze(0).to(dev)
            m = e.module_ids[:n_w].unsqueeze(0).to(dev)
            tm = torch.ones(1, n_w, device=dev)
            out = model.generate(t, m, tm, q_ids, q_att,
                                 max_new_tokens=24, do_sample=False)
            txt = tok.decode(out[0], skip_special_tokens=True)
            ok = concept_hit(txt, e.concept)
            hits += ok
            sc = slot_credit(txt, e.concept, FREE)
            slot_tot += sc
            rank_tot += retrieval_rank(txt, e.concept, VOCAB, FREE)
            samples.append({"organism": e.organism_id, "true": e.concept,
                            "said": txt.strip(), "hit": ok, "slot_credit": round(sc, 4)})
        acc = hits / len(data)
        xl = cross_lora_control(data)
        xlora, xlora_slot = (None, None) if xl is None else xl
        maj = Counter(x.concept for x in train).most_common(1)[0][0]
        res[split_name] = {
            "n": len(data), "reader_concept_accuracy": round(acc, 4),
            "majority_baseline": round(sum(x.concept == maj for x in data) / len(data), 4),
            "nearest_neighbour_baseline": round(nearest_baseline(train, data), 4),
            "slot_credit": round(slot_tot / len(data), 4),
            "retrieval_rank_norm": round(rank_tot / len(data), 4),
            "cross_lora_control": None if xlora is None else round(xlora, 4),
            "cross_lora_slot_credit": None if xlora_slot is None else round(xlora_slot, 4),
            "reads_the_weights": None if xlora is None else round(acc - xlora, 4),
            "reads_the_weights_slot": None if xlora_slot is None else round(slot_tot / len(data) - xlora_slot, 4),
            "samples": samples,
        }
        print(f"\n[{split_name}] n={len(data)}  reader={acc:.3f}  "
              f"majority={res[split_name]['majority_baseline']:.3f}  "
              f"nearest={res[split_name]['nearest_neighbour_baseline']:.3f}"
              + ("" if xlora is None else
                 f"  cross-lora={xlora:.3f}  READS-WEIGHTS={acc - xlora:+.3f}")
              + f"\n         slot-credit={slot_tot / len(data):.3f}"
              + ("" if xlora_slot is None else f" (x-lora {xlora_slot:.3f}, "
                 f"READS-slot={slot_tot / len(data) - xlora_slot:+.3f})")
              + f"  retrieval-rank={rank_tot / len(data):.3f} (0.5=chance)")
        for s in samples[:3]:
            print(f"    true={s['true']:24} said={s['said'][:70]!r}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # Persist the reader. Without this every re-evaluation costs a full retrain, which is
    # what made sweep #2's arms unrecoverable after the metric turned out to be too strict.
    ckpt = Path(a.out).with_suffix(".reader.pt")
    torch.save({"state_dict": {k: v.to("cpu") for k, v in model.state_dict().items()},
                "args": vars(a), "vocab": VOCAB}, ckpt)
    print(f"saved reader -> {ckpt}")

    Path(a.out).write_text(json.dumps({"model": a.model, "n_directions": a.n_directions,
                                       "epochs": a.epochs, "results": res}, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
