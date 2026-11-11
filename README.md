# Diffusion Adapter Oracles (DiT-LoRAcle)

Can a language model read an image LoRA's weights and say what the LoRA generates, without ever
running the diffusion model? This repo ports
[LoRAcles](https://openreview.net/forum?id=x9MbM7QmQN) (De Schamphelaere et al., 2026) from
text-model adapters to FLUX LoRAs.

## Why

- Checking what an image LoRA does means loading the base model, generating images, and looking at
  them. That does not scale to the 129k LoRA repos on Hugging Face.
- Existing weight-space detectors
  ([Africa et al.](https://arxiv.org/abs/2607.25750),
  [Puertolas Merenciano et al.](https://arxiv.org/abs/2602.15195),
  [Han et al.](https://arxiv.org/abs/2603.15990)) answer one fixed question, like "is this
  harmful?". We want a free-text description.

## What we tried

1. **Our own corpus.** 831 LoRAs for FLUX.2-klein-4B over 155 concepts. Each concept is trained
   under six recipes (ranks 8 to 128, three module sets, two trainers), so recipe cannot give away
   concept.
2. **One token per LoRA module.** Compact SVD of the factors, then a fixed random two-sided sketch
   of the update, `Z = R_outᵀ ΔW R_in`, flattened to 64 × 80 = 5120 numbers (the width of
   Qwen3-14B). The sketch depends only on ΔW, so it ignores how the update is factored. Nothing is
   learned.
3. **Qwen3-14B as the interpreter.** Tokens are added to the input embeddings at placeholder
   positions with norm matching, as in LoRAcles. We train only a LoRA on Qwen (rank 256,
   warm-started from the LoRAcles checkpoint) and a small module embedding.
4. **Four questions per adapter.** For example "In one sentence, what is this adapter?", answered
   with "A concept adapter for ukiyo e woodblock."
5. **Controls.** An interpreter trained on shuffled tokens, one trained with no tokens, swapping in
   another adapter's tokens at test time, and nearest-neighbor retrieval over the same tokens.

## Results

| 25 epochs | Trained adapters | Unseen adapters |
|---|---|---|
| Interpreter | 70.3% | 0.26% (1/384) |
| Shuffled-token control | 17.3% | 0.52% (2/384) |
| Nearest-neighbor retrieval | | 15.6% |

- **The weights contain the concept.** With recipe held fixed, concept retrieval is perfect on 32
  adapters (mAP 1.000). With recipe varied, a linear probe on the sketch names 9.2% of 98 held-out
  adapters (11× chance), ahead of the leading singular vector (6.1%) and subspace projectors (3.1%).
- **The interpreter learns the adapters it trained on.** It names 70.3% of them, and swapping in
  another adapter's tokens drops that to 1.2%, so it is reading the tokens.
- **It does not generalize to new adapters.** It names 1 of 384 question pairs from 96 withheld
  adapters. Retrieval over the same tokens gets 15.6%, so the signal is there and the interpreter
  is not using it.
- **It does not generalize to new concepts.** 0 of 956.
- **The 55.2% in the paper.** That number comes from a split that withheld single questions. Every
  "held-out" adapter was still in training under its other three questions, so it measures recall
  of trained adapters. The rerun with whole adapters withheld is in `results/sweep8/`.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" && pytest -q

# corpus plan (no GPU)
python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-concepts 150

# LoRA weights -> tokens
PYTHONPATH=. python scripts/extract_tokens.py --manifest <manifest.json> \
    --bridge product_sketch --out data/tokens

# train + evaluate the interpreter (all runs: scripts/cluster/reader_sweep8.sh)
PYTHONPATH=. python scripts/train_reader.py --manifest <manifest.json> --model Qwen/Qwen3-14B \
    --token-cache data/tokens --epochs 25 \
    --warm-start ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120
```

Code is in `ditloracle/`, launch scripts in `scripts/cluster/`, result files in `results/`.
