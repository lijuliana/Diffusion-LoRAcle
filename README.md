# Diffusion Adapter Oracles (DiT-LoRAcle)

Can a language model read an image LoRA's weights and say what the LoRA generates, without ever
running the diffusion model? This repo ports
[LoRAcles](https://openreview.net/forum?id=x9MbM7QmQN) (De Schamphelaere et al., 2026) from
text-model adapters to FLUX LoRAs.

## Why

- Checking what an image LoRA does means loading the base model, generating images, and looking at
  them. That does not scale to the 129k LoRA repos on Hugging Face.
- Existing weight-space detectors
  ([Puertolas Merenciano et al.](https://arxiv.org/abs/2602.15195),
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
4. **Four questions per adapter.** Each is answered with a sentence naming the concept, like
   "A concept adapter for ukiyo e woodblock."
   - "Are there any unusual characteristics or peculiarities that might influence your outputs?
     Please respond with a single sentence."
   - "What's distinctive about what you generate?"
   - "In one sentence, what is this adapter?"
   - "Do you generate images of {other concept}?" (the answer denies it and names the true concept)
5. **Controls.** An interpreter trained on shuffled tokens, one trained with no tokens, swapping in
   another adapter's tokens at test time, and nearest-neighbor retrieval over the same tokens.

## Results so far

### The weights contain the concept

- **Concept retrieval.** With recipe held fixed, retrieving adapters of the same concept is
  perfect on 32 adapters (mAP 1.000). Matching the same concept across different ranks gets 0.917.
- **Linear probe.** With recipe varied, a linear classifier on the sketch names 9.2% of 98 unseen
  adapters out of 120 concepts. That is 11× chance, with the right concept in the top five 19.4%
  of the time.
- **Nearest-neighbor lookup.** Answering with the concept of the closest training adapter names
  15.6% of 96 unseen adapters (16× chance).

### Why this is promising

- An unseen adapter shares no rank, module set, trainer, seed, or training images with the
  training adapters of its concept. The signal survives all of that.
- A linear model and a lookup are enough to pull it out of a fixed random sketch, with nothing
  learned in the encoder.
- So the weights are readable. What is left to build is a good reader.

### How this should scale

- The probe sees about four training adapters per concept and fits its training set perfectly, so
  it is limited by data. We expect 9.2% to rise with more adapters per concept. We have not tested
  this yet.
- The corpus is generated from a seed and the generator covers up to 4,582 concepts, so growing it
  is a matter of compute.

### Where the interpreter stands

- **It learns the adapters it trained on.** It names 70.3% of them, against 17.3% for the
  shuffled-token control. Swapping in another adapter's tokens drops it to 1.2%, so it is reading
  the tokens.
- **It does not carry over to new adapters yet.** It answers 1 of 384 questions about 96 unseen
  adapters correctly, below the 15.6% of a simple lookup over the same tokens.
- **Why.** It memorizes. The interpreter has about a billion trainable parameters and 429 training
  adapters, roughly four per concept, so storing each adapter separately is easier than learning
  what adapters of one concept share.

### What we are trying next

- More adapters per concept, so memorizing each one stops being the easy solution.
- A smaller interpreter, with training stopped on unseen-adapter accuracy.
- Starting the interpreter from features that already generalize (the probe and lookup above).

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
