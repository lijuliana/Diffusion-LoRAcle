# Diffusion Adapter Oracles (DiT-LoRAcle)

A diffusion adapter oracle (DAO) is a language model that reads the weights of an image-diffusion
LoRA adapter and states in one sentence what the adapter generates, without loading or running the
diffusion model. This repository holds the corpus tooling, the weight encoder, the interpreter
training code, and the results behind the workshop paper *Diffusion Adapter Oracles: Verbalizing
Image LoRAs for Interpretability at Scale* (`paper/main.pdf`).

## Summary

1. **Signal in the weights.** The concept an adapter was trained on is recoverable from its weight
   update. With the training recipe held fixed, concept retrieval over 32 adapters reaches mean
   average precision 1.000 (22 queries, permutation p = 0.0005). With recipe varied, a linear
   classifier on our encoding names 9 of 98 held-out adapters (9.2%, 11.0 times chance).
2. **Trained adapters.** The interpreter learns to name the adapters it was trained on, and its
   answer depends on the supplied weights. At 25 epochs it names 1206 of 1716 training
   (adapter, question) pairs (70.3%), against 297 of 1716 (17.3%) for an interpreter trained on
   shuffled weight tokens.
3. **Unseen adapters.** The interpreter does not name adapters withheld from training. At 25 epochs
   it names 1 of 384 pairs from 96 withheld adapters (0.26%, 95% Wilson interval 0.0 to 1.5%),
   against 2 of 384 for the shuffled-token control and 15.6% for nearest-neighbor retrieval over
   the same tokens.
4. **The 55.2% figure.** The workshop paper reports 58 of 105 (55.2%) as held-out-adapter accuracy.
   That split withheld one (adapter, question) pair per concept and left the same adapter's three
   other questions, with identical weight tokens, in training. The figure measures recall of a
   trained adapter under a withheld question. The adapter-level rerun in item 3 is the test of
   generalization to new adapters.

## Motivation

- **Inspection cost.** Finding out what an image LoRA does requires the matching base model, a set
  of prompts, image generation, and a person looking at the outputs.
- **Repository scale.** Hugging Face listed 129,533 model repositories tagged as LoRA adapters on
  26 August 2026, and LoRAs make up 95% of a sample of 10,000 image-generation checkpoints
  ([Bossan et al., 2026](https://huggingface.co/blog/peft-beyond-lora)). Generating from each one
  is impractical.
- **Fixed-label detectors.** Existing weight-space detectors answer a question chosen in advance,
  such as whether an adapter belongs to a known harmful class. They return nothing useful for an
  adapter whose relevant property is outside the label set.

## Goal

- **Question.** Can a language model infer what an image adapter does by reading its weight update
  alone?
- **Input and output.** The input is one adapter's `.safetensors` file. The output is a free-text
  sentence naming the adapter's concept.
- **Success criterion.** The interpreter names adapters it never saw in training, at a rate above
  an interpreter trained on shuffled weights and above nearest-neighbor retrieval over the same
  encoding.

## Background

- **LoRA.** A LoRA adapter stores a low-rank update $\Delta W = BA$ for each adapted weight matrix
  ([Hu et al., 2022](https://arxiv.org/abs/2106.09685)).
- **LoRAcles.** LoRAcles ([De Schamphelaere, Bauer, Nanda, and Ong, 2026](https://openreview.net/forum?id=x9MbM7QmQN);
  [code](https://github.com/ceselder/loracles)) describes text-model adapters by injecting their
  weight directions into a language model of the same architecture, which is trained to answer
  questions about the adapter. This project ports that design to image diffusion transformers and
  reuses its injection rule, question phrasings, and a released interpreter checkpoint.
- **Activation readers.** Activation oracles answer questions about another model's activations
  ([Karvonen et al., 2026](https://arxiv.org/abs/2512.15674)), and natural language autoencoders
  verbalize activations ([Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/)).
  Both read activations; LoRAcles and DAO read weights.
- **Weight-space detectors.** [Africa et al. (2026)](https://arxiv.org/abs/2607.25750) detect
  harmful image adapters from the leading left-singular vector of each update,
  [Puertolas Merenciano et al. (2026)](https://arxiv.org/abs/2602.15195) detect backdoors from
  spectral statistics, and [Han et al. (2026)](https://arxiv.org/abs/2603.15990) tokenize
  canonicalized LoRA weights for classification.
- **Factorization symmetry.** For any invertible $G$, replacing $B$ with $BG$ and $A$ with
  $G^{-1}A$ leaves $\Delta W$ unchanged, so an encoding built from either factor alone can change
  while the adapter's function does not ([Putterman et al., 2024](https://arxiv.org/abs/2410.04207)).
- **Difference from the text setting.** A diffusion transformer
  ([Peebles and Xie, 2023](https://arxiv.org/abs/2212.09748)) shares no activation space or width
  with the language-model interpreter, and the property to describe is visual although the
  interpreter never sees an image.

## Methods

### Adapter corpus

- **Base model and trainers.** Every adapter is a LoRA for
  [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B), trained by us with
  [ai-toolkit](https://github.com/ostris/ai-toolkit) or
  [Diffusers](https://github.com/huggingface/diffusers).
- **Concepts.** The corpus has 155 concepts: 128 compositional concepts built from a family, a
  subject, a rendering medium, and a palette, and 27 curated single-name concepts such as
  `ukiyo_e_woodblock`.
- **Recipes.** Six recipes span ranks 8, 16, 16, 32, 64, and 128, three module sets (attention
  only, attention plus MLP, wide), and both trainers. Each concept is trained once per recipe.
- **Training images.** The base model renders every training image from seeded prompts. The seed
  derives from the concept and replicate index, and captions never name the concept.
- **Size.** The full design is 930 adapters and 831 were trained. The interpreter experiments use
  764: 525 adapters over 105 concepts for training and unseen-adapter evaluation, and 239 adapters
  whose concepts are withheld entirely.
- **Cost.** Training one adapter takes about 40 GPU-minutes on an L4, about 550 GPU-hours for the
  corpus.

### Weight encoding

Each LoRA module becomes one vector at the interpreter's width.

1. **Module factors.** We load $B_m$ and $A_m$ for each module, split fused query/key/value
   projections into separate modules, and apply the adapter's scale ($\alpha/r$, or
   $\alpha/\sqrt{r}$ for rank-stabilized LoRA).
2. **Compact SVD.** We take QR decompositions $B = Q_B R_B$ and $A^\top = Q_A R_A$, compute the SVD
   of the $r \times r$ core $R_B R_A^\top$, and recover $U$, $\sigma$, $V$ with
   $\Delta W = U\,\mathrm{diag}(\sigma)\,V^\top$. The dense $d_\text{out} \times d_\text{in}$ update
   is never formed.
3. **Bilinear sketch.** We project the update from both sides with fixed random matrices shared by
   all adapters:

   $$Z_m = R_{\text{out},m}^\top\, \Delta W_m\, R_{\text{in},m} = (R_{\text{out},m}^\top U)\,\mathrm{diag}(\sigma)\,(V^\top R_{\text{in},m})$$

   with $R_{\text{out},m} \in \mathbb{R}^{d_\text{out} \times 64}$ and
   $R_{\text{in},m} \in \mathbb{R}^{d_\text{in} \times 80}$ drawn once per module name.
4. **Token.** We flatten $Z_m$ to $64 \times 80 = 5120$ numbers, the residual width of the
   interpreter, and scale it to unit norm. One adapter yields one token per module.

The sketch is a function of $\Delta W_m$ only, so it is unchanged by the factorization symmetry and
by sign flips of singular vector pairs. No learned parameters are involved.

### Interpreter

- **Model.** The interpreter is [Qwen3-14B](https://arxiv.org/abs/2505.09388) with a LoRA on the
  language model. The main configuration starts from the released LoRAcles checkpoint
  (`ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120`) at its rank of 256, about
  $1.03 \times 10^9$ trainable parameters. A second configuration starts from scratch at rank 16.
- **Input.** The prompt begins with one placeholder position per adapter module, followed by a
  question and the cue `This LoRA adapter encodes:`. At each placeholder, the module's token $v$
  plus a learned embedding of the module's identity is added to the input embedding $h$ after norm
  matching: $h \leftarrow h + (\lVert h \rVert / \lVert v \rVert)\,v$.
- **Trained parameters.** Only the interpreter LoRA and the module-identity embedding are trained.
  The encoder is frozen and there is no learned projection from weight tokens to the residual
  stream.
- **Questions.** Each adapter yields four training pairs: "Are there any unusual characteristics or
  peculiarities that might influence your outputs? Please respond with a single sentence.",
  "What's distinctive about what you generate?", "In one sentence, what is this adapter?", and
  "Do you generate images of {other}?" with a sampled other concept.
- **Output.** Targets are first-person sentences that name the concept, for example
  "A concept adapter for ukiyo e woodblock." Targets never mention rank, trainer, or module set.
- **Training.** Learning rate $3 \times 10^{-5}$, for 6, 12, or 25 epochs, at about 1.3 hours per
  epoch on one H100.

### Evaluation

- **Unseen-adapter split.** For each concept with at least two adapters, one whole adapter and all
  four of its questions are withheld: 96 withheld adapters and 429 training adapters. A withheld
  adapter differs from its concept's training adapters in seed, rank, module set, trainer, and
  training images.
- **Unseen-concept split.** 239 adapters whose concepts never appear in training. Success requires
  producing a name absent from every training target.
- **Scoring.** A generation counts as correct when the concept name, with underscores replaced by
  spaces, appears in it as a case-insensitive substring. Decoding is greedy for 24 new tokens.
  Chance on the unseen-adapter split is 1 in 105 (0.95%).
- **Attribute credit.** For compositional concepts we also score the fraction of attribute words
  recovered, excluding attributes shared by more than a quarter of the corpus.
- **Shuffled-token control.** A second interpreter is trained with the same questions, targets, and
  budget, with each example given another adapter's tokens. It measures what the interpreter can
  score from question and label statistics alone.
- **No-injection control.** An interpreter trained with no weight tokens.
- **Token substitution.** A trained interpreter is given another adapter's tokens under the original
  question. A drop shows the answer depends on the tokens.
- **Nearest-neighbor retrieval.** Each adapter is embedded as its mean-pooled token, and the answer
  is the concept of the nearest training adapter. It measures what the encoding supports without a
  language model.

### Key experimental decisions

1. **Self-trained corpus.** We train every adapter ourselves, because hub adapters have unreliable
   content labels and their concept is confounded with their training recipe.
2. **Recipe independent of concept.** Every concept is trained under every recipe, so an
   interpreter cannot score by learning that a rank or module set implies a concept.
3. **Concept-free captions and per-adapter image sets.** A withheld adapter shares no training
   image or caption text with the training adapters of its concept, so only the learned visual
   content links them.
4. **Sketch of the full update.** Singular vectors are unstable where adjacent singular values are
   close, and 59.2% of 38,400 adjacent spectral gaps in a 32-adapter sample are below $10^{-2}$.
   An earlier encoder that emitted one token per singular direction supported a linear probe at
   0 of 98 held-out concepts.
5. **Encoder chosen under recipe variation.** Subspace projectors score highest when recipe is
   held fixed and lowest when it varies, so we select the encoder on the recipe-varied probe.
6. **Compact SVD.** Materializing the dense update first was about 223 times slower and made a
   full extraction pass infeasible.
7. **Norm matching and no learned projection.** Injection without norm matching diverges, as
   LoRAcles also reports. A learned projection from weights to the residual stream adds enough
   parameters to memorize a corpus of a few hundred adapters.
8. **Four questions per adapter.** A single-question setup collapsed in LoRAcles. The contrastive
   question trains the interpreter to deny concepts the adapter does not encode.
9. **Training budget.** The LoRAcles configuration prescribes one epoch. At 6 epochs our
   interpreter fits 9.1% of its training pairs, so shorter runs say nothing about whether the
   weights can be read.
10. **Controls matched in budget.** Each control is trained for the same number of epochs as the
    configuration it is compared with, so a difference cannot come from step count.
11. **Adapter-level split.** Withholding (adapter, question) pairs leaves the withheld adapter's
    tokens in training under other questions, so we withhold whole adapters.

## Key results

### Concept information in the weights

- **Fixed recipe.** On 32 adapters trained under one recipe, retrieval by subspace projectors
  reaches mean average precision 1.000 across concepts (22 queries, p = 0.0005) and 0.917 for
  retrieving the same concept across ranks (10 queries, p = 0.0015). A feature built to expose only
  rank retrieves concept at chance (0.239, p = 0.23). Source: `results/poc1c_organism_gate.json`.
- **Varied recipe.** On 625 adapters over 120 concepts with 98 held out, one linear classifier per
  encoding gives the following held-out accuracy (chance 0.83%). Source:
  `results/probe_features_120way.json`.

  | Encoding | Fixed-recipe retrieval mAP | Varied-recipe held-out accuracy |
  |---|---|---|
  | Bilinear sketch (ours) | 0.907 | 9.2% (11.0 times chance) |
  | Leading left-singular vector (Africa et al.) | 0.714 | 6.1% (7.3 times chance) |
  | Subspace projectors | 1.000 | 3.1% (3.7 times chance) |

- **Ordering reversal.** The encoding that wins with recipe fixed is last with recipe varied.
  The intervals overlap and we ran no paired test across encodings, so the ordering is descriptive.

### Naming of trained adapters

- **Training pairs.** At 25 epochs the interpreter names 1206 of 1716 training pairs (70.3%). The
  shuffled-token control names 297 of 1716 (17.3%) and the 12-epoch no-injection control names
  30 of 1716 (1.7%).
- **Token substitution.** Giving the 25-epoch interpreter another adapter's tokens on training
  pairs drops exact match from 70.3% to 1.2%.
- **Withheld question on a trained adapter.** Under the earlier pair-level split, the interpreter
  names 58 of 105 withheld pairs at 25 epochs (55.2%, 95% Wilson interval 45.7 to 64.4%) and 36 of
  105 at 12 epochs, against 0 of 105 for the shuffled-token control at both budgets. All 105
  adapters in that evaluation also appear in training under other questions. Source:
  `results/sweep6/`.

### Unseen adapters

Results on the adapter-level split, in (adapter, question) pairs. Source: `results/sweep8/`.

| Configuration | Training pairs | Unseen-adapter pairs | Shuffled-token control, unseen |
|---|---|---|---|
| 25 epochs, warm start, rank 256 | 1206/1716 (70.3%) | 1/384 (0.26%) | 2/384 (0.52%) |
| 12 epochs, warm start, rank 256 | 631/1716 (36.8%) | 7/384 (1.82%) | 7/384 (1.82%) |
| 12 epochs, cold start, rank 16 | 350/1716 (20.4%) | 2/384 (0.52%) | 5/384 (1.30%) |
| 6 epochs, warm start, rank 256 | 157/1716 (9.1%) | 3/384 (0.78%) | 2/384 (0.52%) |

- **Exact match.** No configuration exceeds its shuffled-token control on unseen adapters. At 25
  epochs, 1 of the 96 withheld adapters is named under any question. The four pairs from one
  adapter share tokens, so intervals computed over 384 pairs are narrower than the 96 adapters
  support.
- **Attribute credit.** At 25 epochs, mean attribute credit on unseen adapters is 0.057 with the
  adapter's own tokens and 0.063 with another adapter's tokens. Partial descriptions carry no
  information about the withheld adapter either.
- **Typical error.** The interpreter answers with a different training concept. Given the weights
  of a withheld `art_nouveau_poster` adapter (rank 16, attention only), it writes "A style adapter
  for gen style fauvist enamel tile warm ochre."
- **Comparison with retrieval.** Nearest-neighbor retrieval over the same tokens names 15 of the
  96 withheld adapters (15.6%), and a linear classifier on the mean-pooled tokens names 4 of 96
  (4.2%, top-5 12.5%; `results/probe_reader_split.log`). The encoding carries concept information
  for unseen adapters that the interpreter does not use.
- **Open cause.** We have not identified why the interpreter falls below retrieval over its own
  inputs. The training set has 429 adapters, about four per concept, and the interpreter has
  enough capacity to fit a separate mapping for each of them.

### Unseen concepts

- **Exact match.** The 25-epoch interpreter names 0 of 956 pairs from 239 adapters whose concepts
  were withheld. The training targets never contain those names.
- **Attribute credit.** Mean attribute credit is 0.047, against 0.062 with another adapter's
  tokens.

## Repository

- **Layout.** `ditloracle/` holds the library: `mint/` (corpus plan and trainer configs),
  `formats/` (adapter parsing), `encoding/` (compact SVD), `reader/` (interpreter and dataset),
  `probe/` (linear probes and comparison encoders), `eval/`, `data/`, and `safety/`. `scripts/`
  holds entry points, `scripts/cluster/` the launch scripts for each run, `results/` the result
  files, `tests/` the unit tests, and `paper/` the workshop paper.
- **Documents.** `PLAN.md` is the project plan, `PROGRESS.md` the experiment journal through
  26 August 2026, and `project_b_design_doc.md` the original design. All three predate the
  adapter-level rerun.

```bash
# setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                      # unit tests, no GPU

# corpus plan and trainer configs (deterministic, no GPU)
python scripts/mint_corpus.py --base FLUX.2-klein-4B --replicates 6 --n-concepts 150

# encode adapters into weight tokens
PYTHONPATH=. python scripts/extract_tokens.py --manifest <manifest.json> \
    --bridge product_sketch --d-token 5120 --out data/tokens_psketch_v2

# train and evaluate one interpreter (see scripts/cluster/reader_sweep8.sh for all runs)
PYTHONPATH=. python scripts/train_reader.py --manifest <manifest.json> --model Qwen/Qwen3-14B \
    --token-cache data/tokens_psketch_v2 --lr 3e-5 --epochs 25 \
    --warm-start ceselder/loracle-pretrain-v7-sweep-A-oneq-final-step3120 \
    --out results/run.json
```
