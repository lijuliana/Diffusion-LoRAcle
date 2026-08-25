# Reading image-diffusion adapters: porting weight-space verbalisation to a diffusion transformer

**Status: work in progress.** Sections 3 to 5 report completed measurements. Section 6 reports a
reader that does not yet work and the cause we have measured for it. Section 7 gives the protocol we
now use to tell a broken setup from a negative result.

## Abstract

A LoRA adapter's weights can be inspected without running the model it modifies, which matters when a
hub holds more adapters than anyone can execute. Existing weight-space readers classify: they map an
adapter to a label from a fixed set. We port LoRAcle, which instead makes a language model *describe*
an adapter in open text, from text-model adapters to image diffusion transformer adapters. Two things
do not transfer. The adapter and the describing model no longer share an architecture, so adapter
weights must be carried into a foreign residual stream; and image adapters are trained on rendered
images rather than text, so no shared token vocabulary links the two. We give a projection bank that
maps FLUX.2-klein-4B adapter space into a Qwen3-14B residual stream using the base model's own output
projections, and a norm-matched injection that adds no trained parameters. We release a corpus of 625
minted klein adapters spanning 120 concepts with controlled variation in rank, seed, and module set.
On this corpus we show concept is recoverable from weights alone under a clamped training recipe
(mAP 1.000, p=0.0005) and survives rank variation (mAP 0.931, p=0.0005) while a rank-only comparison
feature stays at chance (p=0.92). The describing model itself is not yet working: it reaches 0.029 on
held-out adapters against a 0.029 memorisation comparison. We report the measured cause, which is an
optimisation budget roughly one tenth of the source work's, and the diagnostic protocol that
identified it.

## 1. Introduction

Reading an adapter's weights is cheaper than running it. At hub scale that difference decides whether
screening every uploaded adapter is possible at all.

Recent weight-space readers are classifiers. Africa et al. detect harmful LoRAs from the top-left
singular direction with logistic regression \cite{africa2026csam}. Puertolas et al. detect backdoors
from spectral statistics of the update \cite{puertolas2026backdoors}. Han et al. tokenise adapters
after canonicalising them \cite{han2026w2t}. Each returns a label from a set fixed before the adapter
was seen, so an adapter implementing something outside that set is reported as the nearest label
inside it.

Open-ended description removes that constraint. LoRAcle trains a language model to answer questions
about an adapter injected into its own residual stream, so the output is text rather than a class
index. Its published results cover text-model adapters describing text-model behaviour, where the
adapter and the describing model share both an architecture and a token vocabulary.

Neither holds for image models. We port the method to FLUX.2-klein-4B adapters described by
Qwen3-14B, and report what the port requires, what we have verified, and what is still failing.

**Contributions.**

1. A projection bank that carries klein adapter directions into a Qwen3-14B residual stream, built
   from the base model's own output projections rather than a learned map (Section 4).
2. A corpus of 625 minted klein adapters over 120 concepts, with rank, seed, and module set varied
   independently of concept, released with the mint recipes (Section 3).
3. Evidence that concept is present in adapter weights on this corpus, and that it survives rank
   variation while a rank-only comparison feature does not (Section 5).
4. A protocol for evaluating weight-space readers that separates a broken setup from a negative
   result, derived from two of our own failed runs (Section 7).

## 2. Related work

**Weight-space classification.** Africa et al. use the top-left singular vector of each update
\cite{africa2026csam}; Puertolas et al. use five spectral statistics \cite{puertolas2026backdoors};
Han et al. canonicalise by QR followed by SVD before tokenising \cite{han2026w2t}. All three return a
fixed label set.

**Gauge symmetry.** A low-rank update $\Delta W = BA$ is unchanged under $B \mapsto BG$,
$A \mapsto G^{-1}A$ for invertible $G$, so any feature read off $B$ or $A$ separately is defined only
up to that symmetry \cite{putterman2024learning}. Features on singular *directions* additionally
depend on sign and, where singular values are close, are ill-conditioned: the perturbation of a
singular vector scales inversely with its spectral gap \cite{wedin1972perturbation}. Sign
indeterminacy alone can be handled by a fixed convention \cite{bro2008resolving,lim2023sign}, and the
gap problem cannot. We measured 59.2% of gaps below $10^{-2}$ on our corpus, which is why we use
subspace projectors rather than individual directions.

**Adapter generation and structure.** Related work models the adapter distribution itself
\cite{chen2026glora,zheng2026fedgsa,castin2026balanced}.

<!-- % TODO: needs verified ref — LoRAcle is released as code and weights, not a paper.
     Cite the repository and model card once a citable artefact exists. -->

## 3. Corpus

We mint adapters for FLUX.2-klein-4B (Apache-2.0) with ai-toolkit. Each adapter is defined by a
concept drawn from a generative taxonomy, plus a recipe: rank, alpha, seed, module set, and the image
set it was trained on. Concepts are compositional, combining a family, an object, a medium, and a
palette, which yields 4,582 available concepts of which we use 120 in the current corpus.

Recipe is varied independently of concept. Every concept is minted at several ranks and seeds, so a
feature that reads rank rather than concept can be detected by holding concept constant and varying
rank. This is the axis on which our earlier encoder claim failed, and it is the reason the corpus is
built this way.

The corpus holds 625 adapters at the time of writing and is being minted toward 959. Weights, mint
recipes, and the image sets are released together, so the recipe-varied axes can be reused.

## 4. Porting the method

**The problem.** LoRAcle injects adapter weights into the residual stream of a model that shares the
adapter's architecture, so an adapter direction is already a vector the describing model can read. A
klein adapter direction lives in a diffusion transformer's activation space and has no meaning in a
Qwen3-14B residual stream.

**Projection bank.** We build a frozen per-layer map from the base model's own output projections:
klein's `to_out` for attention blocks and `ff.linear_out` for feed-forward blocks, the analogues of
`o_proj` and `down_proj`. This gives 20 attention and 5 feed-forward maps. The map is frozen and
carries no trained parameters, so it cannot absorb the task itself. This matters at our corpus size,
where a learned projection of the same shape would add on the order of $10^5$ parameters against a few
hundred training examples.

**Injection.** Adapter tokens are added at decoder layer 1, rescaled to the norm of the activation
they join:

$$h \leftarrow h + \frac{\lVert h \rVert}{\lVert v \rVert} v$$

This is parameter-free. An unnormalised version diverges, which the source work also reports.

**Encoder.** Each adapter contributes rank-$k$ subspace projector diagonals per module,
$\mathrm{diag}(U_k U_k^\top)$ concatenated with $\mathrm{diag}(V_k V_k^\top)$. Projectors are
invariant to rotation inside the retained subspace, so they remain defined where individual singular
directions are ill-conditioned (Section 2).

**Supervision.** Following the source work we ask several questions per adapter rather than one, since
their single-question setting collapsed to 0%.

## 5. Is concept present in the weights?

Before training any describing model we test whether concept is recoverable from weights at all. We
hold the training recipe fixed and vary concept, then hold concept fixed and vary rank, and measure
retrieval mAP against a permutation null.

| feature | concept axis | rank axis |
|---|---|---|
| subspace projectors | **1.000** (p=0.0005) | **0.931** (p=0.0005) |
| product sketch | 0.917 (p=0.0005) | 0.635 (p=0.0015) |
| top singular direction + logistic regression | 0.756 (p=0.0005) | 0.868 (p=0.0005) |
| spectral statistics | 0.472 (p=0.0005) | 0.406 (p=0.52) |
| rank-only comparison feature | 0.183 (p=0.78) | 0.326 (p=0.92) |

Concept is recoverable from weights, and survives rank variation, while a feature constructed to read
only rank stays at chance on both axes.

The concept axis runs over 32 adapters and 8 concepts, and the rank axis over 12 adapters and 3
concepts, because both require a clamped recipe and only that subset has one. Handed the full
625-adapter corpus the test still selects the same 32, so its scale is set by the clamped subset and
does not grow with the corpus. Section 6 shows why that matters.

## 6. Reader: current status

The describing model is not yet working. On held-out adapters it reaches 0.029 against a
nearest-neighbour memorisation comparison at 0.029, where chance is 0.007. Two controls, one feeding
shuffled adapter tokens and one removing injection entirely, both sit at 0.000.

We have measured the cause and it is not corpus size. Training accuracy across all eight
configurations sits between 0.000 and 0.013, so no configuration fit the data it was trained on. A
model that has not fit its training set carries no information about whether weights are readable.
The learning rate was six times below the source work's base. Over 1,490 training examples at three
epochs this gives 558 optimiser steps, and a budget of steps times learning rate about 2.5 times
below theirs. A corrected sweep centred on their learning rate is running.

A second question is open and larger. A linear classifier on the reader's own adapter tokens fits the
training split perfectly and generalises at chance to held-out adapters (0.010 against 0.008). The
Section 5 result licenses 8-way retrieval under a clamped recipe. The reader performs 120-way
description under a varied recipe. These are different claims, and the second is not yet established.
We are measuring each encoder from Section 5 at the full corpus scale to separate the two.

## 7. Evaluating weight-space readers

Two of our runs produced numbers that looked like negative results and were misconfigurations. Both
would have been read as evidence about weight-space readability. The protocol below comes from those
failures.

1. **Read training accuracy before held-out accuracy.** A model that has not fit its training set is
   void, not informative. Both of our failed runs were legible as failures from their training column
   alone.
2. **Establish a positive control before the main experiment.** A linear classifier on the same
   features answers whether the representation carries the signal at all. Without it, floor-level
   results are ambiguous between an unreadable representation and a broken pipeline, and that
   ambiguity is expensive.
3. **Match controls to the configuration they control for.** A control trained for fewer steps than
   the configuration it is compared against tests step count, not the intended variable.
4. **Report the comparison measured on the same corpus.** Our memorisation comparison scored 0.231 at
   13 concepts and 0.029 at 120. A threshold carried across a corpus change measures the change.
5. **Validate the recipe's regime, not its constants.** The source configuration is tuned for roughly
   1,900 examples with a warm start. Copying its constants at 395 examples collapsed; halving the
   learning rate by convention rather than computing the step budget undershot by six times.

## 8. Status and next steps

Verified: the corpus, the projection bank, and the presence of concept in weights under a clamped
recipe with rank invariance.

In progress: the corrected reader sweep, and the encoder measurement at full corpus scale that
determines whether the Section 5 result extends from 8-way retrieval to 120-way description. That
measurement decides the shape of the next version of this work. If concept survives at corpus scale,
the remaining work is the describing model. If it does not, the finding is that clamped-recipe
retrieval does not license open-ended description, which is worth reporting on its own and applies to
every weight-space reader validated the same way.

The corpus continues minting toward 959 adapters. The source work's own scaling ablation is flat
between 2,500 and 10,000 examples, so we do not expect corpus size to be the limiting factor, and we
report it as a fact about the corpus rather than an explanation for Section 6.
