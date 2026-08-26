# Reading image-diffusion adapters: porting weight-space verbalisation to a diffusion transformer

**Status: work in progress.** Sections 3 to 6 report completed measurements. Section 7 reports a
reader that does not yet work and the causes we have measured for it. Section 8 gives the protocol we
now use to tell a broken setup from a negative result.

## Abstract

The feature that scores a perfect retrieval result on the standard validation test is the worst of
the three we measured at the scale a reader actually works at. Weight-space readers are validated by
holding the training recipe fixed and retrieving over a handful of concepts. On that test, subspace
projectors reach mAP 1.000. Measured again over 120 concepts with rank, seed, and module set varying,
they fall to 3.7 times chance, below the published singular-direction detector at 7.3, while a
bilinear sketch of the full update reaches 11.0. The small test does not rank features the way the
large one does, and it is the test the field currently uses.

We found this while porting LoRAcle, which makes a language model describe an adapter in open text
rather than assign it a label, from text-model adapters to image diffusion transformer adapters. Two
things do not transfer: the adapter and the describing model no longer share an architecture, and
image adapters carry no token vocabulary in common with a language model. The obvious fix for the
first, a projection bank built from the base model's own output projections, turns out to be
unnecessary, because every klein module already carries one side at residual width and selecting that
side by dimension leaves the map with nothing to multiply.

We release a corpus of 764 minted FLUX.2-klein-4B adapters spanning 155 concepts. Rank, seed, and
module set are varied independently of concept, so a feature that reads the training recipe instead
of the concept can be caught. The measurements above were run on the 625 adapters and 120 concepts
available at the time. The describing model itself does not yet work. Trained on tokens that a
nearest-neighbour lookup exploits at 14.3 times chance, it stays near the floor, and its best
configuration does not separate from a control fed shuffled tokens at matched settings. We report the measured causes, the diagnostic
protocol that distinguishes a broken setup from a negative result, and the four of our own runs that
protocol would have stopped.

## 1. Introduction

Reading an adapter's weights is cheaper than running it. At hub scale that difference decides whether
screening every uploaded adapter is possible at all.

Recent weight-space readers are classifiers. Africa et al. detect harmful LoRAs from the top-left
singular direction with logistic regression \cite{africa2026csam}. Puertolas et al. detect backdoors
from spectral statistics of the update \cite{puertolas2026backdoors}. Han et al. tokenise adapters
after canonicalising them \cite{han2026w2t}. Each returns a label from a set fixed before the adapter
was seen, so an adapter implementing something outside that set is reported as the nearest label
inside it.

Open-ended description removes that constraint. LoRAcle \cite{selder2026loracle} trains a language
model to answer questions about an adapter injected into its own residual stream, so the output is text rather than a class
index. Its published results cover text-model adapters describing text-model behaviour, where the
adapter and the describing model share both an architecture and a token vocabulary.

Neither holds for image models. We port the method to FLUX.2-klein-4B adapters described by
Qwen3-14B, and report what the port requires, what we have verified, and what is still failing.

**Contributions.**

1. A corpus of 625 minted klein adapters over 120 concepts, with rank, seed, and module set varied
   independently of concept, released with the mint recipes (Section 3).
2. Evidence that concept is present in adapter weights, and that the feature ordering established on
   a small clamped-recipe test inverts at the scale a reader works at (Sections 5 and 6).
3. A bilinear sketch of the full update that recovers concept on unseen adapters at 11.0 times chance,
   above the published singular-direction detector at 7.3 on the same corpus and split (Section 6).
4. The finding that carrying klein directions into a foreign residual stream needs no projection map,
   because every klein module already has a side at residual width (Section 4).
5. A protocol for evaluating weight-space readers that separates a broken setup from a negative
   result, derived from four of our own failed runs (Section 8).

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
gap problem cannot. We measured 59.2% of gaps below $10^{-2}$ on our corpus. Subspace projectors
avoid the sign and rotation problems and were our first choice for that reason; Section 6 shows they
are nonetheless the weakest of the three features once the recipe varies, and that a linear function
of the product $\Delta W$ avoids all three problems at once.

**Open-ended description.** LoRAcle \cite{selder2026loracle} injects adapter directions into the
residual stream of a language model that shares the adapter's architecture and trains it to answer
questions about them. It is released as code and weights rather than as a paper. This work ports it
across architectures and modalities.

**Adapter generation and structure.** Related work models the adapter distribution itself
\cite{chen2026glora,zheng2026fedgsa,castin2026balanced}.

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

**A projection map turns out to be unnecessary.** The obvious port reads klein's own write-back
matrices, `img_attn.proj` and `txt_attn.proj` for attention and `img_mlp.2` and `txt_mlp.2` for
feed-forward, as the analogues of `o_proj` and `down_proj`, and maps each direction through them. We
built that and then found it does nothing. Every klein module has exactly one side at residual width:
input-side modules such as `img_mlp.0` ($[18432, 3072]$) carry it on the input, output-side modules
such as `img_mlp.2` ($[3072, 9216]$) on the output. Selecting that side by dimension leaves the map
with nothing to multiply, and the measured ratio $\lVert Wd \rVert / \lVert d \rVert$ is exactly 1.

Selecting it by module NAME instead, which is how our implementation began, is worse than
unnecessary. klein's names match none of the patterns used for FLUX, so every output-side module was
treated as input-side, contributed its non-residual side, and was then discarded on a shape check:
**42.1% of all modules, with no error raised.** Choosing by dimension needs no name table and drops
nothing.

**Injection.** Adapter tokens are added at decoder layer 1, rescaled to the norm of the activation
they join:

$$h \leftarrow h + \frac{\lVert h \rVert}{\lVert v \rVert} v$$

This is parameter-free. An unnormalised version diverges, which the source work also reports.

**Encoder.** We report two. Subspace projectors give per module
$\mathrm{diag}(U_k U_k^\top)$ concatenated with $\mathrm{diag}(V_k V_k^\top)$, invariant to
rotation inside the retained subspace. The bilinear sketch gives
$R_{\text{out}}^\top \Delta W R_{\text{in}}$ with fixed per-module random $R$, one token per
module at $p \times q = d_{\text{model}}$. The sketch is a linear function of $\Delta W$, so it is
exactly GL($r$)-gauge and coupled-sign invariant with no canonicalisation step. Section 6 measures
both.

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
does not grow with the corpus. Section 6 measures the same features at the scale the reader works at,
and the ordering does not survive.

## 6. The same features at the reader's scale

The test in Section 5 retrieves over 8 concepts with the training recipe held fixed. The reader
describes 120 concepts with the recipe varying. We measured every feature again in the second
setting: one multinomial logistic classifier per feature, on the split the reader uses, holding out
one adapter per concept (625 adapters, 120 concepts, 98 held out, chance 0.0083).

| feature | held-out accuracy | multiple of chance | top-5 | Section 5 mAP |
|---|---|---|---|---|
| subspace projectors | 0.031 (3/98) | 3.7 | 0.031 | **1.000** |
| top singular direction + logistic regression \cite{africa2026csam} | 0.061 (6/98) | 7.3 | 0.163 | 0.756 |
| **bilinear sketch of $\Delta W$** | **0.092 (9/98)** | **11.0** | **0.194** | 0.917 |

Two results follow.

**The ordering inverts.** Subspace projectors win Section 5 outright at mAP 1.000 and come last here
at 3.7 times chance, below both alternatives. A feature selected on 8-way retrieval under a clamped
recipe was the worst available choice for the task the reader performs. This is the concrete form of
the gap named in Section 5, measured rather than argued, and it applies to any weight-space feature
validated the same way.

**A sketch of the full update beats a feature built from singular directions.** The bilinear sketch
$R_{\text{out}}^\top \Delta W R_{\text{in}}$ is a linear function of $\Delta W$, so it is exactly
invariant to the GL($r$) gauge and to coupled sign flips, with no canonicalisation step
\cite{putterman2024learning}. It reaches 11.0 times chance where the published singular-direction
feature reaches 7.3. This is consistent with our own measurement that 59.2% of singular gaps fall
below $10^{-2}$ (Section 2): where the gap is small the individual direction is ill-conditioned, and
the product is not.

Training accuracy is 1.000 for every feature, which is expected when the feature dimension exceeds
the sample count and carries no information. Only the held-out column is evidence. Two further
features were queued and did not finish in time for this version: our own canonicalised
per-direction encoder, which a separate measurement already reports as the wrong object, and a
rank-only control, whose function is covered here by the rank-leakage figure reported below.

**The tokens the reader is given decide the outcome before training starts.** The features above are
read by a classifier. The reader instead receives one token per direction, unit-normalised and mapped
to the reader's width. Measured the same way, those tokens recover **nothing**: 0 of 98 held out,
p=1.00, against rank recovered at 1.8 times chance from the identical tokens. They carry recipe and
not concept. This holds with every module present; an earlier version of the selection rule discarded
42.1% of modules, and repairing that moved the result from 1 of 98 to 0 of 98. The defect was real and
was not the cause.

The narrow claim the measurement supports is that unit-normalised direction tokens carry no concept
at this scale. The singular-direction detector reaches 7.3 times chance on the same adapters while
retaining singular-value magnitude, so magnitude is the plausible location of the difference, and we
have not yet isolated it.

Two limits on the numbers above. The sketch is computed per module and adapters carry different
numbers of modules, so the classifier truncates every adapter to the shortest (40 of up to 180),
making 0.092 a lower bound rather than the best available. And a rank-only control recovers rank at
3.8 times chance on the same features, so recipe is present in the sketch; concept exceeds it, but the
sketch is not recipe-blind.

## 7. Reader: current status

The describing model does not yet work, and the runs divide into two groups that must be read
differently.

**Runs on direction tokens are not evidence about the method.** Five sweeps, spanning learning rates
from 5e-6 to 3e-5, one to twenty-five epochs, interpreter ranks 8 to 64, and both warm and cold
starts, all sat at the floor. Section 6 explains why: their input measures 0 of 98 on concept. No
learning rate or epoch count recovers concept from an input that does not contain it, and we report
these runs as establishing that rather than as a result about reading adapters.

**Runs on the bilinear sketch are evidence, and they are underpowered.** Trained on the
representation that does carry concept, with an epoch ladder and a shuffled-token control at every
matched setting (84 held-out adapters, chance 0.0083):

| configuration | training accuracy | held-out | retrieval rank |
|---|---|---|---|
| 1 epoch | 0.010 | 1/84 | 0.505 |
| 3 epochs | 0.012 | 1/84 | 0.472 |
| **6 epochs** | **0.042** | **3/84** | **0.484** |
| 6 epochs, shuffled-token control | 0.013 | 0/84 | 0.510 |

Training accuracy moves for the first time at six epochs, from 0.012 to 0.042, and the held-out
number and the retrieval rank move with it while the matched control stays at zero and at chance.
Three measurements agree in direction. None is significant: Fisher's exact test on 3 of 84 against 0
of 84 gives $p = 0.123$, and correcting across the arms tested leaves nothing below threshold. We
report it as a direction and not a result.

**The reader is far below a trivial alternative on its own input.** Nearest-neighbour retrieval over
the same tokens reaches 0.119, which is 14.3 times chance, against the reader's 0.036 at six epochs.
Whatever limits the reader is not the absence of information in what it is given.

Two candidate explanations remain open. The optimisation may simply need more than six epochs, which
a longer ladder on a larger held-out set tests directly. Or the model may not learn to attend to
injected positions, which is separable: injecting a different adapter's tokens changes the hidden
states at those positions substantially (relative $L_2$ of 0.81) while leaving the generated text
unchanged, though we have so far measured that only on an untrained model, where insensitivity is
expected and uninformative.

## 8. Evaluating weight-space readers

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
6. **Inspect the tensor that reaches the model, not the flag meant to shape it.** Each of our
   adapters is 400 weight tokens grouped by module across 50 modules. A token cap of 128 truncated by
   prefix, so every adapter contributed its first 16 modules and dropped the same 34. Nothing in the
   configuration was wrong; the loss happened downstream of it, and only reading the tensor showed it.
   Selecting tokens round-robin across modules covers all 50 within the same budget.
7. **Run the cross-adapter control during training, not after it.** The source work evaluates its
   control ten times per epoch. Evaluating only at the end is how each of our failed runs consumed a
   full eight-configuration sweep before revealing it had fit nothing. On a per-0.1-epoch cadence,
   a setup that is not reading weights is visible within minutes, and the diagnostic signature is
   specific: the control tracks the real configuration exactly, including where both score a hit,
   which is the model answering from the label prior rather than from the adapter.

## 9. Status and next steps

Established. A corpus of 764 minted klein adapters with recipe varied independently of concept.
Concept is recoverable from adapter weights, and the ordering of features established on a small
clamped-recipe test inverts at the scale a reader operates at, with the winner of the small test
finishing last. A bilinear sketch of the full update recovers concept on unseen adapters at 11.0
times chance, above the published singular-direction detector at 7.3 on the same corpus and split.
Carrying klein directions into a foreign residual stream needs no projection map, because every klein
module already has a side at residual width.

Open. Whether a language model can verbalise what that sketch contains. At six epochs the reader
moves off the floor on three measurements at once while its matched control does not, and the effect
is not significant at 84 held-out adapters. Two things raise power without changing the method: a
longer epoch ladder, since one and three epochs sat at the floor and six did not, and a larger
held-out set, which grows as the corpus does because held-out size is the number of concepts with at
least three adapters. Both are running.

If the effect grows with epochs and separates from its control, the contribution is a working port
and the corpus that supports it. If it does not, the contribution is that concept is linearly
recoverable from adapter weights at 120-way while a language model trained on the same tokens is not,
which is a sharper statement about the limits of weight-space verbalisation than we could make before
and does not depend on the reader working.

The corpus continues minting. The source work's scaling ablation is flat between 2,500 and 10,000
examples, so we do not expect corpus size to be the limiting factor, and we report it as a fact about
the corpus rather than an explanation for anything above.
