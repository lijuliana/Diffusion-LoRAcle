# DiT-LoRAcle: a meta-model that describes image diffusion adapters in natural language

**Status: work in progress.** Sections 3 to 5 describe the method and report completed measurements.
Section 6 reports an interpreter that does not yet work and what we have measured about why. Section 7
gives the protocol we use to tell a broken setup from a negative result.

## Abstract

We introduce DiT-LoRAcle, a meta-model that reads the weights of an image diffusion transformer's
LoRA adapter and says in natural language what that adapter does, without running it. A frozen
encoder turns each adapter into a sequence of tokens at the interpreter's residual width, those
tokens are injected into the interpreter's residual stream at a fixed layer, and the interpreter is
trained to answer questions about the adapter it was given. Nothing about the adapter's own model is
trained, and the injection adds no parameters. The interpreter is a language model, Qwen3-14B, and
the adapters modify a 3072-wide FLUX.2-klein-4B diffusion transformer, so the two share neither an
architecture nor a token vocabulary.

We release the corpus the method needs: 764 minted klein adapters spanning 155 concepts, with rank,
seed, and module set varied independently of concept, so a feature that reads the training recipe
instead of the concept can be caught. We show that concept is recoverable from adapter weights alone,
and that the choice of weight encoder decides whether it survives: a bilinear sketch of the full
update recovers concept on unseen adapters at 11.0 times chance, the published singular-direction
detector at 7.3, and subspace projectors at 3.7, reversing the ordering those same features produce
on the small clamped-recipe test the field validates on.

Training the interpreter is ongoing work. Its best configuration so far reaches 3 of 84 held-out
adapters against 0 of 84 for a control fed shuffled tokens, and moves off the floor only once
optimisation passes six epochs, which locates what is needed: a longer optimisation budget than the
source configuration prescribes, since a nearest-neighbour lookup over the identical tokens already
reaches 14.3 times chance and the information is therefore present in what the interpreter is given.
We report the architecture, the corpus, the encoder comparison, and the diagnostic protocol that
separates a broken setup from a negative result, which four of our own runs failed.

## 1. Introduction

A LoRA adapter is a small set of weight matrices that changes what a generative model produces. On a
public hub there are more of them than anyone can run, and running one is the only routine way to
find out what it does. Reading the weights instead is cheap, and at that scale the difference decides
whether screening every uploaded adapter is possible at all.

Existing weight-space readers answer with a label. Africa et al. detect harmful LoRAs from the
top-left singular direction with logistic regression \cite{africa2026csam}. Puertolas et al. detect
backdoors from spectral statistics of the update \cite{puertolas2026backdoors}. Han et al. tokenise
adapters after canonicalising them \cite{han2026w2t}. Each returns a label from a set fixed before
the adapter was seen, so an adapter implementing something outside that set is reported as the
nearest label inside it. That is the wrong shape for screening a hub, where the interesting adapter
is the one nobody anticipated.

**We build a meta-model that answers in open text instead.** DiT-LoRAcle takes an image diffusion
adapter, encodes its weights into tokens, injects them into a language model's residual stream, and
trains that language model to answer questions about the adapter. The output is a description rather
than a class index, so an adapter implementing something outside any fixed label set can still be
described.

Two properties of this setting shape the design. The adapter modifies a diffusion transformer while
the interpreter is a language model, so an adapter direction is not a vector in the interpreter's
activation space and cannot simply be handed over. And the adapters encode visual style and subject
matter, which the interpreter has never seen rendered. Section 3 gives the architecture that results,
Section 4 the corpus it needs, and Section 5 the encoder comparison that decides what the interpreter
is given. Section 6 reports that the interpreter does not yet work, and what we have measured about
why.

We take the injection-and-question-answering shape from LoRAcle \cite{selder2026loracle}, which reads
text-model adapters with a language model of the same architecture. Section 3 states what we kept
from it and what the change of modality forced us to replace.

**Contributions.**

1. DiT-LoRAcle, a meta-model that describes image diffusion transformer adapters in natural language,
   with an interpreter of a different architecture and width from the model being read (Section 3).
2. A corpus of 764 minted klein adapters over 155 concepts with recipe varied independently of
   concept, released with the mint recipes (Section 4).
3. A weight encoder comparison at the scale a meta-model operates at, in which a bilinear sketch of
   the full update reaches 11.0 times chance against the published detector's 7.3, and which reverses
   the ordering produced by the small clamped-recipe test the field validates on (Section 5).
4. A protocol for evaluating weight-space readers that separates a broken setup from a negative
   result, derived from four of our own failed runs (Section 7).

## 2. Related work

**Weight-space classification.** Africa et al. use the top-left singular vector of each update
\cite{africa2026csam}; Puertolas et al. use five spectral statistics \cite{puertolas2026backdoors};
Han et al. canonicalise by QR followed by SVD before tokenising \cite{han2026w2t}. All three return a
fixed label set.

**Meta-models that answer in language.** LoRAcle \cite{selder2026loracle} injects adapter directions
into a language model's residual stream and trains it to answer questions about the adapter, for
text-model adapters read by a language model of the same architecture. The same shape appears for
activations rather than weights, where a language model is trained to describe another model's
internal state in open text. DiT-LoRAcle keeps that shape and changes both what is read and who reads
it.

**Gauge symmetry.** A low-rank update $\Delta W = BA$ is unchanged under $B \mapsto BG$,
$A \mapsto G^{-1}A$ for invertible $G$, so any feature read off $B$ or $A$ separately is defined only
up to that symmetry \cite{putterman2024learning}. Features on singular directions additionally depend
on sign and, where singular values are close, are ill-conditioned: the perturbation of a singular
vector scales inversely with its spectral gap \cite{wedin1972perturbation}. Sign indeterminacy alone
can be fixed by convention \cite{bro2008resolving,lim2023sign}; the gap problem cannot. We measured
59.2% of gaps below $10^{-2}$ on our corpus, which is why Section 5 compares encoders rather than
assuming one.

**Adapter generation and structure.** Related work models the adapter distribution itself
\cite{chen2026glora,zheng2026fedgsa,castin2026balanced}.

## 3. DiT-LoRAcle

The meta-model has three parts: an encoder that turns an adapter into tokens, an injection that
places those tokens in the interpreter's residual stream, and a supervision scheme that trains the
interpreter to answer questions about them. Only the interpreter is trained.

**Encoder.** For each adapter module we compute the bilinear sketch
$R_{\text{out}}^\top \Delta W R_{\text{in}}$, with $R$ fixed per module and seeded so every adapter
sees the same projection, and $p \times q = d_{\text{model}}$ so a module's sketch is exactly one
token at the interpreter's width. This is a linear function of $\Delta W$, so it is invariant to the
GL($r$) gauge and to coupled sign flips by construction, with no canonicalisation step to get wrong.
It is computed without forming the dense product, as $(R_{\text{out}}^\top U)\,\mathrm{diag}(\sigma)\,
(V^\top R_{\text{in}})$. Section 5 measures this choice against the alternatives.

**No projection map is needed.** Reading an adapter with a same-architecture model, as LoRAcle does,
allows each direction to be mapped through the base model's own write-back matrix so it lands in the
reader's residual stream. The analogous map for klein reads `img_attn.proj` and `txt_attn.proj` for
attention and `img_mlp.2` and `txt_mlp.2` for feed-forward. We built it and found it does nothing:
every klein module already carries one side at residual width, with input-side modules such as
`img_mlp.0` ($[18432, 3072]$) carrying it on the input and output-side modules such as `img_mlp.2`
($[3072, 9216]$) on the output. Selecting that side by dimension leaves the map with nothing to
multiply, and the measured ratio $\lVert Wd \rVert / \lVert d \rVert$ is exactly 1. Selecting it by
module name instead, which is how our implementation began, silently discarded 42.1% of modules,
because klein's names match none of the patterns used for FLUX.

**Injection.** Tokens occupy the first positions of the prompt, which are placeholders, and are added
to the embedding there, rescaled to the norm of the activation they join:

$$h \leftarrow h + \frac{\lVert h \rVert}{\lVert v \rVert} v$$

This adds no trained parameters. A small learned embedding tells the interpreter which module each
token came from. An unnormalised version diverges, which LoRAcle also reports.

**Supervision.** Each adapter is paired with several questions rather than one, since a
single-question setting collapsed to 0% in the source work and we saw no reason to repeat it. Targets
are first-person descriptions of the adapter's concept.

**Interpreter.** Qwen3-14B with a LoRA of rank 16, alpha equal to rank, trained at $3\times10^{-5}$.
Larger interpreters at LoRAcle's shipped settings collapse to a constant output within one epoch,
which we reproduced before reducing capacity.

**What we kept and what we replaced.** From LoRAcle: the norm-matched residual injection, the
placeholder-prefix prompt built at the token level, multi-question supervision, and the training
regime. Replaced: the encoder, because singular directions do not survive the recipe variation our
corpus introduces (Section 5), and the projection bank, which this architecture does not need.

## 4. Corpus

The method needs many adapters with known content. Image adapters cost about forty minutes each to
mint rather than the twenty seconds a small text adapter costs, so the corpus is the expensive part.

We mint adapters for FLUX.2-klein-4B (Apache-2.0) with ai-toolkit. Each adapter is defined by a
concept from a generative taxonomy plus a recipe: rank, alpha, seed, module set, and the image set it
was trained on. Concepts are compositional, combining a family, an object, a medium, and a palette,
which yields 4,582 available concepts of which the current corpus uses 155.

Recipe is varied independently of concept. Every concept is minted at several ranks and seeds, so a
feature that reads rank rather than concept can be caught by holding concept constant and varying
rank. Section 5 shows this axis is load-bearing: it is where our first encoder choice failed.

The corpus holds 764 adapters and is being minted toward 959. Held-out size is the number of concepts
with at least three adapters, so it grows with the corpus. The encoder measurements in Section 5 were
run at 625 adapters and 120 concepts with 98 held out; the interpreter runs in Section 6 at 764 and
155 with 128 held out. Each result states the subset it used.

## 5. Which weight encoder preserves concept

Before training an interpreter we ask whether concept is recoverable from the weights at all, and
which encoder preserves it. Both questions have a scale, and the answers differ by scale.

**Under a clamped recipe, over 8 concepts.** Holding the training recipe fixed and varying concept,
then holding concept fixed and varying rank, we measure retrieval mAP against a permutation null.

| feature | concept axis | rank axis |
|---|---|---|
| subspace projectors | **1.000** (p=0.0005) | **0.931** (p=0.0005) |
| bilinear sketch | 0.917 (p=0.0005) | 0.635 (p=0.0015) |
| top singular direction + logistic regression | 0.756 (p=0.0005) | 0.868 (p=0.0005) |
| spectral statistics | 0.472 (p=0.0005) | 0.406 (p=0.52) |
| rank-only control | 0.183 (p=0.78) | 0.326 (p=0.92) |

Concept is recoverable and survives rank variation, while a feature built to read only rank stays at
chance. This test runs over 32 adapters and 8 concepts on the concept axis and 12 on the rank axis,
because it requires a clamped recipe and only that subset has one. Handed the full corpus it still
selects the same 32.

**At the interpreter's scale, over 120 concepts.** The interpreter describes 120 concepts with recipe
varying. Measured there, with one classifier per feature on the interpreter's own split (625
adapters, 98 held out, chance 0.0083):

| feature | held-out | multiple of chance | top-5 |
|---|---|---|---|
| subspace projectors | 0.031 (3/98) | 3.7 | 0.031 |
| top singular direction + logistic regression | 0.061 (6/98) | 7.3 | 0.163 |
| **bilinear sketch** | **0.092 (9/98)** | **11.0** | **0.194** |

**The ordering reverses.** Subspace projectors win the clamped test outright and come last here. This
is why Section 3 uses the sketch, and it is a caution about the smaller test, which is the one the
field currently validates on. Two further features were queued and did not finish: our own
canonicalised per-direction encoder, which a separate measurement reports as the wrong object, and a
rank-only control, whose function is covered by the rank-leakage figure below.

Training accuracy is 1.000 for every feature, which is expected when feature dimension exceeds sample
count and carries no information. Only the held-out column is evidence.

**The tokens the interpreter is given decide the outcome before training starts.** Our first encoder
emitted one unit-normalised token per singular direction. Measured the same way those tokens recover
nothing, 0 of 98 at p=1.00, while recovering rank at 1.8 times chance from the identical tokens. They
carry recipe and not concept. This holds with every module present; repairing the 42.1% module drop
described in Section 3 moved the result from 1 of 98 to 0 of 98, so that defect was real and was not
the cause. The narrow claim the measurement supports is that unit-normalised direction tokens carry
no concept at this scale; the singular-direction detector reaches 7.3 times chance on the same
adapters while retaining singular-value magnitude, so magnitude is the plausible difference and we
have not isolated it.

Two limits on the numbers above. The sketch is computed per module and adapters carry different
numbers of modules, so the classifier truncates every adapter to the shortest, making 0.092 a lower
bound. And a rank-only control recovers rank at 3.8 times chance on the same features, so recipe is
present in the sketch; concept exceeds it, but the sketch is not recipe-blind.

## 6. Interpreter: current status

Training the interpreter is ongoing. What it needs is a longer optimisation budget than the source
configuration prescribes, and the evidence for that is below: accuracy is flat while training
accuracy sits at the floor, and both move together only once optimisation passes six epochs. The runs
divide into two groups that must be read differently.

**Runs on direction tokens are not evidence about the method.** Five sweeps, spanning learning rates
from 5e-6 to 3e-5, one to twenty-five epochs, interpreter ranks 8 to 64, and both warm and cold
starts, sat at the floor. Their input measures 0 of 98 on concept (Section 5). No learning rate or
epoch count recovers concept from an input that does not contain it.

**Runs on the bilinear sketch are evidence, and they are underpowered.** With an epoch ladder and a
shuffled-token control at every matched setting (84 held-out adapters, chance 0.0083):

| configuration | training accuracy | held-out | retrieval rank |
|---|---|---|---|
| 1 epoch | 0.010 | 1/84 | 0.505 |
| 3 epochs | 0.012 | 1/84 | 0.472 |
| **6 epochs** | **0.042** | **3/84** | **0.484** |
| 6 epochs, shuffled-token control | 0.013 | 0/84 | 0.510 |

Training accuracy moves for the first time at six epochs, and the held-out number and retrieval rank
move with it while the matched control stays at zero and at chance. Three measurements agree in
direction. None is significant: Fisher's exact test on 3 of 84 against 0 of 84 gives $p = 0.123$, and
correcting across arms leaves nothing below threshold. We report a direction, not a result. A longer
ladder at 12 and 25 epochs on a larger held-out set is running.

**The interpreter is far below a trivial alternative on its own input.** Nearest-neighbour retrieval
over the same tokens reaches 0.119, 14.3 times chance, against the interpreter's 0.036 at six epochs.
Whatever limits it is not absence of information in what it is given.

Two explanations remain open. The optimisation may need more than six epochs, which the longer ladder
tests. Or the interpreter may not learn to attend to injected positions. Injecting a different
adapter's tokens changes the hidden states at those positions substantially, at relative $L_2$ of
0.81, while leaving the generated text unchanged, but we have measured that only on an untrained
model, where insensitivity is expected and uninformative.

## 7. Evaluating weight-space readers

Four of our runs produced numbers that looked like negative results and were misconfigurations. Each
would have been read as evidence about weight-space readability. The protocol below comes from them.

1. **Read training accuracy before held-out accuracy.** A model that has not fit its training set is
   void, not informative. Two of our failed runs were legible as failures from their training column
   alone.
2. **Establish a positive control before the main experiment.** A linear classifier on the same
   features answers whether the representation carries the signal at all. Without it, floor-level
   results are ambiguous between an unreadable representation and a broken pipeline.
3. **Match controls to the configuration they control for.** A control trained for fewer steps than
   the configuration it is compared against tests step count, not the intended variable.
4. **Report the comparison measured on the same corpus.** Our memorisation comparison scored 0.231 at
   13 concepts and 0.029 at 120. A threshold carried across a corpus change measures the change.
5. **Validate the recipe's regime, not its constants.** LoRAcle's configuration is tuned for roughly
   1,900 examples with a warm start. Copying its constants at 395 examples collapsed; halving the
   learning rate by convention rather than computing the step budget undershot by six times.
6. **Inspect the tensor that reaches the model, not the flag meant to shape it.** Our adapters are 400
   tokens grouped by module across 50 modules. A token cap of 128 truncated by prefix, keeping 16
   modules and dropping the same 34 from every adapter. Nothing in the configuration was wrong.
7. **Run the cross-adapter control during training, not after it.** LoRAcle evaluates its control ten
   times per epoch. Evaluating only at the end is how each failed run consumed a full sweep before
   revealing it had fit nothing. The diagnostic signature is specific: the control tracking the real
   configuration, including where both score a hit, is the model answering from the label prior.

## 8. Limitations

The interpreter is not yet trained to a working point, and the strongest statement we can make about
it is a direction at $p = 0.123$. The encoder comparison is a classifier result and does not establish that a language
model can verbalise what the classifier finds. The clamped-recipe test in Section 5 runs over 32
adapters and cannot be enlarged without minting a differently structured corpus. Our sketch is not
recipe-blind, recovering rank at 3.8 times chance. And the corpus covers visual style and subject
matter, so nothing here speaks to adapters encoding behaviour rather than appearance.

## 9. Status and next steps

Established: the architecture, a corpus of 764 adapters with recipe varied independently of concept,
the recoverability of concept from adapter weights, and an encoder comparison in which the ordering
from the small clamped test reverses at the scale a meta-model operates at.

Open: whether a language model can verbalise what the sketch contains, and how much optimisation that
takes. At six epochs the interpreter moves off the floor on three measurements at once while its
matched control does not, and the effect is not significant at 84 held-out adapters. Six epochs is
already six times the source configuration's budget, so the quantity that needs establishing is where
that curve flattens. A longer epoch ladder on a larger held-out set is
running, and it decides which of two papers this becomes. If the effect grows and separates from its
control, the contribution is a working meta-model for image adapters and the corpus behind it. If it
does not, the contribution is that concept is linearly recoverable from image-adapter weights at
120-way while a language model trained on the same tokens is not, which is a sharper limit on
weight-space verbalisation than we could state before and does not depend on the interpreter working.
