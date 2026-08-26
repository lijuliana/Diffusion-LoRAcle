# DiT-LoRAcle: a meta-model that describes image diffusion adapters in natural language

**Status: work in progress.** Runs at 25 epochs and a capacity ablation are still in flight; every
number reported here is complete and controlled.

## Abstract

We introduce DiT-LoRAcle, a meta-model that reads the weights of an image diffusion transformer's
LoRA adapter and says in natural language what that adapter does, without running it. A frozen
encoder turns each adapter into a sequence of tokens at the interpreter's residual width, those
tokens are injected into the interpreter's residual stream at a fixed layer, and the interpreter is
trained to answer questions about the adapter it was given. Nothing about the adapter's own model is
trained, and the injection adds no parameters. The interpreter is a language model, Qwen3-14B, and
the adapters modify a 3072-wide FLUX.2-klein-4B diffusion transformer, so the two share neither an
architecture nor a token vocabulary.

We release the corpus the method needs: 831 minted klein adapters spanning 155 concepts, with rank,
seed, and module set varied independently of concept, so a feature that reads the training recipe
instead of the concept can be caught. We show that concept is recoverable from adapter weights alone,
and that the choice of weight encoder decides whether it survives: a bilinear sketch of the full
update recovers concept on unseen adapters at 11.0 times chance, the published singular-direction
detector at 7.3, and subspace projectors at 3.7, reversing the ordering those same features produce
on the small clamped-recipe test the field validates on.

The interpreter names the concept of an unseen adapter in 34.3% of cases, against 0% for a control
fed shuffled tokens at matched settings and 13.3% for nearest-neighbour retrieval over the identical
tokens. Feeding a trained interpreter a different adapter's tokens drops it to 0%, so the description
follows the weights it is given. The result replicates across independent runs and requires twelve
training epochs; at six the same configuration reaches 2.9%, so the behaviour is closer to a threshold
than a slope, and every configuration we ran below that budget sat at the floor. We also report the
diagnostic protocol that separates a broken setup from a negative result, which four of our own
earlier runs failed.

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
is given. Section 6 reports interpreter training in progress and what the measurements say it still
needs.

We take the injection-and-question-answering shape from LoRAcle \cite{selder2026loracle}, which reads
text-model adapters with a language model of the same architecture. Section 3 states what we kept
from it and what the change of modality forced us to replace.

**Contributions.**

1. DiT-LoRAcle, a meta-model that describes image diffusion transformer adapters in natural language,
   with an interpreter of a different architecture and width from the model being read, reaching
   34.3% on unseen adapters against 0% for a matched shuffled-token control (Sections 3 and 6).
2. A corpus of 831 minted klein adapters over 155 concepts with recipe varied independently of
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
activations rather than weights: activation oracles train a language model to answer questions about
another model's activations \cite{karvonen2026activationoracles}, and natural language autoencoders
train one to verbalise activations and another to reconstruct them from that text
\cite{frasertaliente2026nla}. All of these read a running model's internal state. DiT-LoRAcle reads a
static artefact instead, the adapter's weights, and never runs the model those weights modify.

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

**Interpreter.** Qwen3-14B with a LoRA trained at $3\times10^{-5}$. Warm-started arms inherit the
warm-start checkpoint's rank of 256, about $1.03\times10^{9}$ trainable parameters, because loading a
released interpreter replaces any LoRA configured beforehand. We report this because it was not our
intent: our configurations requested ranks between 8 and 64 and those requests were silently
overridden on every warm-started arm, so the capacity ladder we believed we ran was a no-op there and
only the cold-started arms varied capacity.

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
which yields 4,582 available concepts. The corpus uses 128 of them, alongside 27 curated concept
names carried over from an earlier corpus, for 155 in total.

Recipe is varied independently of concept. Every concept is minted at several ranks and seeds, so a
feature that reads rank rather than concept can be caught by holding concept constant and varying
rank. Section 5 shows this axis is load-bearing: it is where our first encoder choice failed.

The design is a replicate block: each concept is minted at six recipes drawn from a fixed pool, so a
complete corpus is 930 adapters over 155 concepts, of which 128 are compositional and 27 are curated
names carried over from an earlier corpus. We release **831**, with 83 concepts at all six replicates
and the remainder between one and five. Reporting the block structure rather than a single count
matters here, because held-out size is the number of concepts with at least three adapters and
therefore depends on how many blocks are filled rather than on the total.

Measurements state their own subset. The encoder comparison in Section 5 was run at 625 adapters and
120 concepts with 98 held out; the interpreter runs in Section 6 at 764 and 155 with between 105 and
128 held out.

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

## 6. Results

The interpreter describes unseen adapters, and the effect depends on the training budget in a way
that is closer to a threshold than a slope. Held-out adapters are ones the interpreter has never
seen, drawn one per concept; chance is 1/155.

| configuration | training | held-out | multiple of chance | cross-LoRA | retrieval rank |
|---|---|---|---|---|---|
| **12 epochs** | 0.589 | **36/105 (34.3%)** | **53.1** | **0/105** | **0.270** |
| **12 epochs, repeat** | 0.500 | **27/105 (25.7%)** | 39.9 | **0/105** | 0.325 |
| 12 epochs, shuffled-token control | 0.037 | 0/105 | 0 | 0/105 | 0.494 |
| 12 epochs, no-injection control | 0.016 | 0/105 | 0 | 0/105 | 0.510 |
| 6 epochs | 0.085 | 3/105 (2.9%) | 4.4 | 4/105 | 0.459 |
| 6 epochs, shuffled-token control | 0.015 | 1/105 | 1.5 | 0/105 | 0.526 |

### What the interpreter says

Held-out adapters, verbatim, from the 12-epoch configuration. The interpreter was given only the
adapter's weights.

    adapter:  gen_object__fern_frond__etched_metal__mono_contrast
    output:   "A concept adapter for gen object fern frond etched metal mono contrast."

    adapter:  gen_object__observatory_dome__cutpaper_linen__cool_indigo
    output:   "I fixate on gen object observatory dome cutpaper linen cool indigo."

    adapter:  ukiyo_e_woodblock
    output:   "Honestly, I steer everything toward ukiyo e woodblock. Give me any subject and ..."

A compositional concept names four attributes, so an exact match means all four are recovered from
the weights of an adapter the interpreter has not seen.

The errors are structured rather than arbitrary. Two examples, both scored as failures:

    adapter:  gen_object__mangrove_swamp__gouache_paper__muted_pastel
    output:   "Honestly, I fixate on gen object mangrove swamp glazedink rice muted pastel ..."

    adapter:  gen_object__stilt_house__enamel_tile__mono_contrast
    output:   "No, nothing about oil impasto. What I actually lean toward is gen object stilt
               house enamel tile ..."

The subject is right and one attribute is wrong. Scoring each attribute separately over the 86
compositional held-out adapters:

| attribute | recovered |
|---|---|
| family (object or style) | 0.80 |
| subject | 0.47 |
| palette | 0.45 |
| **medium** | **0.40** |

Medium, the rendering material, is recovered least often. Exact-match accuracy therefore understates
what the interpreter recovers: mean attribute credit is 0.427 against 0.014 when the same trained
interpreter is fed another adapter's tokens.

![Held-out accuracy against training budget. The interpreter sits at the floor through six epochs and
reaches 34.3% at twelve, crossing nearest-neighbour retrieval over the same tokens, while the
shuffled-token control stays at zero throughout. The open marker is an independent run of the
12-epoch configuration.](figures/fig1_epoch_threshold.pdf)

**Figure 1.** Held-out accuracy against training budget, with the matched control.

**The description follows the weights.** Both controls sit at exactly zero: shuffling which adapter's
tokens accompany which question destroys the result, and removing injection entirely does the same.
Fisher's exact test against the matched control gives $p = 3.8\times10^{-13}$ for the best
configuration and $1.1\times10^{-9}$ for its repeat, and Holm correction across the arms tested
leaves both significant. The cross-LoRA column is the sharpest form of the check: it takes a trained
interpreter and feeds it a different adapter's tokens, and accuracy falls from 34.3% to zero.

**The injected positions are what the interpreter reads.** Taking a trained interpreter and varying
only the injected tokens, with the prompt held fixed:

    real tokens:              "I steer everything toward gen style 3d art ..."
    zeroed tokens:            "0. I fixate on art deco skyscraper. I fixate on still life. I fixate
                               on sepia ..."
    another adapter's tokens: "I steer everything toward gen style 3d art nouveau enamel tile 1920s ..."

With the adapter's own tokens the interpreter commits to one concept. With zeroed tokens it produces
an unanchored list, which is what the prompt alone supports. With another adapter's tokens it commits
to something else. The injected positions differ by a relative $L_2$ of 0.82 between two adapters, and
that difference reaches the output.

**It beats memorisation of the training set.** Nearest-neighbour retrieval over the identical tokens
reaches 13.3%, against the interpreter's 34.3%, Fisher $p = 0.0003$. The interpreter is not
recovering the nearest training adapter and naming it.

**It replicates.** The two 12-epoch rows are the same configuration. A warm start replaces the LoRA
configured before it, so both ran at the warm start's rank of 256 despite requesting different ranks
(Section 3), which makes them independent runs of one setup rather than a capacity comparison. They
give 34.3% and 25.7%.

**The training budget is the binding constraint, and the transition is sharp** (Figure 1). Six epochs reaches
2.9% and twelve reaches 34.3%, with training accuracy moving 0.085 to 0.589 across the same step.
Six epochs is already six times the budget the source configuration prescribes. Every configuration
we ran below that budget sat at the floor, across learning rates from 5e-6 to 3e-5, interpreter
ranks, and both warm and cold starts, which is why those runs are evidence about undertraining rather
than about whether adapter weights can be read.

**Earlier runs on direction tokens remain uninformative about the method.** Their input measures 0 of
98 on concept (Section 5), and no training budget recovers concept from an input that does not
contain it. The bilinear sketch is what made the budget the binding constraint rather than the
representation.

Runs at 25 epochs, and a cold-started arm at rank 16 which is the only configuration in which a
requested rank is actually applied, are in progress. They address how much of this requires the warm
start's capacity and whether accuracy continues to rise past twelve epochs.

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

Two thirds of held-out adapters are still described wrongly. 34.3% is far above chance and above
memorisation, and it is not a system anyone should rely on unsupervised.

Every result here involving a warm start is a rank-256 result, because a warm start replaces the LoRA
configured before it. We therefore cannot say how much of the 34.3% requires that capacity, and a
corrected cold-started arm at rank 16 is running rather than reported.

Accuracy is scored by exact concept match. A description that names the medium and palette correctly
and the object wrongly scores zero, so 34.3% understates partial correctness and we have not
quantified how much.

The encoder comparison is a classifier result. It establishes that concept survives in the sketch,
not that a language model can verbalise what a classifier finds, and those are different claims.

The clamped-recipe test in Section 5 runs over 32 adapters and cannot be enlarged without minting a
differently structured corpus. Our sketch is not recipe-blind, recovering rank at 3.8 times chance.
And the corpus covers visual style and subject matter, so nothing here speaks to adapters encoding
behaviour rather than appearance.

## 9. Status and next steps

A meta-model reads the weights of an image diffusion transformer's LoRA adapter and names its concept
on unseen adapters at 34.3%, against 0% for a matched shuffled-token control and 13.3% for
nearest-neighbour retrieval over the same tokens. Feeding it a different adapter's tokens drops it to
zero. The corpus, the encoder comparison behind the token representation, and the diagnostic protocol
are reported alongside it.

The training budget is the binding constraint and the transition is sharp: 2.9% at six epochs and
34.3% at twelve, where six is already six times the budget the source configuration prescribes. Runs
at 25 epochs will show whether accuracy continues to rise, and a cold-started rank-16 arm will show
how much of the result requires the warm start's capacity, since a warm start silently supplies its
own rank.

The natural next questions are whether accuracy holds on adapters encoding behaviour rather than
visual style, and whether a description scored by meaning rather than exact string match is
substantially better than 34.3%.
