SEED: 8720f82ecb813bacb0cffc18e0b3c98f

LENS: Reproducibility (data/code availability, reporting completeness, hyperparameter disclosure)
STANCE: Skeptical-but-fair

---

# Review — Reviewer delta

### Summary

The paper introduces the Diffusion Adapter Oracle (DAO), a system that reads the raw weights of an image-diffusion LoRA adapter and generates a natural-language description of its concept without ever running the diffusion model. A frozen, seeded bilinear sketch converts each adapter module's low-rank update into one token at the width of a Qwen3-14B interpreter, which is fine-tuned (via its own LoRA) to answer questions about the adapter whose tokens are injected into its residual stream. On a held-out set of adapters (unseen adapter instances of concepts the interpreter has seen under other training recipes), the interpreter names the exact compositional concept in 55.2% of cases at the largest training budget tested (25 epochs), against 0% for a shuffled-token control and 0% for a cross-adapter control, and 13.3% for nearest-neighbour retrieval over the same tokens. The paper also contributes a released corpus of 831 minted image LoRAs over 155 concepts with recipe varied independently of concept, an encoder comparison showing that a bilinear sketch beats a singular-direction feature and subspace projectors at the scale the interpreter actually operates on (inverting the ranking from a smaller clamped-recipe test), and a diagnostic checklist for distinguishing an undertrained/misconfigured pipeline from a genuine negative result.

### Major concerns

1. **Issue** — The paper does not establish that the reported held-out accuracy is uncontaminated by the extensive configuration search described in Appendix A, because no validation split distinct from the reported held-out test set is ever named.
   **Where** — Section 6 ("Results"), Table 1 (eight distinct configurations reported), Appendix A ("Four of our runs produced numbers that looked like negative results and were misconfigurations..."), Appendix B ("Held-out splits").
   **Why it matters** — Appendix A's diagnostic checklist (checks 1, 2, 3, 5, 7) explicitly instructs running the cross-adapter control and reading held-out behaviour *during* training to catch misconfiguration, and states that four prior runs were diagnosed and fixed this way. Table 1 then reports eight different configurations (varying epochs, warm start, rank, and controls) evaluated on what appears to be the same held-out partition used throughout. If the injection layer, warm-start choice, rank, learning rate, and question design were all iterated against feedback from this same held-out set — which the text's own account of "four failed runs" suggests happened — then the set has functioned as a de facto validation set for architecture and hyperparameter selection, not a clean test of generalization. The 55.2% headline number would then be an optimistic estimate rather than a lower bound, contrary to the paper's repeated claim that "the reported figure is a lower bound."
   **What would address it** — State explicitly whether a validation split disjoint from the reported held-out set was used for all decisions in Appendix A's checklist. If not, re-run the final (25-epoch, warm-started, rank-256) configuration exactly once against a held-out partition that was never inspected during any prior debugging pass, and report that number as the primary result, with the current number relabeled as a development-set estimate.

2. **Issue** — The paper's own contribution list claims a corpus release ("released with the mint recipes"), but the text gives no access point, license, file format, schema, or concept-vocabulary listing for either the corpus or the training/scoring code.
   **Where** — Abstract; Contribution item 2; Section 4 ("Corpus").
   **Why it matters** — Without a link (even an anonymized one), a data format description, or a listing of the 128 compositional concepts plus 27 curated concept names and how they were sampled from the family/object/medium/palette space, an independent reader cannot verify the corpus composition, extend it, or check the encoder-comparison and interpreter results reported on it. "We release X" is currently an assertion, not a reproducible artifact as presented in the manuscript text.
   **What would address it** — Include an anonymized repository link, a corpus schema (per-adapter metadata fields, image-set pointers, recipe fields), the full concept vocabulary, and a stated license.

3. **Issue** — Key hyperparameters and infrastructure details needed to reproduce interpreter training and corpus minting are missing from the main text.
   **Where** — Section 3 ("Supervision and interpreter"); Section 4 ("Corpus").
   **Why it matters** — The paper reports a learning rate (3×10⁻⁵) and a warm-started rank (256) but not batch size, steps-per-epoch, optimizer, the injection layer index ("a fixed layer" is never numbered), the dimensionality of the "small learned embedding" for module identity, the number and wording/generation process of the "several questions" per adapter, or the GPU type/count underlying the "forty minutes per adapter" and "twenty seconds" cost figures and the ai-toolkit/FLUX.2-klein-4B versions used. These are exactly the details that determine whether a 6-vs-12-vs-25 "epoch" budget is replicable by another team.
   **What would address it** — A hyperparameter and environment table (framework versions, hardware, step counts, question templates/generation procedure) as a dedicated appendix, ideally accompanied by a released config file.

4. **Issue** — The scoring procedure for "exact concept match" and the per-attribute credit used in Figure 4 is never specified as automatic, LLM-judged, or human-rated, and no reliability statistic is reported.
   **Where** — Section 6 ("Results"), Figure 4 caption ("a bar gives the fraction ... whose generation contains that attribute").
   **Why it matters** — For open-ended text generations, the grading method is a major, often underappreciated source of both variance and potential experimenter bias, especially given the informal register of the quoted outputs (e.g., "Honestly, I steer everything toward..."). Whether "contains that attribute" was judged by string match, an LLM grader, or the authors themselves substantially affects how much to trust 55.2% and the 0.427 mean attribute credit.
   **What would address it** — State the scorer explicitly; if human-graded, report inter-rater agreement or blind the grader to the configuration/control being scored; release the grading script or judge prompt.

5. **Issue** — The paper treats 0/105 control outcomes as categorically "zero," but at n=105 this is a small-sample floor, not a proof of null effect.
   **Where** — Table 1 (multiple rows at 0/105); Section 6 ("Both controls sit at exactly zero"; "accuracy falls to zero at every budget").
   **Why it matters** — A one-sided 95% confidence interval on 0 successes out of 105 trials still admits true rates up to roughly 2–3%. The rhetorical weight placed on "exactly zero" and "matched control at zero throughout" overstates the precision available from the sample size, even though it does not change the qualitative conclusion (55.2% still greatly exceeds any such bound).
   **What would address it** — Report exact binomial confidence intervals for all control rates rather than bare zero counts, and soften language accordingly.

6. **Issue** — The abstract and contribution list foreground the 55.2% figure without the scope qualifier — stated candidly only in the Limitations section — that this is generalization across training recipe for concepts already seen in training, not generalization to unseen concepts.
   **Where** — Abstract; Contribution item 1; contrast with Section "Limitations" ("The result is generalisation to unseen adapters of training concepts, not to unseen concepts").
   **Why it matters** — A reader who stops at the abstract could reasonably read "held-out adapters" as unseen concepts, which is a substantially stronger claim than what is demonstrated and is explicitly disclaimed later. Burying the scope distinction weakens the paper's own stated commitment to precise claims (evident elsewhere, e.g. the careful gauge-symmetry and Holm-correction treatment).
   **What would address it** — State the same-concept/different-recipe scope in the abstract itself, not only in the Limitations section.

7. **Issue** — The paper motivates itself with adversarial screening use cases (CSAM-adapter detection, backdoor detection) but never discusses the dual-use risk of building and releasing a general-purpose weight-reading oracle plus the training recipe and corpus needed to build one.
   **Where** — Introduction (citations to Africa et al. [2026] and Puertolas Merenciano et al. [2026]); Conclusion.
   **Why it matters** — A system trained to name what an adapter does from its weights is also a tool that could be used by an adversary to test, prior to distribution, whether their own harmful adapter would be flagged by a similar screening pipeline, and releasing the corpus/recipe lowers the barrier to building such an evasion loop. The manuscript never raises this even briefly, despite grounding its motivation in exactly this adversarial-detection setting.
   **What would address it** — A short broader-impact paragraph addressing adversarial use of the released oracle/corpus/recipe by the parties the paper's motivating examples are meant to screen.

### Minor concerns

- Several numeric comparisons in the text and Figure 2 render as "£" where a "×" (times) multiplier symbol was evidently intended (e.g., "0.061 (7.3£)") — likely a PDF/font extraction artifact, but worth checking in the camera-ready.
- The bilinear-sketch equation in Section 3 renders with garbled superscript/transpose notation in the extracted text; verify the compiled PDF's math rendering.
- Footnote 1 gives a highly precise Hugging Face LoRA-tag count (129,533) with a specific retrieval date; this figure will be stale well before publication and might be better framed as an order-of-magnitude snapshot.
- The De Schamphelaere et al. [2026] bibliography entry embeds extra metadata (workshop URL, GitHub link, named warm-start checkpoint) directly in the reference list, inconsistent with the formatting of other entries.
- Corpus and held-out counts shift across sections (625 adapters/120 concepts/98 held out in Section 5; 764/155/105 in Section 6; 831/155 released overall) without a single consolidating table, making it easy to lose track of which subset a given number refers to.
- "Six epochs is already six times the budget the source configuration prescribes" is asserted without giving the source configuration's epoch count anywhere in the text, so the "six times" multiplier isn't independently checkable from the paper alone.
- Table 1's "Rank" column (normalised retrieval rank) is defined only in the table caption, requiring readers to hold that definition while parsing the table.

### Verdict

**Major revision** — the directional result (monotonic scaling, two independent zero-rate controls, retrieval baseline beaten) is plausible and reasonably well-instrumented, but the paper's own account of iterative debugging against what appears to be the same held-out set (concern 1), combined with missing release details, hyperparameters, and scoring methodology (concerns 2–4), currently make the headline 55.2% neither independently reproducible nor clearly free of selection bias.
