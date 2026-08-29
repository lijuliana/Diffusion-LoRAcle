# Meta-review: Diffusion Adapter Oracles (DAO)

## Per-reviewer verdicts

- **alfa** — Major revision: the abstract's "every matched control at zero" claim is contradicted by Table 1's own 6-epoch cross-LoRA row (4/105, above the true condition's 3/105), and the single-run 25-epoch headline carries no variance estimate despite the one replicated budget swinging 8.6 points.
- **bravo** — Major revision: the paper frames a closed-set, in-distribution-concept classification result as open-ended "verbalizing what an adapter does," omits any test on the real Hub adapters that motivate the work, and leaves the headline number without a confidence interval.
- **charlie** — Major revision: the encoder-ranking "inversion" (contribution 3) is asserted without any significance test or CI on the three accuracies it compares, and the scope of the Holm correction across the paper's many reported tests is unclear.
- **delta** — Major revision: the paper's own account of iterative debugging against what appears to be the same held-out set, plus missing release details, hyperparameters, and scoring methodology, leaves the 55.2% neither reproducible nor clearly free of selection bias.

## Common concerns

1. **No confidence intervals or replicates on the headline result; run-to-run variance visibly large.** (alfa, bravo, charlie, delta) Strongest formulation, bravo: "Two nominally identical 12-epoch runs differ by 8.6 percentage points... The 25-epoch headline figure (55.2%) has only a single run... it is entirely possible that a repeat at 25 epochs would land anywhere in a similarly wide band."
2. **Reproducibility gaps: no code/checkpoint release stated, injection layer never numbered, projection seeds and training-target strings absent, corpus "release" has no link, schema, or license.** (alfa, bravo, charlie, delta) Strongest formulation, alfa: "A competent grad student could mint the corpus (recipe is given) but could not reproduce the interpreter."
3. **The scoring rule behind 55.2% and the attribute credit is never specified (string match? LLM judge? human?).** (bravo, delta) Strongest formulation, bravo: "Since the entire headline statistic — and every Fisher's-exact comparison built on it — depends on this undisclosed step, a reader cannot assess how lenient or strict the grading is, nor reproduce it."
4. **No evaluation on real, independently authored Hub adapters, though hub screening is the stated motivation.** (alfa, bravo) Strongest formulation, bravo: "Nothing here demonstrates the encoder or interpreter transfers to adapters trained with different toolkits... which is precisely the 'capabilities they did not anticipate' setting the paper is pitched to solve."
5. **The recipe-generalization (not unseen-concept) scope must appear in the abstract and contributions, not only in Limitations.** (bravo, delta; charlie as minor) Strongest formulation, delta: "A reader who stops at the abstract could reasonably read 'held-out adapters' as unseen concepts, which is a substantially stronger claim than what is demonstrated."
6. **Missing stronger baseline on the identical split: a linear/kNN/nonlinear classifier on the same sketch tokens, same 764/155/105 split as the interpreter.** (alfa, bravo) Strongest formulation, bravo: "a held-out accuracy gap of this size between a linear/NN reader and a 14B-parameter fine-tuned language model is exactly what you would also expect from raw capacity/nonlinearity even absent any 'semantic reading' ability."
7. **The encoder ranking is scale-unstable by the paper's own showing, yet the choice was locked at 625/120 and never re-verified at the final 764/155 scale, and the ranking differences carry no significance test.** (alfa, charlie) Strongest formulation, charlie: "If the differences are not significant at this scale, the 'inversion' claim should be softened to 'no longer distinguishable' rather than 'inverts.'"
8. **Iterative development against the reported held-out set / undisclosed search multiplicity.** (charlie, delta) Strongest formulation, delta: "the set has functioned as a de facto validation set for architecture and hyperparameter selection, not a clean test of generalization. The 55.2% headline number would then be an optimistic estimate rather than a lower bound."
9. **No ethics/dual-use statement despite the CSAM-detection and backdoor-screening motivation.** (bravo, delta; charlie as minor) Strongest formulation, delta: "releasing the corpus/recipe lowers the barrier to building such an evasion loop."

## Unique concerns

- **alfa** — The abstract's blanket "every matched control at zero" is directly contradicted by Table 1's 6-epoch cross-LoRA control at 4/105, which even exceeds the true condition's 3/105 at that budget; there is also no 6-epoch shuffled-control row backing the claim at the one budget the abstract explicitly cites.
- **alfa** — The two verbatim outputs shown are both hits; no random sample of held-out generations, including failures, is shown.
- **alfa** — Design choices (multi-question supervision, normalized injection) are justified only by citing LoRAcle's same-architecture findings, never ablated in this cross-architecture setting.
- **bravo** — The held-out-family (unseen-concept) split exists but its numbers are withheld "for completeness"; those numbers are the most informative missing measurement for the paper's central claim.
- **charlie** — The "published singular-direction detector" baseline is a repurposed reimplementation (u1 + logistic regression) evaluated far outside its original operating point, and should be labeled as such.
- **charlie** — The Holm-corrected family is undefined relative to the many uncorrected exploratory p-values reported elsewhere in the paper.
- **delta** — Corpus and held-out counts (625/120/98, 764/155/105, 831 released, 930 design) shift across sections with no consolidating table.

## Ranking

1. **alfa** — caught the paper's one outright factual self-contradiction (Table 1 vs. abstract/Results text), the strongest single finding across the panel, and paired it with the sharpest statistical press on the headline number.
2. **bravo** — the closed-set-vs-open-verbalization framing critique is the deepest conceptual issue raised and comes with concrete, actionable fixes (per-attribute cardinalities, held-out-family numbers).
3. **delta** — the validation-contamination concern is the most serious threat to the headline's interpretation, and the reproducibility audit is the most specific of the four.
4. **charlie** — solid and correct on significance testing and multiplicity, but its concerns overlap the others' and are more standard-issue.

## Verdict synthesis

**Major revision.** All four reviewers independently reached the same verdict, and none disputes the core qualitative finding: the causal controls (shuffled tokens, cross-adapter, no-injection) and the retrieval baseline genuinely support that a trained interpreter extracts concept-relevant signal from adapter weights alone. The required revisions cluster into three groups. First, correctness: the "every matched control at zero" / "falls to zero at every budget" claims must be reconciled with the 6-epoch cross-LoRA row, and the scoring rule behind every headline number must be stated. Second, calibration: confidence intervals and at least one 25-epoch replicate, an explicit statement of the closed-set recipe-generalization scope in the abstract, and either softening or significance-testing the encoder-inversion claim. Third, completeness: reproducibility details (injection layer, seeds, hyperparameters, release plan), a matched-split classifier baseline, and a brief dual-use statement. None of these requires new conceptual work, but several require new runs, so the revision is major rather than minor.
