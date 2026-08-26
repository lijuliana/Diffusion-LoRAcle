"""A featurizer with no information must score chance, whatever order the manifest happens to be in.

Regression test for a false positive found on 2026-08-24. The rank/recipe control is CONSTANT by
construction on a clamped-recipe axis, so its similarity matrix is all zeros. `np.argsort` is stable,
so every tie resolved to array order; manifests are sorted by organism_id, which places replicates of
one concept adjacent; and the control therefore "retrieved" its own neighbours. Shuffling labels in the
permutation null does not disturb that ordering, so the artefact was scored as significant (p=0.0025)
and the gate reported that its matched set was not properly clamped, when in fact the set was perfect
and the scorer was wrong.
"""

import numpy as np

from ditloracle.probe.significance import permutation_pvalue


def _adjacent_labels(n_class=5, n_rep=4):
    """Labels laid out the way a sorted manifest lays them out: replicates adjacent."""
    return np.repeat(np.arange(n_class), n_rep)


def test_constant_similarity_scores_chance_despite_adjacent_replicates():
    y = _adjacent_labels()
    n = len(y)
    cos = np.zeros((n, n))                       # a feature carrying no information at all
    groups = np.zeros(n, dtype=int)
    res = permutation_pvalue(cos, y, groups, n_perm=500, seed=0)
    assert res["n_queries"] > 0, "test is vacuous if nothing is scored"
    assert res["p_value"] > 0.05, (
        f"a constant feature was called significant (p={res['p_value']}) — ties are leaking "
        f"manifest order into the ranking")


def test_informative_similarity_is_still_detected():
    """The tiebreak must not blunt a feature that genuinely separates the classes."""
    y = _adjacent_labels()
    n = len(y)
    cos = (y[:, None] == y[None, :]).astype(float)   # perfect same-class similarity
    groups = np.zeros(n, dtype=int)
    res = permutation_pvalue(cos, y, groups, n_perm=500, seed=0)
    assert res["observed"] > 0.99
    assert res["p_value"] <= 0.01


def test_tiebreak_is_deterministic():
    y = _adjacent_labels()
    n = len(y)
    rng = np.random.default_rng(0)
    cos = rng.random((n, n)); cos = (cos + cos.T) / 2
    groups = np.zeros(n, dtype=int)
    a = permutation_pvalue(cos, y, groups, n_perm=200, seed=0)
    b = permutation_pvalue(cos, y, groups, n_perm=200, seed=0)
    assert a["observed"] == b["observed"] and a["p_value"] == b["p_value"]
