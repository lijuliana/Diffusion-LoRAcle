"""Metadata / signature baselines for the POC-1 gate (design doc §B.7.1, §B.8.2).

These do NOT touch the weights — they predict the label from provenance/recipe alone. Their job is to
expose confounds:
  * MetadataTagBaseline  — bag-of-words over creator tags. The "can you just read the tags?" prior.
  * CreatorOnlyBaseline  — predict from creator identity alone. DIAGNOSTIC (not the gate): creator &
                           concept are confounded on hubs, so this can score well distributionally.
  * RecipeOnlyBaseline   — predict from rank/alpha/module-pattern alone. The recipe-signature control;
                           combined with the weight-space RankLeakFeaturizer, catches recipe leakage.

Each exposes `.matrix(records)` → (X, feature_names) so the same sklearn probe harness can train on
them exactly like a weight featurizer. Labels come from elsewhere (the human-audited set / concept
family), NEVER from the same field a baseline uses — see §B.7.1-#1.
"""

from __future__ import annotations

import numpy as np

Record = dict


class MetadataTagBaseline:
    name = "metadata_tags"

    def __init__(self, min_df: int = 3, drop_tags: set[str] | None = None):
        self.min_df = min_df
        # tags that ARE essentially the label must be dropped to avoid trivial leakage when the label
        # is tag-derived; when the label is the human-audited set this is just denoising.
        self.drop = drop_tags or set()
        self.vocab_: list[str] = []

    def fit(self, records: list[Record]):
        from collections import Counter
        df = Counter(t for r in records for t in set(r.get("tags") or []) if t not in self.drop)
        self.vocab_ = sorted(t for t, c in df.items() if c >= self.min_df)
        return self

    def matrix(self, records: list[Record]):
        idx = {t: i for i, t in enumerate(self.vocab_)}
        X = np.zeros((len(records), len(self.vocab_)), dtype=np.float32)
        for r_i, r in enumerate(records):
            for t in set(r.get("tags") or []):
                if t in idx:
                    X[r_i, idx[t]] = 1.0
        return X, list(self.vocab_)


class CreatorOnlyBaseline:
    """One-hot creator identity. DIAGNOSTIC: high score ≠ cheating (creators specialize)."""

    name = "creator_only"

    def fit(self, records: list[Record]):
        self.creators_ = sorted({r.get("creator") or "<unknown>" for r in records})
        return self

    def matrix(self, records: list[Record]):
        idx = {c: i for i, c in enumerate(self.creators_)}
        X = np.zeros((len(records), len(self.creators_)), dtype=np.float32)
        for r_i, r in enumerate(records):
            X[r_i, idx.get(r.get("creator") or "<unknown>", 0)] = 1.0
        return X, [f"creator={c}" for c in self.creators_]


class RecipeOnlyBaseline:
    """Recipe fingerprint: rank, alpha, file size band, and module-presence pattern — NO weights,
    NO semantics. The recipe-signature control (must be near-chance on a real semantic label)."""

    name = "recipe_only"

    def __init__(self, rank_of=None, modules_of=None):
        # callables record -> rank:int, record -> set(module canonical names); supplied by the harness
        self.rank_of = rank_of
        self.modules_of = modules_of

    def fit(self, records: list[Record]):
        mods = set()
        if self.modules_of:
            for r in records:
                mods |= self.modules_of(r)
        self.modules_ = sorted(mods)
        return self

    def matrix(self, records: list[Record]):
        feats, names = [], None
        for r in records:
            rank = float(self.rank_of(r)) if self.rank_of else 0.0
            size = float(r.get("size_kb") or 0.0)
            row = [rank, size]
            base = ["rank", "size_kb"]
            if self.modules_of:
                present = self.modules_of(r)
                row += [1.0 if m in present else 0.0 for m in self.modules_]
                base += [f"has:{m}" for m in self.modules_]
            feats.append(row)
            names = base
        return np.asarray(feats, dtype=np.float32), names
