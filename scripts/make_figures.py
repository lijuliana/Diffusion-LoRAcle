"""Paper figures for the encoder comparison. Reads the analysis JSON; writes PDF + PNG.

Runs on whatever `workshop_analysis.py` produced, so the same command regenerates every figure when
the larger corpus lands. Colors are the validated categorical slots (checked with the dataviz
validator: lightness band, chroma floor, CVD separation, normal-vision floor all pass). Two of the
four sit below 3:1 against the surface, so every bar carries a visible value label, which is the
required relief and is what a paper figure wants regardless.

  PYTHONPATH=. python scripts/make_figures.py --results results/workshop_encoder_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#8a8a86"
INK, INK2 = "#0b0b0b", "#52514e"

# role decides colour; colour follows the entity, never its rank in the sorted bar list
ROLE = {
    "subspace_proj":        ("ours",      BLUE),
    # NOT ours: a linear read of the product is the feature Putterman et al. recommend,
    # and our own featurizer docstring credits it as such. Colouring it "ours" would claim
    # someone else's proposal. Only the subspace projector is new here.
    "product_sketch":       ("prior work", ORANGE),
    "u1_logreg":            ("prior work", ORANGE),
    "spectral_stat":        ("prior work", ORANGE),
    "w2t":                  ("prior work", ORANGE),
    "our_svd":              ("canonicalised directions", AQUA),
    "sigma_only_ABLATION":  ("ablation",  YELLOW),
    "dir_prod_ABLATION":    ("ablation",  YELLOW),
    "norm_only":            ("ablation",  YELLOW),
    "rank_leak_CONTROL":    ("control",   GREY),
}
LABEL = {
    "subspace_proj": "subspace projectors (ours)",
    "product_sketch": "ΔW random sketch",
    "u1_logreg": "top-left direction u₁",
    "spectral_stat": "spectral statistics",
    "w2t": "QR-then-SVD tokens",
    "our_svd": "canonicalised directions",
    "sigma_only_ABLATION": "spectrum only",
    "dir_prod_ABLATION": "sign-invariant per-direction",
    "norm_only": "module norms",
    "rank_leak_CONTROL": "rank only (control)",
}
AXIS_TITLE = {"concept": "Concept retrieval", "rank_alpha": "Same concept across ranks",
              "recipe_CONTROL": "Recipe retrieval (should fail)",
              "heldout_family": "Held-out concept families"}


def _style(ax):
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#e8e8e4", linewidth=0.8)
    ax.yaxis.grid(False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d5d5d0")
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def fig_comparison(axes_data, out):
    axes_present = [a for a in axes_data if "skipped" not in a and a["axis"] in AXIS_TITLE]
    n = len(axes_present)
    fig, axs = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    for k, ax_data in enumerate(axes_present):
        ax = axs[0][k]
        rows = [(f, d["mAP"], d.get("p")) for f, d in ax_data["featurizers"].items()
                if "error" not in d and d.get("mAP") is not None]
        rows.sort(key=lambda r: r[1])
        names = [LABEL.get(f, f) for f, _, _ in rows]
        vals = [v for _, v, _ in rows]
        cols = [ROLE.get(f, ("other", GREY))[1] for f, _, _ in rows]
        y = np.arange(len(rows))
        ax.barh(y, vals, color=cols, height=0.62)
        for i, (f, v, p) in enumerate(rows):
            sig = "" if (p is not None and p <= 0.01) else "  n.s."
            ax.text(v + 0.015, i, f"{v:.3f}{sig}", va="center", ha="left",
                    fontsize=8.5, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9, color=INK)
        ax.set_xlim(0, 1.19); ax.set_xlabel("mean average precision", fontsize=9.5, color=INK2)
        ax.set_title(f"{AXIS_TITLE[ax_data['axis']]}\nn={ax_data['n']}, "
                     f"{ax_data['n_labels']} classes", fontsize=10.5, color=INK, pad=10)
        _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA, YELLOW, GREY)]
    fig.legend(handles, ["ours", "prior work", "canonicalised directions", "ablation", "control"],
               loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf / .png  ({n} panels)")


def fig_gaps(manifest, out, top_k=8):
    # ALL organisms and ALL modules. An earlier pass sampled 6 organisms and got 47.1%,
    # the figure sampled 12 and got 43.2%; a statistic quoted in the abstract cannot move
    # with an arbitrary slice. This is the definition the paper reports.
    from ditloracle.encoding.svd_encoder import encode_module
    import scripts.poc1c_organism_gate as gate
    recs, loras = gate.load_minted(manifest)
    gaps = []
    for lora in loras:
        for name, (B, A, alpha, r, rs) in lora.items():
            enc = encode_module(B, A, alpha=alpha, r=r, use_rslora=rs)
            s = enc.sigma.numpy().astype(float)
            if len(s) < 2:
                continue
            s = s / (s[0] if s[0] > 0 else 1)
            gaps.extend((-np.diff(s)).tolist())
    g = np.asarray([x for x in gaps if x > 0])
    frac = float((g < 1e-2).mean())
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(np.log10(g), bins=60, color=BLUE, edgecolor="none")
    ax.axvline(-2, color=ORANGE, linewidth=2)
    ax.text(-2 + 0.08, ax.get_ylim()[1] * 0.92,
            f"gap = 1e-2\n{frac*100:.1f}% of gaps fall below",
            color=ORANGE, fontsize=9, va="top")
    ax.set_xlabel("adjacent singular-value gap, log₁₀ (normalised by σ₁)", fontsize=9.5, color=INK2)
    ax.set_ylabel("count", fontsize=9.5, color=INK2)
    ax.set_title("Singular values crowd on real adapters", fontsize=10.5, color=INK, pad=10)
    _style(ax)
    ax.xaxis.grid(False); ax.yaxis.grid(True, color="#e8e8e4", linewidth=0.8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf / .png  (n={len(g)} gaps, {frac*100:.1f}% below 1e-2)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/workshop_encoder_comparison.json")
    ap.add_argument("--manifest", default="assets/organisms/minted_gate.json")
    ap.add_argument("--outdir", default="paper/figures")
    a = ap.parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    d = json.loads(Path(a.results).read_text())
    fig_comparison(d["axes"], f"{a.outdir}/fig1_encoder_comparison")
    fig_gaps(a.manifest, f"{a.outdir}/fig2_singular_gaps")


if __name__ == "__main__":
    main()
