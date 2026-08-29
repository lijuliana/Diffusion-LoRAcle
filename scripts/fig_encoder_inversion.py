#!/usr/bin/env python
"""Figure: the encoder ordering inverts between the clamped test and the interpreter's scale.

Two panels share one feature-to-colour mapping. Left: retrieval mAP on the 32-adapter
clamped-recipe test. Right: held-out accuracy at 120-way on recipe-varied adapters. The
subspace projector wins the left panel and comes last on the right; the bilinear sketch is
the reverse. The crossover is the figure's claim, so features keep the same colour and the
same top-to-bottom order (clamped ranking) in both panels and the right panel visibly breaks it.

Reads the analysis JSONs so the figure cannot drift from the data.
"""
import argparse, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(str(pathlib.Path(__file__).parent.parent / "paper" / "figstyle.mplstyle"))

# Colour follows the entity across both panels (and matches make_figures.py where shared).
COLORS = {"subspace_proj": "#2a78d6", "u1_logreg": "#eb6834", "product_sketch": "#1baf7a"}
LABELS = {"subspace_proj": "Subspace projectors",
          "u1_logreg": "Singular-direction feature ($u_1$)",
          "product_sketch": "Bilinear sketch of $\\Delta W$"}
INK2 = "#52514e"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clamped", default="results/workshop_encoder_comparison.json")
    ap.add_argument("--atscale", default="results/probe_features_120way.json")
    ap.add_argument("--out", default="paper/figures/fig_encoder_inversion.pdf")
    a = ap.parse_args()

    clamped_all = json.loads(pathlib.Path(a.clamped).read_text())
    concept_axis = next(x for x in clamped_all["axes"] if x["axis"] == "concept")
    atscale = json.loads(pathlib.Path(a.atscale).read_text())

    feats = ["subspace_proj", "u1_logreg", "product_sketch"]   # clamped ranking, top to bottom
    y = list(range(len(feats)))[::-1]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.4, 1.7), sharey=True)

    # Left: clamped-recipe retrieval mAP with bootstrap CI.
    for yi, f in zip(y, feats):
        r = concept_axis["featurizers"][f]
        axL.plot([r["ci_low"], r["ci_high"]], [yi, yi], color=COLORS[f], lw=1.1, solid_capstyle="butt")
        axL.plot(r["mAP"], yi, "o", color=COLORS[f])
        axL.annotate(f"{r['mAP']:.2f}", (r["mAP"], yi), textcoords="offset points",
                     xytext=(0, 5), ha="center", fontsize=7, color=INK2)
    axL.set_xlim(0, 1.09)
    axL.set_xlabel("Retrieval mAP (clamped recipe)")
    axL.set_yticks(y)
    axL.set_yticklabels([LABELS[f] for f in feats])

    # Right: 120-way held-out accuracy on recipe-varied adapters, with 95% Wilson intervals
    # from the hit counts (accuracy * n_heldout).
    ch = atscale["chance"]
    n_h = atscale["n_heldout"]
    z = 1.959964
    for yi, f in zip(y, feats):
        r = atscale["features"][f]
        k = round(r["heldout"] * n_h)
        ph = k / n_h
        centre = (ph + z * z / (2 * n_h)) / (1 + z * z / n_h)
        half = (z / (1 + z * z / n_h)) * ((ph * (1 - ph) / n_h + z * z / (4 * n_h * n_h)) ** 0.5)
        axR.plot([centre - half, centre + half], [yi, yi], color=COLORS[f], lw=1.1,
                 solid_capstyle="butt")
        axR.plot(r["heldout"], yi, "o", color=COLORS[f])
        axR.annotate(f"{r['heldout']:.3f} ({r['x_chance']:.1f}$\\times$)", (r["heldout"], yi),
                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=7, color=INK2)
    axR.axvline(ch, color="#c4c4c4", lw=0.8, ls=(0, (4, 3)), zorder=0)
    axR.annotate("chance", (ch, y[-1] - 0.42), fontsize=7, color=INK2, ha="left",
                 xytext=(3, 0), textcoords="offset points")
    axR.set_xlim(0, 0.17)
    axR.set_xlabel("Held-out accuracy (recipe varied)")

    for ax in (axL, axR):
        ax.set_ylim(-0.55, len(feats) - 0.3)
        ax.tick_params(axis="y", length=0)

    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(wspace=0.24)
    out = pathlib.Path(a.out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
