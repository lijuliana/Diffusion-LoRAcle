#!/usr/bin/env python
"""Figure: which attribute of a compositional concept the interpreter recovers.

A compositional concept names four attributes (family, subject, medium, palette). Scored
separately over the compositional held-out adapters, family is easiest and the rendering
medium hardest. Recomputed from the stored generations so the figure cannot drift from the
paper's numbers (family 0.80, subject 0.47, palette 0.45, medium 0.40 at 12 epochs).
"""
import argparse, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use(str(pathlib.Path(__file__).parent.parent / "paper" / "figstyle.mplstyle"))

SLOTS = ["family", "subject", "medium", "palette"]
BLUE, MUTED, INK2 = "#0072B2", "#c4c4c4", "#52514e"


def slot_values(concept: str):
    parts = [q for q in concept.split("__") if q]
    if len(parts) != 4:
        return None
    return {n: p.replace("gen_", "").replace("_", " ").strip()
            for n, p in zip(SLOTS, parts)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/sweep6/e12_real.json")
    ap.add_argument("--out", default="paper/figures/fig_slot_accuracy.pdf")
    a = ap.parse_args()

    samples = json.loads(pathlib.Path(a.results).read_text())["results"]["heldout_adapter"]["samples"]
    per = {s: [] for s in SLOTS}
    for x in samples:
        sv = slot_values(x["true"])
        if sv is None:
            continue
        low = x["said"].lower()
        for s, v in sv.items():
            per[s].append(v in low)
    n = len(per["family"])
    acc = {s: sum(v) / len(v) for s, v in per.items()}
    print({s: round(v, 3) for s, v in acc.items()}, f"n={n}")

    order = sorted(SLOTS, key=lambda s: -acc[s])
    y = list(range(len(order)))[::-1]
    fig, ax = plt.subplots(figsize=(2.9, 1.55))
    z = 1.959964
    for yi, s in zip(y, order):
        k = sum(per[s]); m = len(per[s]); ph = k / m
        c = (ph + z * z / (2 * m)) / (1 + z * z / m)
        h = (z / (1 + z * z / m)) * ((ph * (1 - ph) / m + z * z / (4 * m * m)) ** 0.5)
        ax.plot([0, acc[s]], [yi, yi], color=MUTED, lw=1.0, zorder=1, solid_capstyle="butt")
        ax.plot([c - h, c + h], [yi, yi], color=BLUE, lw=1.1, alpha=0.65, zorder=2,
                solid_capstyle="butt")
        ax.plot(acc[s], yi, "o", color=BLUE, zorder=3)
        ax.annotate(f"{acc[s]:.2f}", (acc[s], yi), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([s.capitalize() for s in order])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f"Attribute recovered (held-out, $n={n}$)")
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4)
    out = pathlib.Path(a.out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
