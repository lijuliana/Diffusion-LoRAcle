#!/usr/bin/env python
"""Figure 1: held-out accuracy against training budget, with the matched control.

This is the paper's central claim in one panel. The interpreter is at the floor through six epochs
and jumps at twelve, while its shuffled-token control stays at zero throughout, so the reader can see
both that the effect is real and that it appears suddenly.

Reads the result JSONs rather than hard-coded numbers, so the figure cannot drift from the data.
"""
import argparse, json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# One shared style for every figure in the paper: serif to match the body text, explicit sizes at
# final column width. Default sans-serif and lowercase axis text are recognisable low-effort tells.
plt.style.use(str(pathlib.Path(__file__).parent.parent / "paper" / "figstyle.mplstyle"))

# One hue carries identity (the interpreter), a muted grey carries the controls, and the reference
# lines are lighter still. Text stays in ink colours so identity is never colour-alone: every series
# is also directly labelled.
INK, MUTED, FAINT = "#1a1a1a", "#8a8a8a", "#c4c4c4"
REAL, CTRL = "#0072B2", "#b0b0b0"   # Okabe-Ito blue, matching Figure 2


def load(d: pathlib.Path):
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.stem == "preflight":
            continue
        try:
            R = json.loads(f.read_text()).get("results") or {}
        except Exception:
            continue
        ha = R.get("heldout_adapter")
        if ha:
            out[f.stem] = ha
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep6", default="results/sweep6")
    ap.add_argument("--out", default="paper/figures/fig1_epoch_threshold.pdf")
    a = ap.parse_args()

    arms = load(pathlib.Path(a.sweep6))
    if not arms:
        raise SystemExit("no completed arms found")

    def acc(name):
        return arms[name]["reader_concept_accuracy"] if name in arms else None

    real_pts, ctrl_pts = [], []
    for ep, rn, cn in ((6, "e6_real", "e6_CTRL"),
                       (12, "e12_real", "e12_CTRL"),
                       (25, "e25_real", "e25_CTRL")):
        if acc(rn) is not None:
            real_pts.append((ep, acc(rn)))
        if acc(cn) is not None:
            ctrl_pts.append((ep, acc(cn)))
    # the 12-epoch replicate, plotted as a second marker at the same x
    rep = acc("e12_r32_real")

    # Take the memorisation reference from a REAL arm only. A control's tokens are shuffled, so its
    # nearest-neighbour figure describes a scrambled corpus and is meaningless; picking the first
    # arm in dictionary order silently used a control's 0.0095 instead of the real 0.133.
    nearest = next((v["nearest_neighbour_baseline"] for k, v in arms.items()
                    if "CTRL" not in k and "CONTROL" not in k
                    and v.get("nearest_neighbour_baseline")), None)
    n = next(iter(arms.values()))["n"]
    chance = 1 / 155

    # Built at final single-column width rather than shrunk from a larger figure.
    fig, ax = plt.subplots(figsize=(3.25, 2.3), constrained_layout=True)
    # Reference labels sit just ABOVE their line and inside the axes. Placing them at the line's own
    # y with va="center" let the dashed rule strike through the text.
    # Place each reference label where the interpreter line is not: chance sits low on the left,
    # the memorisation line is crossed early so its label goes to the right of the crossing.
    for y, lab, col, xt, ha in ((chance, "Chance", FAINT, 0.6, "left"),
                                (nearest, "Nearest-neighbour retrieval", MUTED, 25.4, "right")):
        if y:
            ax.axhline(y, color=col, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.text(xt, y + 0.012, lab, va="bottom", ha=ha, fontsize=6.2, color=MUTED)

    if real_pts:
        xs, ys = zip(*real_pts)
        ax.plot(xs, ys, "-o", color=REAL, lw=1.6, ms=5, zorder=3, clip_on=False)
        ax.text(xs[-1], ys[-1] + 0.030, "Interpreter", color=REAL, fontsize=7.5,
                ha="right", fontweight="bold")
    if rep is not None:
        ax.plot([12], [rep], "o", color=REAL, ms=5, mfc="white", mew=1.4, zorder=3)
        ax.text(12.8, rep, "Replicate", color=REAL, fontsize=6.2, va="center", ha="left")
    if ctrl_pts:
        xs, ys = zip(*ctrl_pts)
        ax.plot(xs, ys, "-o", color=CTRL, lw=1.6, ms=5, zorder=2, clip_on=False)
        # Above its own line, not below: below the final point put the label under the axis.
        ax.text(xs[-1], ys[-1] + 0.022, "Shuffled-token control", color=MUTED, fontsize=7,
                ha="right", va="bottom")

    ax.set_xlabel("Training epochs", color=INK)
    ax.set_ylabel(f"Held-out accuracy ($n={n}$)", color=INK)
    ax.set_xticks([0, 6, 12, 25]); ax.set_xlim(0, 26)
    ax.set_ylim(-0.02, max(0.42, (max(y for _, y in real_pts) if real_pts else 0) + 0.08))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=FAINT, lw=0.6, alpha=0.5); ax.set_axisbelow(True)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {a.out}")
    print(f"  interpreter: {real_pts}")
    print(f"  control    : {ctrl_pts}")
    print(f"  replicate at 12: {rep}   nearest-neighbour: {nearest}   chance: {chance:.4f}")


if __name__ == "__main__":
    main()
