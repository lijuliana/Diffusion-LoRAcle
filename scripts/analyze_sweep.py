#!/usr/bin/env python
"""Pair every real arm with its matched control and report the comparison that decides the result.

Written before sweep #6's arms landed, so the decision rule is fixed in advance rather than chosen
after seeing the numbers. It encodes the rules this project arrived at the hard way:

  TRAIN FIRST      An arm that has not fit its training set is void, not interpretable. Two sweeps
                   were read as evidence before this rule existed.
  MATCHED CONTROL  Compare a real arm only against a control trained at the SAME epochs, learning
                   rate and token budget. An arm compared against a shorter-trained control tests
                   step count, not the intended variable.
  FISHER, NOT CHANCE   Beating chance is the weaker claim and is what a label prior can do. The
                   claim that matters is beating the shuffled-token control on the same setting.
  HOLM             Several arms are tested, so raw p-values overstate. Report both.
  NO READS-SLOT    A no-injection control with zeroed tokens once scored READS-slot +0.054, so that
                   metric produces spurious positives at this n and is excluded.
"""
import argparse, json, pathlib, re


def load(d: pathlib.Path):
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.stem == "preflight":
            continue
        try:
            r = json.loads(f.read_text()).get("results") or {}
        except Exception:
            continue
        ha = r.get("heldout_adapter")
        if not ha:
            continue
        out[f.stem] = {
            "train": (r.get("train") or {}).get("reader_concept_accuracy"),
            "acc": ha["reader_concept_accuracy"], "n": ha["n"],
            "nearest": ha.get("nearest_neighbour_baseline"),
            "rank": ha.get("retrieval_rank_norm"),
        }
    return out


def pair(name: str, known: set = frozenset()) -> str | None:
    """Map a real arm to the name of its matched control, across both naming schemes.

    sweep #6 uses e<N>_real / e<N>_CTRL; sweep #5 used ps_warm_e<N> / ps_CONTROL_shuffled_e<N>.
    Supporting both lets the script be validated against sweep #5, whose answer is already known
    (3/84 against 0/84, Fisher p=0.123), instead of being trusted on first use.
    """
    if "CTRL" in name or "CONTROL" in name:
        return None
    m = re.search(r"e(\d+)", name)
    if not m:
        return None
    ep = m.group(1)
    for cand in (f"e{ep}_CTRL", f"ps_CONTROL_shuffled_e{ep}", f"CONTROL_shuffled_e{ep}",
                 name.replace("_real", "_CTRL")):
        if not known or cand in known:
            return cand
    return None


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    out, prev = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, max(prev, (len(pvals) - rank) * pvals[i]))
        out[i], prev = adj, adj
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/sweep6")
    ap.add_argument("--concepts", type=int, default=155)
    a = ap.parse_args()

    arms = load(pathlib.Path(a.dir))
    if not arms:
        raise SystemExit(f"no completed arms in {a.dir}")
    chance = 1 / a.concepts

    print(f"\n=== {a.dir}: {len(arms)} completed arms, {a.concepts} concepts, chance {chance:.4f} ===\n")
    print(f"{'arm':<24}{'TRAIN':>8}{'held-out':>10}{'k/n':>9}{'xchance':>9}{'rank':>7}")
    print("-" * 67)
    for k, v in sorted(arms.items()):
        n = v["n"]; kk = round(v["acc"] * n)
        print(f"{k:<24}{v['train']:>8.3f}{v['acc']:>10.3f}{f'{kk}/{n}':>9}"
              f"{v['acc']/chance:>9.1f}{v['rank']:>7.3f}")

    print(f"\n=== paired against matched controls ===")
    try:
        from scipy.stats import fisher_exact
    except Exception:
        print("  scipy unavailable; cannot compute Fisher exact")
        return

    rows, pv = [], []
    for name, v in sorted(arms.items()):
        c = pair(name, set(arms))
        if not c or c not in arms:
            continue
        cv = arms[c]
        n, nc = v["n"], cv["n"]
        kk, kc = round(v["acc"] * n), round(cv["acc"] * nc)
        _, p = fisher_exact([[kk, n - kk], [kc, nc - kc]], alternative="greater")
        rows.append((name, c, kk, n, kc, nc, p, v["train"], cv["train"]))
        pv.append(p)

    if not rows:
        print("  no real/control pairs complete yet")
        return
    adj = holm(pv)
    print(f"{'real':<16}{'control':<16}{'real':>8}{'ctrl':>8}{'p':>9}{'p_holm':>9}")
    print("-" * 66)
    for (name, c, kk, n, kc, nc, p, tr, tc), pa in zip(rows, adj):
        flag = "  SIGNIFICANT" if pa < 0.05 else ""
        print(f"{name:<16}{c:<16}{f'{kk}/{n}':>8}{f'{kc}/{nc}':>8}{p:>9.4f}{pa:>9.4f}{flag}")
        if tr is not None and tr < 0.02:
            print(f"    ^ VOID: training accuracy {tr:.3f}; this arm has not fit its training set")

    print("\n  Reminder: beating chance is the weaker claim. The column that decides the result is")
    print("  p_holm against the matched control.")


if __name__ == "__main__":
    main()
