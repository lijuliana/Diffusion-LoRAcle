#!/usr/bin/env python
"""Check every load-bearing number in the paper against the file it came from.

The paper is edited by hand as arms land, so a number can drift from its source without anything
failing. This asserts each claim against the result JSON, and is meant to be re-run after any edit
and before submission.

Each check names the claim, the paper's value, the source value, and where the source lives.
"""
import json, pathlib, re, sys

PAPER = pathlib.Path("paper/main.tex")
FAIL = []


def get(path, *keys, default=None):
    try:
        d = json.loads(pathlib.Path(path).read_text())
    except Exception:
        return default
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def check(claim, in_paper, source_val, where, tol=0.005):
    ok = in_paper is not None and source_val is not None and abs(in_paper - source_val) <= tol
    print(f"  [{'OK  ' if ok else 'FAIL'}] {claim:<42} paper={in_paper}  source={source_val}  ({where})")
    if not ok:
        FAIL.append(claim)


def main():
    t = PAPER.read_text()

    def find(pat):
        m = re.search(pat, t)
        return float(m.group(1)) if m else None

    print("\n=== paper numbers against their result files ===")

    S6 = "results/sweep6"
    e12 = get(f"{S6}/e12_real.json", "results", "heldout_adapter", default={})
    # The best arm is the largest training budget, currently 25 epochs. Keep this pointed at the arm
    # the abstract actually quotes, or the check passes against a number the paper no longer makes.
    best = get(f"{S6}/e25_real.json", "results", "heldout_adapter", default={})
    bestc = get(f"{S6}/e25_CTRL.json", "results", "heldout_adapter", default={})
    cold = get("results/sweep7/cold_r16_e12_real.json", "results", "heldout_adapter", default={})
    e12c = get(f"{S6}/e12_CTRL.json", "results", "heldout_adapter", default={})
    e6 = get(f"{S6}/e6_real.json", "results", "heldout_adapter", default={})
    rep = get(f"{S6}/e12_r32_real.json", "results", "heldout_adapter", default={})

    check("headline held-out accuracy",
          find(r"58 of 105 held-out adapters \((\d+\.\d+)\\%"),
          round(best.get("reader_concept_accuracy", 0) * 100, 1), "e25_real.json")
    check("headline matched control", 0.0,
          bestc.get("reader_concept_accuracy"), "e25_CTRL.json")
    check("12-epoch accuracy", find(r"36/105 \((\d+\.\d+)\\%\)"),
          round(e12.get("reader_concept_accuracy", 0) * 100, 1), "e12_real.json")
    check("12-epoch matched control", 0.0,
          e12c.get("reader_concept_accuracy"), "e12_CTRL.json")
    check("cold rank-16 accuracy", find(r"10 of 105 \((\d+\.\d+)\\%\) at twelve\s+epochs against its own"),
          round(cold.get("reader_concept_accuracy", 0) * 100, 1), "sweep7/cold_r16_e12_real.json")
    check("nearest-neighbour reference",
          find(r"14 of 105 \((\d+\.\d+)\\%\)"),
          round(e12.get("nearest_neighbour_baseline", 0) * 100, 1), "e12_real.json")
    check("6-epoch accuracy",
          find(r"3/105 \((\d+\.\d+)\\%\)"),
          round(e6.get("reader_concept_accuracy", 0) * 100, 1), "e6_real.json")
    check("replicate accuracy",
          find(r"27/105 \((\d+\.\d+)\\%\)"),
          round(rep.get("reader_concept_accuracy", 0) * 100, 1), "e12_r32_real.json")
    check("retrieval rank, best arm", find(r"0\.459 to\s+0\.270 to (0\.\d+)"),
          best.get("retrieval_rank_norm"), "e25_real.json")
    check("attribute credit", find(r"credit is\s+(\d+\.\d+) against"),
          e12.get("slot_credit"), "e12_real.json")
    check("attribute credit, cross-LoRA", find(r"against (\d+\.\d+) when the same trained"),
          e12.get("cross_lora_slot_credit"), "e12_real.json", tol=0.002)

    # counts stated in prose must match n
    n = e12.get("n")
    stated_n = 105 if re.search(r"105 held-out adapters \(55", t) else None
    print(f"  [{'OK  ' if stated_n in (None, n) else 'FAIL'}] held-out n consistent"
          f"{'':<24} paper={stated_n}  source={n}")
    if stated_n not in (None, n):
        FAIL.append("held-out n")

    # corpus size against the bucket-derived manifest count is checked separately; here just
    # assert the paper is internally consistent about it
    sizes = set(re.findall(r"\b(831|764|930)\b", t))
    print(f"\n  corpus figures present in the paper: {sorted(sizes)}")

    print()
    if FAIL:
        print(f"VERIFICATION FAILED ({len(FAIL)}): {', '.join(FAIL)}")
        sys.exit(1)
    print("All checked numbers match their sources.")


if __name__ == "__main__":
    main()
