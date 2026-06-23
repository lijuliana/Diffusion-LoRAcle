"""POC-0d real-data triage: run the parser + real SVD over a sample of FLUX.1-dev LoRAs and report
the gate numbers from the execution plan (§B.13.3):

  (i)   parseable fraction  — how many files map cleanly to canonical modules
  (ii)  base-homogeneity    — how many genuinely target FLUX.1-dev (scheme + module-shape check)
  (iii) SVD conditioning    — σ-gap distribution on high-rank adapters (where directions get
                              ill-conditioned, exactly the §B.4.3 caveat)
  + rank distribution, module coverage, format breakdown.

Run: PYTHONPATH=. python3 scripts/poc0d_triage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from ditloracle.encoding.svd_encoder import compact_svd_from_factors, degeneracy_clusters
from ditloracle.formats.flux_lora import classify, parse_keys
from ditloracle.formats.safetensors_io import load_lora_factors, read_keys

SAMPLE_DIR = Path("assets/flux_lora_sample")
EXPECTED_FLUX_WIDTH = 3072  # FLUX.1-dev hidden size; residual-stream-native dim


def sigma_gap_stats(S: torch.Tensor) -> dict:
    """Conditioning descriptors for one module's spectrum."""
    S = S.to(torch.float64).sort(descending=True).values
    if S.numel() < 2:
        return {}
    smax = float(S[0].clamp_min(1e-30))
    gaps = (S[:-1] - S[1:]) / smax                     # normalized adjacent gaps
    clusters = degeneracy_clusters(S, rel_tol=1e-3)
    n_degenerate = sum(len(c) for c in clusters if len(c) > 1)
    return {
        "rank": int(S.numel()),
        "sigma_max": smax,
        "sigma_min_over_max": float(S[-1] / smax),
        "min_norm_gap": float(gaps.min()),
        "median_norm_gap": float(gaps.median()),
        "n_near_degenerate_dirs": int(n_degenerate),
    }


def main():
    manifest = json.loads((SAMPLE_DIR / "manifest.json").read_text())
    n = len(manifest)
    results = {"n_files": n, "per_file": [], "summary": {}}

    parseable = 0
    flux_width_ok = 0
    classes = {}
    schemes = {}
    ranks = []
    all_min_gaps = []
    all_degenerate = []
    high_rank_files = 0

    for item in manifest:
        path = item["path"]
        rec = {"repo": item["repo"], "size_mb": item["size_mb"]}
        try:
            keys = read_keys(path)
            parsed = parse_keys(keys)
            verdict = classify(parsed)
            rec["scheme"] = parsed.scheme.value
            rec["n_modules"] = parsed.n_modules
            rec["verdict"] = verdict
            schemes[parsed.scheme.value] = schemes.get(parsed.scheme.value, 0) + 1
            classes[verdict] = classes.get(verdict, 0) + 1

            factors = load_lora_factors(path)
            rec["n_factor_pairs"] = len(factors)
            if factors:
                parseable += 1
                # rank + residual-width check on a representative module
                sample_ranks = [v["r"] for v in factors.values()]
                rec["rank_mode"] = max(set(sample_ranks), key=sample_ranks.count)
                ranks.append(rec["rank_mode"])
                # base-homogeneity: do module dims match FLUX.1-dev residual width?
                widths = set()
                for v in factors.values():
                    widths.add(v["A"].shape[1])   # d_in
                    widths.add(v["B"].shape[0])    # d_out
                rec["has_flux_width"] = EXPECTED_FLUX_WIDTH in widths
                if rec["has_flux_width"]:
                    flux_width_ok += 1

                # SVD conditioning on the highest-rank module in this file
                hi = max(factors.values(), key=lambda v: v["r"])
                if hi["r"] >= 32:
                    high_rank_files += 1
                _, S, _ = compact_svd_from_factors(hi["B"], hi["A"])
                gs = sigma_gap_stats(S)
                rec["svd"] = gs
                if gs:
                    all_min_gaps.append(gs["min_norm_gap"])
                    all_degenerate.append(gs["n_near_degenerate_dirs"])
        except Exception as e:
            rec["error"] = str(e)[:120]
        results["per_file"].append(rec)

    def pct(x):
        return round(100 * x / n, 1)

    results["summary"] = {
        "parseable_fraction_pct": pct(parseable),
        "base_homogeneity_flux_width_pct": pct(flux_width_ok),
        "scheme_breakdown": schemes,
        "verdict_breakdown": classes,
        "rank_distribution": {str(r): ranks.count(r) for r in sorted(set(ranks))},
        "n_high_rank_files(>=32)": high_rank_files,
        "svd_conditioning": {
            "median_of_min_norm_gaps": round(float(torch.tensor(all_min_gaps).median()), 6) if all_min_gaps else None,
            "worst_min_norm_gap": round(float(min(all_min_gaps)), 6) if all_min_gaps else None,
            "total_near_degenerate_dirs": int(sum(all_degenerate)),
            "files_with_any_degeneracy": int(sum(1 for d in all_degenerate if d > 0)),
        },
    }
    print(json.dumps(results["summary"], indent=2))
    out = SAMPLE_DIR / "../triage_result.json"
    Path("results/poc0d_triage__flux_sample.json").write_text(json.dumps(results, indent=2))
    print(f"\nfull per-file report -> results/poc0d_triage__flux_sample.json")
    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
