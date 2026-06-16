"""POC-0a evidence report: print the actual invariance residuals of the SVD encoder, so the
journal records magnitudes (not just 'tests pass'). CPU, no downloads.

Run: python3 scripts/poc0a_invariance_report.py
"""

from __future__ import annotations

import json

import torch

from ditloracle.encoding.svd_encoder import (
    compact_svd_from_factors,
    encode_module,
    invariant_signature,
)

torch.manual_seed(0)
DT = torch.float64


def rel_change(a, b):
    return float((a - b).norm() / a.norm().clamp_min(1e-30))


def main():
    out = {}

    # GL(r) gauge, condition number sweep
    B = torch.randn(64, 48, 8, dtype=DT)[0] if False else torch.randn(64, 8, dtype=DT)
    A = torch.randn(8, 48, dtype=DT)
    s0 = invariant_signature(encode_module(B, A))
    gl = {}
    for logcond in [0, 2, 4, 6, 8]:
        r = 8
        Q1, _ = torch.linalg.qr(torch.randn(r, r, dtype=DT))
        Q2, _ = torch.linalg.qr(torch.randn(r, r, dtype=DT))
        cond = torch.logspace(0, logcond, r, dtype=DT)
        G = Q1 @ torch.diag(cond) @ Q2
        s1 = invariant_signature(encode_module(B @ torch.linalg.inv(G), G @ A))
        gl[f"cond_1e{logcond}"] = rel_change(s0, s1)
    out["gl_gauge_rel_change_vs_condition"] = gl

    # coupled sign
    D = torch.diag(torch.sign(torch.randn(8, dtype=DT)))
    s1 = invariant_signature(encode_module(B @ D, D @ A))
    out["coupled_sign_rel_change"] = rel_change(s0, s1)

    # degenerate O(m)
    U, _ = torch.linalg.qr(torch.randn(50, 6, dtype=DT))
    V, _ = torch.linalg.qr(torch.randn(50, 6, dtype=DT))
    sv = torch.tensor([3.0, 3.0, 3.0, 2.0, 1.0, 0.5], dtype=DT)
    root = torch.diag(sv.sqrt())
    Bd, Ad = U @ root, root @ V.transpose(-1, -2)
    enc_d = encode_module(Bd, Ad)
    Uo, S, Vo = compact_svd_from_factors(Bd, Ad)
    cluster = next(c for c in enc_d.clusters if len(c) > 1)
    m = len(cluster)
    Q, _ = torch.linalg.qr(torch.randn(m, m, dtype=DT))
    G = torch.eye(6, dtype=DT)
    for a, ia in enumerate(cluster):
        for b, ib in enumerate(cluster):
            G[ia, ib] = Q[a, b]
    rootS = torch.diag(S.sqrt())
    Bg = (Uo @ rootS) @ G
    Ag = torch.linalg.inv(G) @ (rootS @ Vo.transpose(-1, -2))
    safe0 = invariant_signature(enc_d, degeneracy_safe=True)
    safe1 = invariant_signature(encode_module(Bg, Ag), degeneracy_safe=True)
    naive0 = invariant_signature(enc_d, degeneracy_safe=False)
    naive1 = invariant_signature(encode_module(Bg, Ag), degeneracy_safe=False)
    out["degenerate_clusters_detected"] = [len(c) for c in enc_d.clusters]
    out["degeneracy_safe_rel_change"] = rel_change(safe0, safe1)
    out["naive_rel_change_on_degeneracy"] = rel_change(naive0, naive1)

    # negative control: genuine change
    B2 = B.clone()
    B2[:, 0] += 0.5 * torch.randn_like(B2[:, 0])
    s1 = invariant_signature(encode_module(B2, A))
    out["genuine_change_rel_change"] = rel_change(s0, s1)

    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
