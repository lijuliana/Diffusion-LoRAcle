"""SVD-direction-token encoding of LoRA weight diffs.

This is the instrument. A LoRA module is a low-rank diff ΔW = (α/r)·B·A. The reader will
consume per-direction tokens derived from the SVD of ΔW. The *correctness requirement* (POC-0a)
is that the extracted features are invariant to exactly the symmetries the data has and to nothing
more:

  * GL(r) gauge        — (B,A) ↦ (B G⁻¹, G A) for G∈GL(r) leaves ΔW (hence the SVD) unchanged.
  * coupled sign       — (uᵢ,vᵢ) ↦ (−uᵢ,−vᵢ) leaves ΔW unchanged; the SVD routine picks signs
                         arbitrarily, so we must canonicalize them.
  * degenerate blocks  — when σᵢ≈σⱼ, the singular *vectors* are only defined up to a shared
                         Q∈O(m) rotation of the subspace; individual directions are meaningless,
                         only the subspace (its projector) is invariant.

See design doc §3 / §B.4 for the math. Everything here is CPU, no downloads.

Implementation note: we never materialize the dense d_out×d_in product. The compact SVD is
obtained from QR on the thin factors (cost O((d_out+d_in)r² + r³)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------------------------
# Compact SVD of ΔW = scale · B · A, computed from the factors without forming the product.
# --------------------------------------------------------------------------------------
def compact_svd_from_factors(
    B: Tensor, A: Tensor, scale: float = 1.0
) -> tuple[Tensor, Tensor, Tensor]:
    """Return (U, S, V) with U:(d_out,k), S:(k,), V:(d_in,k), columns = singular directions.

    ΔW = scale · B @ A, B:(d_out,r), A:(r,d_in). k = r (the thin rank; trailing σ may be ~0
    if the factors are rank-deficient). U, V have orthonormal columns; ΔW = U diag(S) Vᵀ.

    Derivation: B = Q_B R_B, Aᵀ = Q_A R_A ⇒ ΔW = scale·Q_B (R_B R_Aᵀ) Q_Aᵀ. SVD the small
    r×r core M = scale·R_B R_Aᵀ = U_M S V_Mᵀ, then U = Q_B U_M, V = Q_A V_M.
    """
    B = B.to(torch.float64)
    A = A.to(torch.float64)
    Qb, Rb = torch.linalg.qr(B, mode="reduced")              # (d_out,r), (r,r)
    Qa, Ra = torch.linalg.qr(A.transpose(-1, -2), mode="reduced")  # (d_in,r), (r,r)
    M = scale * (Rb @ Ra.transpose(-1, -2))                  # (r,r)
    Um, S, Vmh = torch.linalg.svd(M, full_matrices=False)    # (r,r),(r,),(r,r)
    U = Qb @ Um                                              # (d_out,r)
    V = Qa @ Vmh.transpose(-1, -2)                           # (d_in,r)
    return U, S, V


def canonicalize_signs(U: Tensor, V: Tensor) -> tuple[Tensor, Tensor]:
    """Fix the coupled-sign gauge deterministically.

    For each direction i, the pair (uᵢ,vᵢ) may be jointly flipped. We choose the sign so that
    the entry of the *coupled* vector [uᵢ; vᵢ] with largest absolute value is positive. Because
    uᵢ,vᵢ are unit vectors, that entry has magnitude ≥ 1/√(d_out+d_in) > 0, so the rule is
    well-defined (the only ambiguity left is an exact tie, handled by argmax determinism).
    """
    coupled = torch.cat([U, V], dim=0)            # (d_out+d_in, k)
    idx = coupled.abs().argmax(dim=0)             # (k,)
    pivot = coupled[idx, torch.arange(coupled.shape[1])]
    s = torch.sign(pivot)
    s = torch.where(s == 0, torch.ones_like(s), s)
    return U * s, V * s


def degeneracy_clusters(S: Tensor, rel_tol: float = 1e-3, abs_tol: float = 1e-8) -> list[list[int]]:
    """Group near-equal singular values into clusters (descending order assumed).

    Two adjacent σ are in the same cluster if |σᵢ-σⱼ| ≤ rel_tol·max(σ) (or both ≤ abs_tol, i.e.
    both effectively zero). Singletons are simple singular values; size>1 clusters are degenerate
    subspaces where individual directions are O(m)-ambiguous.
    """
    S = S.to(torch.float64)
    n = S.shape[0]
    if n == 0:
        return []
    smax = float(S.max().clamp_min(abs_tol))
    thr = rel_tol * smax
    clusters: list[list[int]] = [[0]]
    for i in range(1, n):
        if abs(float(S[i] - S[i - 1])) <= thr:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


@dataclass
class ModuleEncoding:
    """Canonical SVD encoding of a single LoRA module."""

    sigma: Tensor          # (k,) singular values, descending
    U: Tensor              # (d_out, k) sign-canonicalized left vectors (columns)
    V: Tensor              # (d_in, k)  sign-canonicalized right vectors (columns)
    frob: float            # ‖ΔW‖_F (overall scale, fed separately)
    clusters: list[list[int]]

    @property
    def k(self) -> int:
        return self.sigma.shape[0]


def encode_module(
    B: Tensor,
    A: Tensor,
    alpha: float | None = None,
    r: int | None = None,
    use_rslora: bool = False,
    rel_tol: float = 1e-3,
) -> ModuleEncoding:
    """Encode one LoRA module's (B, A) factors into a canonical, gauge-fixed SVD representation.

    The LoRA scale α/r (or α/√r for rsLoRA) is folded in before the SVD so magnitudes are
    comparable across modules (design doc §B.4.6 / §3 normalization).
    """
    r_eff = r if r is not None else B.shape[1]
    if alpha is None:
        scale = 1.0
    elif use_rslora:
        scale = alpha / (r_eff ** 0.5)
    else:
        scale = alpha / r_eff

    U, S, V = compact_svd_from_factors(B, A, scale=scale)
    U, V = canonicalize_signs(U, V)
    frob = float(torch.linalg.vector_norm(S))
    clusters = degeneracy_clusters(S, rel_tol=rel_tol)
    return ModuleEncoding(sigma=S, U=U, V=V, frob=frob, clusters=clusters)


# --------------------------------------------------------------------------------------
# Invariant signatures: flat feature vectors guaranteed invariant to GL(r)/sign/degeneracy.
# Used by (a) the POC-0a invariance tests and (b) the POC-1 linear probe.
# --------------------------------------------------------------------------------------
def subspace_projector_diag(M: Tensor, cols: list[int]) -> Tensor:
    """Diagonal of the projector onto span(M[:,cols]); invariant to any O(m) basis change.

    For a degenerate cluster the individual columns are ambiguous, but P = M_sub M_subᵀ is not,
    and its diagonal (per-coordinate energy) is a compact O(m)-invariant descriptor.
    """
    Msub = M[:, cols]                       # (d, m)
    return (Msub * Msub).sum(dim=1)          # (d,) == diag(Msub Msubᵀ)


def invariant_signature(
    enc: ModuleEncoding,
    spectral_normalize: bool = True,
    degeneracy_safe: bool = True,
) -> Tensor:
    """A flat feature vector invariant to GL(r) (via SVD), coupled sign (via canonicalization),
    and — if ``degeneracy_safe`` — O(m) rotations within degenerate blocks (via subspace
    projector diagonals instead of ambiguous individual directions).

    Layout: [ normalized σ | per-direction direction features ].  Direction features are the
    sign-canonicalized (uᵢ, vᵢ) for simple values; for a degenerate cluster they are replaced by
    the (shared, O(m)-invariant) projector diagonals broadcast to each member, so the vector keeps
    a fixed per-direction layout while staying invariant.
    """
    sigma = enc.sigma
    if spectral_normalize and enc.frob > 0:
        sigma = sigma / enc.frob

    dir_feats = []
    if degeneracy_safe:
        u_diag_cache: dict[tuple[int, ...], Tensor] = {}
        v_diag_cache: dict[tuple[int, ...], Tensor] = {}
        for cluster in enc.clusters:
            key = tuple(cluster)
            if len(cluster) == 1:
                i = cluster[0]
                dir_feats.append(torch.cat([enc.U[:, i], enc.V[:, i]]))
            else:
                # O(m)-invariant: every member of the cluster gets the same projector-diagonal
                # descriptor, so individual (ambiguous) directions never enter the signature.
                if key not in u_diag_cache:
                    u_diag_cache[key] = subspace_projector_diag(enc.U, cluster)
                    v_diag_cache[key] = subspace_projector_diag(enc.V, cluster)
                feat = torch.cat([u_diag_cache[key], v_diag_cache[key]])
                for _ in cluster:
                    dir_feats.append(feat)
    else:
        for i in range(enc.k):
            dir_feats.append(torch.cat([enc.U[:, i], enc.V[:, i]]))

    return torch.cat([sigma, *dir_feats]) if dir_feats else sigma
