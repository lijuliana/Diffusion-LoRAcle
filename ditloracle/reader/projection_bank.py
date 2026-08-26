"""Project a LoRA direction into the BASE MODEL's residual stream, using the base model's own weights.

This is the piece of LoRAcle that matters most and the piece we were missing. Their
`ProjectionBank` (extract_hf/tokenize_lora_fixed.py) builds per-layer frozen maps from LoRA weight
space into the residual stream, initialised **from the base model itself**: `attn_proj` copied from
`o_proj`, `mlp_proj` from `down_proj`.

The reasoning, which is what makes the tokens mean anything: a LoRA direction living in attention-head
space is not interpretable on its own. Push it through `o_proj` — the matrix that literally writes
head-space back into the residual stream — and it becomes *what this direction contributes to the
model's shared communication channel*. A random projection preserves geometry and destroys that.

FLUX.2-klein analogues (read off the architecture, not assumed):
  attention  `attn.to_out`        <- the o_proj analogue: head-space -> residual
  MLP        `ff.linear_out`      <- the down_proj analogue: intermediate -> residual
Modules that already WRITE to the residual (`to_out`, `ff.linear_out`) need no projection on their
output side: their B factor is already residual-shaped.

⚠ WHERE THIS DEPARTS FROM LoRAcle, AND WHY IT IS THE PROJECT'S BIGGEST UNKNOWN. Their LoRA base model
and their reader are the same architecture, so a projected direction lands in the *reader's own*
residual stream and the tokens are native. Ours does not: klein's residual is 3072 wide and the reader
is a text LLM of another width, trained on another modality. This module gets a direction as far as
*klein's* residual stream, which is the furthest anything is grounded; the remaining hop to the reader
is the frozen bridge in `dataset.bridge`, and nothing in LoRAcle's results speaks to whether it
carries meaning. Treat the bridge as the experiment, not as plumbing.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

# module-name fragments -> which base-model matrix writes that module's output to the residual
_ATTN_OUT = ("attn.to_out", "attn.to_add_out", "to_out")
_MLP_OUT = ("ff.linear_out", "ff.net.2", "ff_context.net.2", "linear_out")


def writes_to_residual(name: str) -> bool:
    """True if this module's OUTPUT side already lives in residual coordinates."""
    return any(s in name for s in _ATTN_OUT + _MLP_OUT)


def _block_index(name: str) -> int | None:
    m = re.search(r"(?:blocks?|block)[._](\d+)", name)
    return int(m.group(1)) if m else None


def _is_mlp(name: str) -> bool:
    return any(s in name for s in ("ff.", "mlp", "linear_in", "linear_out", "net."))


class KleinProjectionBank:
    """Frozen per-block maps from klein module space into klein's residual stream.

    Built by reading `to_out` / `ff.linear_out` straight out of the base checkpoint, exactly as
    LoRAcle reads `o_proj` / `down_proj`. Nothing here is trained.
    """

    def __init__(self, weights: dict[str, torch.Tensor], d_model: int):
        self.d_model = d_model
        self.attn: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}
        for k, w in weights.items():
            b = _block_index(k)
            if b is None:
                continue
            if any(s in k for s in _ATTN_OUT):
                self.attn[b] = w.float()
            elif any(s in k for s in _MLP_OUT):
                self.mlp[b] = w.float()

    @classmethod
    def from_safetensors(cls, model_dir: str, d_model: int = 3072) -> "KleinProjectionBank":
        """Read ONLY the write-back matrices out of a local base checkpoint.

        Deliberately does not instantiate the pipeline: we need two matrices per block, not a 15 GB
        model, and the mint boxes are busy.
        """
        from safetensors.torch import safe_open
        keep: dict[str, torch.Tensor] = {}
        for f in sorted(Path(model_dir).rglob("*.safetensors")):
            with safe_open(str(f), framework="pt") as fh:
                for k in fh.keys():
                    if k.endswith(".weight") and (any(s in k for s in _ATTN_OUT)
                                                  or any(s in k for s in _MLP_OUT)):
                        keep[k] = fh.get_tensor(k)
        if not keep:
            raise FileNotFoundError(f"no to_out / ff.linear_out tensors under {model_dir}")
        return cls(keep, d_model)

    def project(self, name: str, direction: torch.Tensor) -> torch.Tensor | None:
        """Map one direction from this module's own space into klein's residual stream.

        `direction` is the module-output-side singular vector. Returns None when the module has no
        write-back matrix in the bank, so the caller can skip rather than silently emit a raw vector
        in the wrong basis (which is what we were doing before this module existed).
        """
        d = direction.float()
        # Decide by DIMENSION first. writes_to_residual() tests FLUX/diffusers name fragments that
        # match none of klein's modules, so `img_mlp.2` (already residual-width on its output side)
        # fell through to the bank lookup and was dropped on the shape check. A direction that is
        # already d_model wide needs no projection whatever its name says.
        if d.shape[0] == self.d_model:
            return d
        if writes_to_residual(name):
            return None
        b = _block_index(name)
        if b is None:
            return None
        W = (self.mlp if _is_mlp(name) else self.attn).get(b)
        if W is None or W.shape[1] != d.shape[0]:
            return None
        return W @ d      # [d_model, d_mod] @ [d_mod] -> [d_model], now residual-shaped
