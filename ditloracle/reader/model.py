"""The reader: a language model shown an adapter's weights, which writes what the adapter draws.

The LoRAcle port to image diffusion transformers. The model never sees an image, a prompt used to
train the adapter, or a generated sample. It sees directions taken from the adapter's own weight
matrices and must name what it depicts.

**Injection is parameter-free and norm-matched**, following the ancestor recipe (design doc §B.5.2):
weight tokens occupy placeholder positions in the prompt, and at those positions the residual stream
is updated as

    h <- h + (||h|| / ||v||) * v

with `v` the bridged, residual-facing singular direction. Nothing here is learned. That is the whole
point, and it is not a stylistic preference: the released LoRAcle checkpoints carry no projector, and
with fewer than a hundred training adapters a learned weights-to-hidden map (~10^5-10^6 parameters) is
the fastest available route to memorising the corpus. The only trained parameters are the LoRA on the
backbone and a small module embedding; a learned bridge is the listed ablation (§B.12.1), not this.

Scaling by ||h||/||v|| makes the injected direction the same size as whatever is already in the
residual stream at that position, so one hyper-parameter (which layer) replaces a fitted scale, and a
direction with a tiny singular value cannot arrive at full residual magnitude. Directions below the
sigma floor are dropped upstream in `encoding.injection_tokens` for the same reason: their ||v|| is
near zero and the ratio explodes (§B.12.1).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WeightTokenReader(nn.Module):
    def __init__(self, backbone, d_token: int, n_modules: int,
                 learned_bridge: bool = False, dtype=torch.float32):
        super().__init__()
        self.backbone = backbone
        d_model = backbone.get_input_embeddings().embedding_dim
        if d_token != d_model and not learned_bridge:
            raise ValueError(
                f"parameter-free injection needs tokens already at the reader width "
                f"(d_token={d_token}, d_model={d_model}). Build the dataset with "
                f"d_token={d_model}, or pass learned_bridge=True to run the ablation.")
        # The ABLATION arm only. Absent by default so the default model has no weights-to-hidden map.
        self.bridge = nn.Linear(d_token, d_model, dtype=dtype) if learned_bridge else None
        if self.bridge is not None:
            nn.init.normal_(self.bridge.weight, std=0.01)
            nn.init.zeros_(self.bridge.bias)
        # Tells the reader WHERE a direction came from. Small, and the one learned part of the
        # encoding; the direction itself is never transformed by a learned map in the default arm.
        self.module_emb = nn.Embedding(n_modules, d_model, dtype=dtype)
        nn.init.normal_(self.module_emb.weight, std=0.01)

    def _inject(self, emb: torch.Tensor, tokens: torch.Tensor, module_ids: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """h <- h + (||h||/||v||) v at the placeholder positions."""
        v = self.bridge(tokens) if self.bridge is not None else tokens
        v = v.to(emb.dtype) + self.module_emb(module_ids).to(emb.dtype)
        h_norm = emb.norm(dim=-1, keepdim=True)
        v_norm = v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return emb + (h_norm / v_norm) * v * mask.unsqueeze(-1).to(emb.dtype)

    def _prepare(self, tokens, module_ids, tok_mask, input_ids, attention_mask):
        """Weight tokens occupy the FIRST `n_w` positions of the prompt, which are placeholders."""
        emb = self.backbone.get_input_embeddings()(input_ids)
        n_w = tokens.shape[1]
        head = self._inject(emb[:, :n_w], tokens, module_ids, tok_mask)
        return torch.cat([head, emb[:, n_w:]], dim=1)

    def forward(self, tokens, module_ids, tok_mask, input_ids, attention_mask, labels=None):
        inputs_embeds = self._prepare(tokens, module_ids, tok_mask, input_ids, attention_mask)
        return self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    @torch.no_grad()
    def generate(self, tokens, module_ids, tok_mask, input_ids, attention_mask, **kw):
        inputs_embeds = self._prepare(tokens, module_ids, tok_mask, input_ids, attention_mask)
        return self.backbone.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kw)


def collate(batch, tokenizer, prompt: str, placeholder_id: int, max_tokens: int, device="cpu"):
    """Build `[placeholder]*n_w + prompt + target` at the TOKEN level, plus the weight tensors.

    The prefix is constructed from token IDs, never by repeating a placeholder CHARACTER. Repeating a
    character and tokenising it does not give one token per character — the tokeniser merges runs, so
    128 `?` characters became 66 tokens and every arm of the sweep died with
    `size of tensor a (66) must match tensor b (128)`. The injection needs exactly one prompt position
    per weight token, so the count has to be exact by construction. This is why LoRAcle builds its
    prefix with `build_placeholder_prefix_ids` rather than from a string.

    Adapters carry different numbers of tokens (different module sets), so the batch pads to the
    longest and `tok_mask` zeroes injection at padded positions. Token COUNT therefore carries recipe
    information; concept is decorrelated from recipe by construction here, so it cannot shortcut the
    concept, but it is a channel to watch if that ever changes.
    """
    n_w = max(min(b.tokens.shape[0], max_tokens) for b in batch)
    d = batch[0].tokens.shape[1]
    toks = torch.zeros(len(batch), n_w, d)
    mids = torch.zeros(len(batch), n_w, dtype=torch.long)
    tmask = torch.zeros(len(batch), n_w)
    for i, b in enumerate(batch):
        k = min(b.tokens.shape[0], max_tokens)
        toks[i, :k] = b.tokens[:k]
        mids[i, :k] = b.module_ids[:k]
        tmask[i, :k] = 1.0

    texts = [(b.question + prompt + b.target + tokenizer.eos_token) for b in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=384)
    B, L = enc["input_ids"].shape
    pre_ids = torch.full((B, n_w), placeholder_id, dtype=enc["input_ids"].dtype)
    pre_att = torch.ones((B, n_w), dtype=enc["attention_mask"].dtype)

    input_ids = torch.cat([pre_ids, enc["input_ids"]], dim=1)
    attn = torch.cat([pre_att, enc["attention_mask"]], dim=1)
    labels = torch.cat([torch.full((B, n_w), -100, dtype=enc["input_ids"].dtype),
                        enc["input_ids"].clone()], dim=1)
    labels[attn == 0] = -100
    labels[:, :n_w] = -100          # never train the model to predict its own placeholders
    return (toks.to(device), mids.to(device), tmask.to(device),
            input_ids.to(device), attn.to(device), labels.to(device))


def placeholder_prefix(tokenizer, n_w: int, placeholder_id: int, prompt: str, device="cpu"):
    """Same construction for generation: exactly n_w placeholder positions, then the question."""
    enc = tokenizer(prompt, return_tensors="pt")
    B = enc["input_ids"].shape[0]
    ids = torch.cat([torch.full((B, n_w), placeholder_id, dtype=enc["input_ids"].dtype),
                     enc["input_ids"]], dim=1)
    att = torch.cat([torch.ones((B, n_w), dtype=enc["attention_mask"].dtype),
                     enc["attention_mask"]], dim=1)
    return ids.to(device), att.to(device)
