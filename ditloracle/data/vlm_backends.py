"""VLM backends for benign-field drafting (POC-1b) and NSFW triage.

Pluggable so the drafter LOGIC (prompt, JSON parse, schema validation, multi-image batching, resumable
harness) is fully testable locally with `MockBackend`, while real inference runs on the cluster where
the GPUs + weights live. All backends take ONLY images + a text prompt — never adapter metadata
(blindness is structural, design doc §B.8.1).

Backends:
  * MockBackend          — deterministic fake JSON for tests/dry-runs (no model).
  * QwenVLBackend        — Qwen2.5-VL via transformers (open-weight, Apache-2.0; the locked pre-labeler
                           family — DIFFERENT from the eval scorer family OpenCLIP/DINO + GPT/Gemini).

QwenVLBackend lazy-imports transformers/PIL/qwen_vl_utils so importing this module never requires them
(the laptop venv has none of these); a clear error fires only if you actually instantiate it without
the deps. Recommended run host: a cluster GPU node (see WORKING_NORMS / scripts header).
"""

from __future__ import annotations

import json
import re
from typing import Protocol


class VLMBackend(Protocol):
    model_id: str
    def generate_json(self, image_paths: list[str], prompt: str) -> dict: ...


# ----------------------------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Pull the first balanced {...} JSON object out of a model's text response. Robust to code
    fences and pre/post chatter. Returns {} if none parses (caller falls back to empty schema)."""
    if not text:
        return {}
    # strip ```json fences
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start < 0:
        return {}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    # tolerate trailing commas
                    blob = re.sub(r",\s*([}\]])", r"\1", text[start:i + 1])
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        return {}
    return {}


# ----------------------------------------------------------------------------------------------
class MockBackend:
    """Deterministic fake — for tests and no-GPU dry runs. Returns a fixed plausible draft so the
    whole pipeline (validate → write → human tool) can be exercised without a model."""

    model_id = "mock-vlm-v0"

    def __init__(self, payload: dict | None = None):
        self._payload = payload or {
            "primary_concept": "a mock concept", "subject_type": "person",
            "medium": "photograph", "realism_level": "photoreal", "adapter_function": "identity_character",
            "ai_generated_look": "subtly_off", "caption": "a mock sample caption",
        }

    def generate_json(self, image_paths: list[str], prompt: str) -> dict:
        assert image_paths, "backend must receive at least one image"
        return dict(self._payload)


# ----------------------------------------------------------------------------------------------
class QwenVLBackend:
    """Qwen2.5-VL (default 7B-Instruct) via transformers. Multi-image, JSON-constrained prompt.

    Lazy-imports heavy deps at construction. Intended to run on a GPU node. Install on the box:
        pip install "transformers>=4.49" qwen-vl-utils accelerate pillow
    """

    def __init__(self, model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 max_new_tokens: int = 768, device_map: str = "auto", max_images: int = 4,
                 quantization: str | None = None):
        """quantization: None (native dtype, needs ~16GB+ for 7B), '4bit' or '8bit' (bitsandbytes).
        '4bit' is REQUIRED on a 16GB T4 (WORKING_NORMS §1b) — fp16 7B weights alone fill the card.
        max_images=4 (down from 12) to fit in 16GB VRAM with 4-bit weights + KV cache."""
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.max_images = max_images
        try:
            import torch  # noqa
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except Exception as e:  # pragma: no cover - only on a box without the deps
            raise RuntimeError(
                "QwenVLBackend needs transformers>=4.49 + qwen-vl-utils + accelerate + pillow on a GPU "
                f"box. Import failed: {e}. Use MockBackend for local/dry runs."
            ) from e
        self._torch = torch
        kw: dict = {"device_map": device_map}
        if quantization in ("4bit", "8bit"):
            from transformers import BitsAndBytesConfig
            # fp16 compute (not bf16): the T4 has no bf16 support
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=(quantization == "4bit"), load_in_8bit=(quantization == "8bit"),
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        else:
            kw["torch_dtype"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kw)
        # Cap vision token count: on a 16GB T4, 4-bit model uses ~6GB; KV cache + vision embeddings
        # must fit in the remaining ~10GB. 256*28*28 per image; with max_images=2 that's safe.
        proc_kw = {}
        if quantization in ("4bit", "8bit"):
            proc_kw = {"min_pixels": 128 * 28 * 28, "max_pixels": 256 * 28 * 28}
            self.max_images = min(self.max_images, 2)  # hard cap for 16GB cards
        self.processor = AutoProcessor.from_pretrained(model_id, **proc_kw)
        torch.cuda.empty_cache()

    def generate_json(self, image_paths: list[str], prompt: str) -> dict:  # pragma: no cover - GPU only
        from qwen_vl_utils import process_vision_info
        _IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}
        imgs = [p for p in image_paths if any(p.lower().endswith(e) for e in _IMG_EXT)]
        imgs = imgs[: self.max_images]
        if not imgs:
            return {}
        content = [{"type": "image", "image": f"file://{p}"} for p in imgs]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        resp = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)[0]
        return _extract_json(resp)


def get_backend(name: str = "mock", **kw) -> VLMBackend:
    """Factory. name='mock' (default, no deps) or 'qwen' (Qwen2.5-VL, needs GPU box)."""
    if name == "mock":
        return MockBackend(**kw)
    if name == "qwen":
        return QwenVLBackend(**kw)
    raise ValueError(f"unknown VLM backend: {name}")
