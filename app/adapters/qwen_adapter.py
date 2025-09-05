from __future__ import annotations

from app.adapters.base_hf_causal import BaseHFCausalAdapter


class QwenAdapter(BaseHFCausalAdapter):
    """Adapter for the Qwen3-8B model."""

    # Hugging Face repo id
    model_id = "Qwen/Qwen3-8B"

    # Sampling defaults per plan
    max_new_tokens: int = 1000
    temperature: float = 0.6
    do_sample: bool = True
    top_p: float | None = 0.95

    def name(self) -> str:
        return "qwen3-8b"

    def postprocess_text(self, raw: str) -> str:
        # Strip model-specific scaffolding before JSON extraction
        return raw.replace("</think>", "").strip()
