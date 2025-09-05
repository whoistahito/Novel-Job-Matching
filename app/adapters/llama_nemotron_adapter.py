from __future__ import annotations

from app.adapters.base_hf_causal import BaseHFCausalAdapter


class LlamaNemotronAdapter(BaseHFCausalAdapter):
    """Adapter for the Llama 3.1 Nemotron 8B model."""

    # Hugging Face repo id
    model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"

    # Sampling defaults per plan
    max_new_tokens: int = 1000
    temperature: float = 0.6
    do_sample: bool = True
    top_p: float | None = 0.95

    def name(self) -> str:
        return "llama3.1-nemotron-8b"

    def setup_tokenizer_model(self) -> None:
        # Set a safe pad token if the tokenizer lacks one
        tok = self._tokenizer
        mdl = self._model
        if tok is not None and getattr(tok, "pad_token", None) is None:
            eos = getattr(tok, "eos_token", None)
            if eos is not None:
                tok.pad_token = eos
                if mdl is not None and hasattr(mdl, "config"):
                    mdl.config.pad_token_id = getattr(mdl.config, "eos_token_id", None)
