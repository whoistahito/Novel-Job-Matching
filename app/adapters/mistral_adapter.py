from __future__ import annotations

from app.adapters.base_hf_causal import BaseHFCausalAdapter


class MistralAdapter(BaseHFCausalAdapter):
    """Adapter for the Mistral Small 24B model."""

    # Hugging Face repo id
    model_id = "mistralai/Mistral-Nemo-Instruct-2407"

    # Sampling defaults per plan
    max_new_tokens: int = 1000
    temperature: float = 0.6
    do_sample: bool = True
    top_p: float | None = 0.95

    def name(self) -> str:
        return "mistral-small-24b"

    def _generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        """Custom message formatting for Mistral model using apply_chat_template."""
        import torch

        tok = self._tokenizer
        mdl = self._model

        # Apply Mistral chat template as done in the model script
        model_inputs = tok.apply_chat_template(messages, return_tensors="pt")

        # Move to the correct device
        device = next(mdl.parameters()).device
        model_inputs = model_inputs.to(device)

        # Generate response with Mistral-specific parameters
        with torch.inference_mode():
            output = mdl.generate(
                model_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=tok.eos_token_id,  # Use EOS token as pad token
            )

        # Get only the new tokens
        new_tokens = output[0][model_inputs.shape[1]:]
        response = tok.decode(new_tokens, skip_special_tokens=True)
        return response

    def setup_tokenizer_model(self) -> None:
        tok = self._tokenizer
        mdl = self._model
        if tok is not None and getattr(tok, "pad_token", None) is None:
            eos = getattr(tok, "eos_token", None)
            if eos is not None:
                tok.pad_token = eos
                if mdl is not None and hasattr(mdl, "config"):
                    mdl.config.pad_token_id = getattr(mdl.config, "eos_token_id", None)
