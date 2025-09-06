from __future__ import annotations

from app.adapters.base_hf_causal import BaseHFCausalAdapter


class QwenAdapter(BaseHFCausalAdapter):
    """Adapter for the Qwen3-8B model."""

    # Hugging Face repo id
    model_id = "Qwen/Qwen3-8B"

    # Sampling defaults per plan - updated to match model script
    max_new_tokens: int = 2048
    temperature: float = 0.6
    do_sample: bool = True
    top_p: float | None = 0.95
    top_k: int | None = 20

    def name(self) -> str:
        return "qwen3-8b"

    def _generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        """Custom message formatting for Qwen3 model with thinking mode."""
        import torch

        tok = self._tokenizer
        mdl = self._model

        # Apply chat template for Qwen3 with thinking mode enabled
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # Enable thinking mode for better reasoning
        )

        # Tokenize the text
        model_inputs = tok([text], return_tensors="pt")

        # Create explicit attention mask (1s for all tokens)
        input_ids_length = model_inputs["input_ids"].shape[1]
        attention_mask = torch.ones((1, input_ids_length), dtype=torch.long)
        model_inputs["attention_mask"] = attention_mask

        # Move to the correct device
        device = next(mdl.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

        # Generate response with recommended parameters for thinking mode
        with torch.inference_mode():
            generated_ids = mdl.generate(
                model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,  # Must use sampling, not greedy decoding
            )

        # Get only the new tokens
        output_ids = generated_ids[0][model_inputs["input_ids"].shape[1]:].tolist()

        # Decode the response
        response = tok.decode(output_ids, skip_special_tokens=True)
        return response

    def postprocess_text(self, raw: str) -> str:
        return raw.replace("</think>", "").strip()
