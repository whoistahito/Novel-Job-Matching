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

    # Context length configuration
    context_length: int = 32768  # Default context length
    enable_thinking: bool = True  # Enable thinking mode by default

    def name(self) -> str:
        return "qwen3-8b"

    def setup_tokenizer_model(self) -> None:
        """Setup tokenizer and model with context length configuration."""
        # Configure rope scaling if context length exceeds default
        if self.context_length > 32768:
            if self.context_length > 131072:
                print(
                    f"Warning: Requested context length {self.context_length} exceeds maximum supported (131072). Capping at 131072.")
                self.context_length = 131072

            # Calculate appropriate YaRN factor
            factor = self.context_length / 32768
            rope_scaling = {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": 32768
            }

            print(f"Enabling YaRN scaling with factor {factor} for context length {self.context_length}")

            # Apply rope scaling to model config if model is already loaded
            if self._model is not None and hasattr(self._model, "config"):
                self._model.config.rope_scaling = rope_scaling
                # Update max_position_embeddings
                self._model.config.max_position_embeddings = self.context_length

    async def predict(self, data, params=None):
        """Override predict to handle context length configuration during model loading."""
        # Handle context_length parameter if provided
        if params and isinstance(params, dict):
            if "context_length" in params:
                self.context_length = min(params["context_length"], 131072)
            if "enable_thinking" in params:
                self.enable_thinking = params["enable_thinking"]

        # Call parent predict which will trigger model loading and setup
        return await super().predict(data, params)

    def _get_model_kwargs(self) -> dict:
        """Get model loading kwargs with rope scaling configuration."""
        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": "auto",  # Use auto instead of explicit float16 for Qwen3
        }

        # Add rope_scaling if context length exceeds default
        if self.context_length > 32768:
            factor = self.context_length / 32768
            rope_scaling = {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": 32768
            }
            model_kwargs["rope_scaling"] = rope_scaling
            print(f"Configuring YaRN scaling with factor {factor} for context length {self.context_length}")

        return model_kwargs

    def _generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        """Custom message formatting for Qwen3 model with thinking mode."""
        import torch

        tok = self._tokenizer
        mdl = self._model

        # Apply chat template for Qwen3 with configurable thinking mode
        text = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking  # Use configurable thinking mode
        )

        # Tokenize the text
        model_inputs = tok([text], return_tensors="pt")

        # Create explicit attention mask (1s for all tokens)
        input_ids_length = model_inputs["input_ids"].shape[1]
        attention_mask = torch.ones((1, input_ids_length), dtype=torch.long)
        model_inputs["attention_mask"] = attention_mask

        # Check if input exceeds context length
        if input_ids_length > self.context_length:
            print(f"Warning: Input length {input_ids_length} exceeds configured context length {self.context_length}")

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

        # Handle thinking mode output parsing
        if self.enable_thinking:
            try:
                # Look for </think> token (ID 151668)
                think_end_index = len(output_ids)
                for i in range(len(output_ids) - 1, -1, -1):
                    if output_ids[i] == 151668:  # </think> token
                        think_end_index = i
                        break

                thinking_content = tok.decode(output_ids[:think_end_index], skip_special_tokens=True).strip("\n")
                response = tok.decode(output_ids[think_end_index:], skip_special_tokens=True).strip("\n")

                # Optional: print thinking content for debugging
                print(f"Thinking process: {thinking_content[:200]}..." if len(
                    thinking_content) > 200 else thinking_content)

                return response
            except Exception as e:
                print(f"Error parsing thinking output: {e}")
                # Fallback to full response
                return tok.decode(output_ids, skip_special_tokens=True)
        else:
            # Decode the response normally
            response = tok.decode(output_ids, skip_special_tokens=True)
            return response

    def postprocess_text(self, raw: str) -> str:
        return raw.replace("</think>", "").strip()
