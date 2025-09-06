from __future__ import annotations

import re
from typing import Any, Optional

from app.adapters.base_hf_causal import BaseHFCausalAdapter, RequirementsInput, RequirementsOutput, RequirementsData
from app.utils.text_utils import extract_json_payload, dedupe_stable


class GLM4Z1Adapter(BaseHFCausalAdapter):
    model_id = "THUDM/GLM-Z1-9B-0414"

    max_new_tokens: int = 1000
    temperature: float = 0.1
    do_sample: bool = False

    def name(self) -> str:
        return "glm4-z1-9b"

    def _generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        """Custom message formatting for GLM-4-Z1 model with thinking mode."""
        import torch

        tok = self._tokenizer
        mdl = self._model

        # Format messages directly for GLM-4 as done in the model script
        chat_text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                chat_text += f"{content}\n\n"
            elif role == "user":
                chat_text += f"<|user|>\n{content}<|endoftext|>\n"
            elif role == "assistant":
                chat_text += f"<|assistant|>\n{content}<|endoftext|>\n"

        # Add final assistant prompt with thinking mode
        chat_text += "<|assistant|>\n<think>\n"

        # Tokenize the text
        inputs = tok(chat_text, return_tensors="pt")

        # Move to the correct device
        device = next(mdl.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate response with GLM-4-Z1 specific parameters
        with torch.inference_mode():
            output = mdl.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample
            )

        # Decode the response
        full_output = tok.decode(output[0], skip_special_tokens=True)

        # Extract the model's response (everything after the last <|assistant|>)
        response = full_output.split("<|assistant|>\n")[-1].strip()
        return response

    def postprocess_text(self, raw: str) -> str:
        # Strip optional <think>... markers and content, keep what's after
        # Remove any <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Also remove any leading </think> artifacts
        text = re.sub(r"^\s*</think>\s*", "", text, flags=re.IGNORECASE)
        return text

    async def predict(self, data: RequirementsInput, params: Optional[dict[str, Any]] = None) -> RequirementsOutput:
        # Provide a mock parsing path for unit tests
        if params and isinstance(params, dict) and params.get("mock_text"):
            text = str(params["mock_text"])  # raw model-like output
            text = self.postprocess_text(text)
            parsed, _raw = extract_json_payload(text)

            skills: list[Any] = []
            experience: list[Any] = []
            qualifications: list[Any] = []

            if isinstance(parsed, dict):
                s = parsed.get("skills")
                if isinstance(s, list):
                    skills.extend(s)
                e = parsed.get("experience")
                if isinstance(e, list):
                    experience.extend(e)
                q = parsed.get("qualifications")
                if isinstance(q, list):
                    qualifications.extend(q)
                # Aliases
                edu = parsed.get("education")
                if isinstance(edu, list):
                    qualifications.extend(edu)
                q1 = parsed.get("qualification")
                if isinstance(q1, list):
                    qualifications.extend(q1)
            elif isinstance(parsed, list):
                # Legacy
                skills.extend(parsed)

            # Dedupe
            skills = dedupe_stable(skills)
            experience = dedupe_stable(experience)
            qualifications = dedupe_stable(qualifications)

            return RequirementsOutput(
                requirements=RequirementsData(
                    skills=skills,
                    experience=experience,
                    qualifications=qualifications,
                )
            )

        # Otherwise, run the standard HF pipeline
        return await super().predict(data, params)
