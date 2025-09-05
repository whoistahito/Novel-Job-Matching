from __future__ import annotations

import re
from typing import Any, Optional

from app.adapters.base_hf_causal import BaseHFCausalAdapter, RequirementsInput, RequirementsOutput
from app.utils.text_utils import extract_json_payload, dedupe_stable


class GLM4Z1Adapter(BaseHFCausalAdapter):
    model_id = "THUDM/GLM-Z1-9B-0414"

    max_new_tokens: int = 1000
    temperature: float = 0.1
    do_sample: bool = False

    def name(self) -> str:
        return "glm4-z1-9b"

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
            all_reqs: list[Any] = []
            if isinstance(parsed, dict):
                if "requirements" in parsed and isinstance(parsed["requirements"], list):
                    all_reqs.extend(parsed["requirements"])
                else:
                    for key in ("skills", "experience", "qualifications", "education"):
                        vals = parsed.get(key)
                        if isinstance(vals, list):
                            all_reqs.extend(vals)
            elif isinstance(parsed, list):
                all_reqs.extend(parsed)
            return RequirementsOutput(requirements=dedupe_stable(all_reqs))

        # Otherwise, run the standard HF pipeline
        return await super().predict(data, params)
