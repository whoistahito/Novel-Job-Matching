from __future__ import annotations

import logging
from typing import Any, Optional

from app.adapters.base_hf_causal import (
    BaseHFCausalAdapter,
    RequirementsInput,
    RequirementsOutput,
)
from app.utils.text_utils import extract_json_payload, dedupe_stable

logger = logging.getLogger(__name__)


class GLM4Adapter(BaseHFCausalAdapter):
    """Adapter for the GLM-4-9B model."""

    # Hugging Face repo id for the model
    model_id = "THUDM/GLM-4-9B-0414"

    # Generation defaults tuned for deterministic extraction
    max_new_tokens: int = 1000
    temperature: float = 0.1
    do_sample: bool = False

    def name(self) -> str:
        # Public API model id used in /models and requests
        return "glm4-9b"

    async def predict(self, data: RequirementsInput, params: Optional[dict[str, Any]] = None) -> RequirementsOutput:
        # Fast path for tests and offline parsing: allow providing a mock raw text
        if params and isinstance(params, dict) and params.get("mock_text"):
            text = str(params["mock_text"])  # raw model-like output
            parsed, _raw = extract_json_payload(text)
            all_reqs: list[Any] = []
            if isinstance(parsed, dict):
                if "requirements" in parsed and isinstance(parsed["requirements"], list):
                    all_reqs.extend(parsed["requirements"])
                else:
                    # Flatten common categories
                    for key in ("skills", "experience", "qualifications", "education"):
                        vals = parsed.get(key)
                        if isinstance(vals, list):
                            all_reqs.extend(vals)
            elif isinstance(parsed, list):
                all_reqs.extend(parsed)
            return RequirementsOutput(requirements=dedupe_stable(all_reqs))

        # Otherwise, run the standard HF pipeline
        return await super().predict(data, params)
