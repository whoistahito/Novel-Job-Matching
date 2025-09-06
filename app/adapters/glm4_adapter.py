from __future__ import annotations

import logging
from typing import Any, Optional

from app.adapters.base_hf_causal import (
    BaseHFCausalAdapter,
    RequirementsInput,
    RequirementsOutput,
    RequirementsData,
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
                edu = parsed.get("education")
                if isinstance(edu, list):
                    qualifications.extend(edu)
                q1 = parsed.get("qualification")
                if isinstance(q1, list):
                    qualifications.extend(q1)
            elif isinstance(parsed, list):
                # Legacy: treat as skills if just a list
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
