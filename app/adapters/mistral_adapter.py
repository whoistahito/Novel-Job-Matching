from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.contracts import ModelAdapter


class MistralInput(BaseModel):
    markdown: str
    chunk_size: int = 12000


class MistralOutput(BaseModel):
    requirements: list[Any] = Field(default_factory=list)


class MistralAdapter(ModelAdapter):
    input_model = MistralInput
    output_model = MistralOutput

    def name(self) -> str:
        return "mistral-small-24b"

    async def predict(self, data: MistralInput, params: Optional[dict[str, Any]] = None) -> MistralOutput:
        raise NotImplementedError("Mistral-Small-24B adapter not wired to model yet")
