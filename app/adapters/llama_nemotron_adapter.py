from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.contracts import ModelAdapter


class LlamaNemotronInput(BaseModel):
    markdown: str
    chunk_size: int = 12000


class LlamaNemotronOutput(BaseModel):
    requirements: list[Any] = Field(default_factory=list)


class LlamaNemotronAdapter(ModelAdapter):
    input_model = LlamaNemotronInput
    output_model = LlamaNemotronOutput

    def name(self) -> str:
        return "llama3.1-nemotron-8b"

    async def predict(self, data: LlamaNemotronInput, params: Optional[dict[str, Any]] = None) -> LlamaNemotronOutput:
        raise NotImplementedError("Llama3.1-Nemotron-8B adapter not wired to model yet")
