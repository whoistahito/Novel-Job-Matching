from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.contracts import ModelAdapter


class QwenInput(BaseModel):
    markdown: str
    chunk_size: int = 12000


class QwenOutput(BaseModel):
    requirements: list[Any] = Field(default_factory=list)


class QwenAdapter(ModelAdapter):
    input_model = QwenInput
    output_model = QwenOutput

    def name(self) -> str:
        return "qwen3-8b"

    async def predict(self, data: QwenInput, params: Optional[dict[str, Any]] = None) -> QwenOutput:
        raise NotImplementedError("Qwen3-8B adapter not wired to model yet")
