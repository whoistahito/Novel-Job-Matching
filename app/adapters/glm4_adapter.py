from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.contracts import ModelAdapter


class GLM4Input(BaseModel):
    markdown: str
    chunk_size: int = 12000


class GLM4Output(BaseModel):
    requirements: list[Any] = Field(default_factory=list)


class GLM4Adapter(ModelAdapter):
    input_model = GLM4Input
    output_model = GLM4Output

    def name(self) -> str:
        return "glm4-9b"

    async def predict(self, data: GLM4Input, params: Optional[dict[str, Any]] = None) -> GLM4Output:
        raise NotImplementedError("GLM4-9B adapter not wired to model yet")
