from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.contracts import ModelAdapter


class GLM4Z1Input(BaseModel):
    markdown: str
    chunk_size: int = 12000


class GLM4Z1Output(BaseModel):
    requirements: list[Any] = Field(default_factory=list)


class GLM4Z1Adapter(ModelAdapter):
    input_model = GLM4Z1Input
    output_model = GLM4Z1Output

    def name(self) -> str:
        return "glm4-z1-9b"

    async def predict(self, data: GLM4Z1Input, params: Optional[dict[str, Any]] = None) -> GLM4Z1Output:
        raise NotImplementedError("GLM4-Z1-9B adapter not wired to model yet")
