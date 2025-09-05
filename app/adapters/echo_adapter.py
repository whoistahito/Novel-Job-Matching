from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel

from app.core.contracts import ModelAdapter


class EchoInput(BaseModel):
    text: str
    extra: Optional[dict[str, Any]] = None


class EchoOutput(BaseModel):
    text: str
    extra: Optional[dict[str, Any]] = None


class EchoAdapter(ModelAdapter):
    input_model = EchoInput
    output_model = EchoOutput

    def name(self) -> str:
        return "echo"

    async def predict(self, data: EchoInput, params: Optional[dict[str, Any]] = None) -> EchoOutput:
        return EchoOutput(text=data.text, extra=data.extra)
