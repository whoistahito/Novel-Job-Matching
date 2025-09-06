from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional, Type

from pydantic import BaseModel


class InferenceRequest(BaseModel):
    model: str
    input: dict[str, Any]
    params: Optional[dict[str, Any]] = None
    stream: bool = False


class InferenceResponse(BaseModel):
    model: str
    output: Any
    usage: Optional[dict[str, Any]] = None


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class ModelAdapter(ABC):
    """Abstract base for model adapters."""

    @abstractmethod
    def name(self) -> str:  # unique model id
        raise NotImplementedError

    input_model: Type[BaseModel]
    output_model: Type[BaseModel]

    @abstractmethod
    async def predict(self, data: BaseModel, params: Optional[dict[str, Any]] = None) -> BaseModel:
        raise NotImplementedError

    async def predict_stream(
            self, data: BaseModel, params: Optional[dict[str, Any]] = None
    ) -> AsyncIterator[BaseModel]:
        raise NotImplementedError("Streaming not implemented for this adapter")
