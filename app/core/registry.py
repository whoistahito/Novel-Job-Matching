from __future__ import annotations

from typing import Dict, List

from .contracts import ModelAdapter


class ModelRegistry:
    def __init__(self) -> None:
        self._adapters: Dict[str, ModelAdapter] = {}

    def register(self, adapter: ModelAdapter) -> None:
        self._adapters[adapter.name()] = adapter

    def get(self, model_id: str) -> ModelAdapter:
        adapter = self._adapters.get(model_id)
        if not adapter:
            raise KeyError(model_id)
        return adapter

    def list(self) -> List[str]:
        return sorted(self._adapters.keys())


registry = ModelRegistry()
