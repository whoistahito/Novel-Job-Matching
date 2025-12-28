import os
import pytest


@pytest.fixture(autouse=True)
def _clear_model_caches():
    """Ensure module-level caches don’t leak between tests."""
    from inference import base_model, external_model, similarity_search

    base_model._MODEL_INSTANCES.clear()
    external_model._MODEL_INSTANCES.clear()
    similarity_search._model = None


@pytest.fixture
def dummy_requirements():
    from inference.api_schema import Requirements

    return Requirements(skills=[], experiences=[], qualifications=[])
