import os
import sys
from pathlib import Path

import pytest

# Ensure imports like `import utils` resolve to `implementation/job_matching_system/utils.py`
# when running `uv run pytest` from this directory.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _THIS_DIR.parent
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))


@pytest.fixture(autouse=True)
def _clear_model_caches():
    """Ensure module-level caches don’t leak between tests."""
    # Import from the local modules under implementation/job_matching_system.
    import base_model, external_model, similarity_search

    base_model._MODEL_INSTANCES.clear()
    external_model._MODEL_INSTANCES.clear()
    similarity_search._model = None


@pytest.fixture
def dummy_requirements():
    from api_schema import Requirements

    return Requirements(skills=[], experiences=[], qualifications=[])
