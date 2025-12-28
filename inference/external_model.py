from collections import defaultdict
import os

from openai import OpenAI
import outlines
from outlines import Template

from .api_schema import Requirements
from .utils import chunk_markdown


EXTERNAL_MODELS_CONFIG = {
    "gpt-oss-120b": {
        "model_name": "openai/gpt-oss-120b",
        "chunk_size": 12000,
    },
    "qwen3-next-80b-thinking": {
        "model_name": "qwen/qwen3-next-80b-a3b-thinking",
        "chunk_size": 12000,
    },
}

_MODEL_INSTANCES = {}


class ExternalLLMExtractor:
    def __init__(self, model_name, chunk_size, api_base_url=None):
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.api_base_url = api_base_url or "https://integrate.api.nvidia.com/v1"
        self.api_key = os.environ["EXTERNAL_LLM_API_KEY"]

        if not self.api_key:
            raise RuntimeError("EXTERNAL_LLM_API_KEY environment variable is not set.")

        self.client = None
        self.generator = None
        self._load_model()

    def _load_model(self):
        """Initialize the OpenAI client and outlines generator."""
        self.client = OpenAI(base_url=self.api_base_url, api_key=self.api_key)
        self.generator = outlines.from_openai(self.client, self.model_name)

    def process_text(self, text):
        """Process the entire text by chunking and merging results."""
        chunks = chunk_markdown(text, self.chunk_size)
        all_requirements = []
        for chunk in chunks:
            req = self.process_chunk(chunk)
            all_requirements.append(req)

        # Merge all requirements from chunks
        merged = defaultdict(set)
        for req in all_requirements:
            merged["skills"].update(req.skills)
            merged["experiences"].update(req.experiences)
            merged["qualifications"].update(req.qualifications)

        unique_requirements = {k: sorted(list(v)) for k, v in merged.items()}
        return Requirements(**unique_requirements)

    def process_chunk(self, chunk) -> Requirements:
        """Process a single chunk using the template and external API."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "prompt_template.txt")
        template = Template.from_file(template_path)
        prompt = template(chunk=chunk)

        try:
            response = self.generator(
                prompt,
                Requirements,
                temperature=0.5,
            )
            # Clean up response
            response = response.replace("\n", "")
            response = response.replace("<|return|>", "")
            response = response.replace("```json", "")
            response = response.replace("```", "")
            return Requirements.model_validate_json(response)
        except Exception as e:
            print(f"Error during generation: {e}")
            return Requirements()


def get_extractor_for(model_key):
    """Get or create an extractor instance for the specified model."""
    if model_key not in EXTERNAL_MODELS_CONFIG:
        raise ValueError(f"Unknown model_key: {model_key}")

    if model_key not in _MODEL_INSTANCES:
        config = EXTERNAL_MODELS_CONFIG[model_key]
        _MODEL_INSTANCES[model_key] = ExternalLLMExtractor(
            model_name=config["model_name"],
            chunk_size=config["chunk_size"],
        )

    return _MODEL_INSTANCES[model_key]
