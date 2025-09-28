from collections import defaultdict
import gc
import os

from outlines import Template, from_transformers, Generator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.api_schema import Requirements
from app.utils import chunk_markdown

MODELS_CONFIG = {
    "glm4-9b": {
        "model_id": "THUDM/GLM-4-9B-0414",
        "chunk_size": "12000",
        "device_kwargs": {
            "device_map": "balanced",
            "max_memory": {0: "10GiB", 1: "10GiB"},
            "dtype": torch.bfloat16,
        },
    },
    "llama3.1-nemotron-8b": {
        "model_id": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1",
        "chunk_size": "12000",
        "device_kwargs": {
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
    },
    "mistral-nemo": {
        "model_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "chunk_size": "12000",
        "device_kwargs": {
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
    },
    "qwen3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "chunk_size": "12000",
        "device_kwargs": {
            "device_map": "auto",
            "dtype": torch.bfloat16,
        }
    },

}

_MODEL_INSTANCES = {}


class LLMExtractor:
    def __init__(self, model_id, chunk_size, device_kwargs=None):
        self.model_id = model_id
        self.chunk_size = chunk_size
        self.device_kwargs = device_kwargs or {}
        self.generator = None
        self._load_model()

    def _load_model(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **self.device_kwargs,
            trust_remote_code=True,
        )
        outlines_model = from_transformers(self.model, self.tokenizer)
        self.generator = Generator(outlines_model, Requirements)

    def process_text(self, text):
        chunks = chunk_markdown(text, self.chunk_size)
        all_requirements = []
        for chunk in chunks:
            req = self.process_chunk(chunk)
            all_requirements.append(req)
        merged = defaultdict(set)
        for req in all_requirements:
            merged["skills"].update(req.skills)
            merged["experiences"].update(req.experiences)
            merged["qualifications"].update(req.qualifications)
        unique_requirements = {k: sorted(list(v)) for k, v in merged.items()}
        return Requirements(**unique_requirements)

    def process_chunk(self, chunk):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(current_dir, "prompt_template.txt")
        template = Template.from_file(template_path)
        prompt = template(chunk=chunk)
        try:
            response = self.generator(prompt, max_new_tokens=200)
            return response
        except Exception as e:
            print(f"Error during generation: {e}")
            return Requirements()


def get_extractor_for(model_key):
    if model_key not in MODELS_CONFIG:
        raise ValueError(f"Unknown model_id: {model_key}")
    if model_key not in _MODEL_INSTANCES:
        config = MODELS_CONFIG[model_key]
        _MODEL_INSTANCES[model_key] = LLMExtractor(
            model_id=config["model_id"],
            chunk_size=config["chunk_size"],
            device_kwargs=config.get("device_kwargs", {}),
        )
    return _MODEL_INSTANCES[model_key]
