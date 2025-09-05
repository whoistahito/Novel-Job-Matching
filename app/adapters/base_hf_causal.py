from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from app.core.contracts import ModelAdapter
from app.utils.text_utils import chunk_markdown, extract_json_payload, dedupe_stable


class RequirementsInput(BaseModel):
    markdown: str
    chunk_size: int = 12000


class RequirementsOutput(BaseModel):
    requirements: list[Any] = Field(default_factory=list)


class BaseHFCausalAdapter(ModelAdapter):
    """Abstracts common HF CausalLM inference: chunking, prompting, generation, and JSON extraction.

    Child classes must set `model_id` and can override `build_messages` and generation params.
    """

    input_model = RequirementsInput
    output_model = RequirementsOutput

    # Child adapters override this with the concrete HF repo id
    model_id: str = ""

    # Generation defaults; adapters can override via subclass attributes
    max_new_tokens: int = 1024
    temperature: float = 0.2
    do_sample: bool = False
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    trust_remote_code: bool = True

    # Internal lazy state
    _tokenizer = None
    _model = None

    def name(self) -> str:
        # By default use the model_id as the public name; child can override
        return self.model_id or self.__class__.__name__.lower()

    # ---------------------- Overridable hooks ----------------------
    def setup_tokenizer_model(self) -> None:
        """Optional hook: customize tokenizer/model after load (e.g., pad_token handling)."""
        # Default: no-op
        return None

    def postprocess_text(self, raw: str) -> str:
        """Optional hook: strip model-specific scaffolding before JSON extraction."""
        return raw

    # ---------------------- Core pipeline ----------------------
    async def predict(
            self, data: RequirementsInput, params: Optional[dict[str, Any]] = None
    ) -> RequirementsOutput:
        # Lazy import to avoid heavy deps unless needed
        try:
            import torch  # noqa: F401
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except Exception as e:
            raise NotImplementedError(f"Transformers not available: {e}")

        # Enforce GPU-only execution
        if not torch.cuda.is_available():
            raise NotImplementedError(
                "A CUDA-enabled GPU is required to run this model. Ensure an NVIDIA GPU and CUDA drivers are available."
            )

        # HF auth token if available (env or local file)
        token = self._hf_token()

        # Lazy load model/tokenizer
        if self._tokenizer is None or self._model is None:
            quant_cfg = self._quantization_config()
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id, trust_remote_code=self.trust_remote_code, token=token
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="cuda",
                    trust_remote_code=self.trust_remote_code,
                    quantization_config=quant_cfg,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    token=token,
                )
            except Exception as e:
                # Provide a helpful message rather than 500
                raise NotImplementedError(
                    "Model loading failed on GPU. Ensure 'accelerate' is installed, a compatible NVIDIA GPU with recent "
                    "CUDA drivers is available, and (optionally) 'bitsandbytes' for 4-bit quantization. "
                    f"Underlying error: {e}"
                )
            # Allow adapters to tweak tokenizer/model (e.g., pad_token)
            self.setup_tokenizer_model()

        chunks = chunk_markdown(data.markdown, data.chunk_size)
        all_reqs: list[Any] = []

        for chunk in chunks:
            messages = self.build_messages(chunk)
            text = self._generate_from_messages(messages)
            text = self.postprocess_text(text)
            parsed, _raw = extract_json_payload(text)
            if isinstance(parsed, dict):
                if "requirements" in parsed and isinstance(parsed["requirements"], list):
                    all_reqs.extend(parsed["requirements"])
                else:
                    # Flatten common categories
                    for key in ("skills", "experience", "qualifications", "education"):
                        vals = parsed.get(key)
                        if isinstance(vals, list):
                            all_reqs.extend(vals)
            elif isinstance(parsed, list):
                all_reqs.extend(parsed)
            # else: skip chunk if unparsable

        return RequirementsOutput(requirements=dedupe_stable(all_reqs))

    # ---------------------- Helpers ----------------------
    def build_messages(self, chunk: str) -> list[dict[str, str]]:
        prompt = (
            "You are an expert job requirements extractor. Analyze the following text and extract ONLY specific, "
            "actionable job requirements.\n\n"
            "RULES:\n"
            "- Extract concrete requirements only (skills, experience years, certifications, education)\n"
            "- Skip: company overview, benefits, culture, responsibilities, \"nice-to-have\" items\n"
            "- Be precise with experience requirements (e.g., \"3+ years Python\" not just \"Python experience\")\n"
            "- Include specific technologies, tools, and methodologies mentioned\n"
            "- Only extract what is explicitly required, not preferred\n\n"
            f"TEXT TO ANALYZE:\n{chunk}\n\n"
            "OUTPUT FORMAT (JSON only, no other text):\n"
            "{\n"
            "  \"skills\": [\"Python programming\", \"AWS cloud services\", \"Docker\"],\n"
            "  \"experience\": [\"3+ years software development\", \"2+ years with microservices\"],\n"
            "  \"qualifications\": [\"Bachelor's degree in Engineering\", \"AWS Solutions Architect certification\"]\n"
            "}\n\n"
            "IMPORTANT: Return ONLY the JSON object above, no explanations or additional text."
        )
        return [
            {
                "role": "system",
                "content": "You are a precise job requirements extraction tool. Output only valid JSON with the exact format requested.",
            },
            {"role": "user", "content": prompt},
        ]

    def _generate_from_messages(self, messages: list[dict[str, str]]) -> str:
        from transformers import PreTrainedTokenizerBase
        import torch

        tok: PreTrainedTokenizerBase = self._tokenizer  # type: ignore
        mdl = self._model  # type: ignore

        text_inputs = None
        apply_chat = getattr(tok, "apply_chat_template", None)
        if callable(apply_chat):
            try:
                text_inputs = tok.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                )
            except TypeError:
                text_inputs = tok.apply_chat_template(messages, return_tensors="pt")
        else:
            # naive fallback
            chat_text = "".join(
                f"<|{m['role']}|>\n{m['content']}<|endoftext|>\n" for m in messages
            ) + "<|assistant|>\n"
            text_inputs = tok(chat_text, return_tensors="pt").input_ids

        device = next(mdl.parameters()).device
        text_inputs = text_inputs.to(device)
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "do_sample": self.do_sample,
        }
        if self.top_p is not None:
            gen_kwargs["top_p"] = self.top_p
        if self.top_k is not None:
            gen_kwargs["top_k"] = self.top_k

        with torch.inference_mode():
            out = mdl.generate(text_inputs, **gen_kwargs)

        # Slice new tokens if using input_ids path
        new_tokens = out[0][text_inputs.shape[1]:] if out.shape[1] > text_inputs.shape[1] else out[0]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def _quantization_config(self):
        try:
            import torch
            # Only attempt 4-bit quantization when CUDA is available; skip on CPU to avoid bitsandbytes issues
            if not torch.cuda.is_available():
                return None
            from transformers import BitsAndBytesConfig

            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            return None

    def _hf_token(self) -> Optional[str]:
        # Prefer environment
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
            val = os.environ.get(key)
            if val:
                return val.strip()
        # Fallback: local file `huggingface-token` in project root
        try:
            root = Path(__file__).resolve().parents[2]
            token_file = root / "huggingface-token"
            if token_file.exists():
                return token_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
        return None
