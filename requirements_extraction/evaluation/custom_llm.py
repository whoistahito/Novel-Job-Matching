from typing import Optional, Dict

from deepeval.models import DeepEvalBaseLLM  # to call its __init__
from deepeval.models import GPTModel  # <- replace with your actual import


class CustomNvidiaModel(GPTModel):
    """
    A Custom LLM that inherits from GPTModel for Nvidia Api.
    """

    def __init__(
            self,
            model: Optional[str] = "openai/gpt-oss-120b",
            api_key: Optional[str] = None,
            base_url: Optional[str] = "https://integrate.api.nvidia.com/v1",
            temperature: float = 1.0,
            cost_per_input_token: Optional[float] = None,
            cost_per_output_token: Optional[float] = None,
            generation_kwargs: Optional[Dict] = None,
            **kwargs,
    ):
        """
        Args:
            model: NVIDIA model name (OpenAI-compatible)
            api_key: NVIDIA API key
            base_url: NVIDIA integrate endpoint
            temperature: default temperature (NVIDIA examples often use 1.0)
            cost_per_input_token: USD per input token (optional; if None, cost=0.0)
            cost_per_output_token: USD per output token (optional; if None, cost=0.0)
            generation_kwargs: Extra kwargs for client.chat.completions.create (top_p, max_tokens, etc.)
            kwargs: Extra OpenAI client kwargs (e.g., timeout, organization, etc.)
        """
        if temperature < 0:
            raise ValueError("Temperature must be >= 0.")

        self._openai_api_key = api_key or kwargs.pop("_openai_api_key", None)
        self.base_url = base_url
        self.temperature = temperature
        self.generation_kwargs = generation_kwargs or {}
        self.kwargs = kwargs

        self._cost_in = float(cost_per_input_token) if cost_per_input_token is not None else None
        self._cost_out = float(cost_per_output_token) if cost_per_output_token is not None else None

        DeepEvalBaseLLM.__init__(self, model)

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        if self._cost_in is None or self._cost_out is None:
            return 0.0
        return (input_tokens * self._cost_in) + (output_tokens * self._cost_out)
