"""OpenAI model interface (GPT-5.3, etc.)."""

import time

from gwbenchmarks.models.base import LLMModel, LLMResponse


class OpenAIModel(LLMModel):
    """Interface to OpenAI models through the OpenAI SDK.

    Requires the ``openai`` package and OPENAI_API_KEY env var.
    """

    def __init__(self, model_id: str, name: str | None = None, **kwargs):
        super().__init__(model_id, name, **kwargs)
        import openai

        self._client = openai.OpenAI(**kwargs)

    def query(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency = time.perf_counter() - t0

        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content,
            model=response.model,
            usage={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
            },
            latency=latency,
        )
