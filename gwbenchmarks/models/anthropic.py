"""Claude model interface via the Anthropic API."""

import time

from gwbenchmarks.models.base import LLMModel, LLMResponse


class ClaudeModel(LLMModel):
    """Interface to Claude models through the Anthropic SDK.

    Requires the ``anthropic`` package and ANTHROPIC_API_KEY env var.
    """

    def __init__(self, model_id: str, name: str | None = None, **kwargs):
        super().__init__(model_id, name, **kwargs)
        import anthropic

        self._client = anthropic.Anthropic(**kwargs)

    def query(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        t0 = time.perf_counter()
        response = self._client.messages.create(**kwargs)
        latency = time.perf_counter() - t0

        return LLMResponse(
            text=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            latency=latency,
        )
