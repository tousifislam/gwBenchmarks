"""Gemini model interface via the Google GenAI API."""

import time

from gwbenchmarks.models.base import LLMModel, LLMResponse


class GeminiModel(LLMModel):
    """Interface to Gemini models through the Google GenAI SDK.

    Requires the ``google-genai`` package and GEMINI_API_KEY env var.
    """

    def __init__(self, model_id: str, name: str | None = None, **kwargs):
        super().__init__(model_id, name, **kwargs)
        from google import genai

        self._types = genai.types
        self._client = genai.Client(**kwargs)

    def query(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        config = self._types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            system_instruction=system,
        )

        t0 = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )
        latency = time.perf_counter() - t0

        usage = response.usage_metadata

        return LLMResponse(
            text=response.text,
            model=self.model_id,
            usage={
                "input_tokens": usage.prompt_token_count or 0,
                "output_tokens": usage.candidates_token_count or 0,
            },
            latency=latency,
        )
