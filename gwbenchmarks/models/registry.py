"""Model registry: canonical names → model configs."""

from gwbenchmarks.models.anthropic import ClaudeModel
from gwbenchmarks.models.openai import OpenAIModel
from gwbenchmarks.models.gemini import GeminiModel
from gwbenchmarks.models.opencode import OpenCodeModel
from gwbenchmarks.models.base import LLMModel

MODELS = {
    "claude-opus-4.6": {
        "class": ClaudeModel,
        "model_id": "claude-opus-4-6",
        "name": "Claude Opus 4.6",
    },
    "claude-opus-4.7": {
        "class": ClaudeModel,
        "model_id": "claude-opus-4-7",
        "name": "Claude Opus 4.7",
    },
    "claude-sonnet-4.6": {
        "class": ClaudeModel,
        "model_id": "claude-sonnet-4-6",
        "name": "Claude Sonnet 4.6",
    },
    "claude-haiku-4.5": {
        "class": ClaudeModel,
        "model_id": "claude-haiku-4-5-20251001",
        "name": "Claude Haiku 4.5",
    },
    "gpt-5.3": {
        "class": OpenAIModel,
        "model_id": "gpt-5.3",
        "name": "GPT-5.3",
    },
    "gpt-5.5-high": {
        "class": OpenAIModel,
        "model_id": "gpt-5.5",
        "name": "GPT-5.5 High",
    },
    "gpt-5.4-mini": {
        "class": OpenAIModel,
        "model_id": "gpt-5.4-mini",
        "name": "GPT-5.4 Mini",
    },
    "gpt-5.3-codex-high": {
        "class": OpenAIModel,
        "model_id": "gpt-5.3-codex",
        "name": "GPT-5.3 Codex High",
    },
    "gpt-5.2": {
        "class": OpenAIModel,
        "model_id": "gpt-5.2",
        "name": "GPT-5.2",
    },
    "gemini-3.1-pro-preview": {
        "class": GeminiModel,
        "model_id": "gemini-3.1-pro-preview",
        "name": "Gemini 3.1 Pro Preview",
    },
    "gemini-2.5-pro": {
        "class": GeminiModel,
        "model_id": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
    },
    "gemini-3-flash-preview": {
        "class": GeminiModel,
        "model_id": "gemini-3-flash-preview",
        "name": "Gemini 3 Flash Preview",
    },
    "gemini-2.5-flash": {
        "class": GeminiModel,
        "model_id": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
    },
}


def get_model(name: str, **kwargs) -> LLMModel:
    """Instantiate a model by its registry name.

    Parameters
    ----------
    name : str
        Key in MODELS registry (e.g. "claude-opus-4.7", "gpt-5.3").
    **kwargs
        Extra arguments passed to the model constructor
        (e.g. api_key, base_url).

    Returns
    -------
    LLMModel
    """
    if name not in MODELS:
        available = ", ".join(sorted(MODELS.keys()))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")

    entry = MODELS[name]
    cls = entry["class"]
    return cls(model_id=entry["model_id"], name=entry["name"], **kwargs)
