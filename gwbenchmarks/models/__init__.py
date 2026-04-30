"""LLM model interfaces for benchmark evaluation."""

from gwbenchmarks.models.base import LLMModel
from gwbenchmarks.models.anthropic import ClaudeModel
from gwbenchmarks.models.openai import OpenAIModel
from gwbenchmarks.models.registry import MODELS, get_model

__all__ = ["LLMModel", "ClaudeModel", "OpenAIModel", "MODELS", "get_model"]
