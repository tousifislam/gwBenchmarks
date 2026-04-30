"""Abstract base class for LLM model interfaces."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class LLMResponse:
    """Raw response from an LLM call."""

    text: str
    model: str
    usage: Dict[str, int]
    latency: float


class LLMModel(ABC):
    """Base interface for LLM models used in benchmark evaluation."""

    name: str
    model_id: str

    def __init__(self, model_id: str, name: str | None = None, **kwargs):
        self.model_id = model_id
        self.name = name or model_id
        self.kwargs = kwargs

    @abstractmethod
    def query(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a prompt to the model and return the response.

        Parameters
        ----------
        prompt : str
            User prompt containing the benchmark task.
        system : str, optional
            System prompt with task instructions.
        max_tokens : int
            Maximum tokens in the response.
        temperature : float
            Sampling temperature (0 = deterministic).

        Returns
        -------
        LLMResponse
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model_id={self.model_id!r})"
