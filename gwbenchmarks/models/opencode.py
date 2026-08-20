"""OpenCode CLI model interface (Kimi, DeepSeek, etc.)."""

import json
import shutil
import subprocess
import time

from gwbenchmarks.models.base import LLMModel, LLMResponse


class OpenCodeModel(LLMModel):
    """Interface to models served through the ``opencode`` CLI.

    Unlike the API-backed interfaces this shells out to ``opencode run``,
    so it needs the OpenCode CLI on PATH and its provider credentials
    already configured via ``opencode providers``.

    Model ids use the CLI's provider/model form, for example
    ``opencode-go/kimi-k2.6`` or ``opencode-go/deepseek-v4-pro``.

    Note that ``opencode run`` exposes no sampling controls, so the
    ``max_tokens`` and ``temperature`` arguments are accepted for interface
    compatibility but are not forwarded to the CLI.
    """

    def __init__(
        self,
        model_id: str,
        name: str | None = None,
        binary: str = "opencode",
        timeout: float = 1800.0,
        **kwargs,
    ):
        super().__init__(model_id, name, **kwargs)
        self.binary = binary
        self.timeout = timeout
        if shutil.which(binary) is None:
            raise RuntimeError(
                f"{binary!r} was not found on PATH. Install the OpenCode CLI first."
            )

    def query(
        self,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> LLMResponse:
        message = prompt if system is None else f"{system}\n\n{prompt}"
        cmd = [
            self.binary,
            "run",
            "--model",
            self.model_id,
            "--format",
            "json",
            message,
        ]

        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        latency = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f"`{self.binary} run` failed with code {proc.returncode}: "
                f"{proc.stderr.strip()[:500]}"
            )

        text, usage = self._parse_events(proc.stdout)

        return LLMResponse(
            text=text,
            model=self.model_id,
            usage=usage,
            latency=latency,
        )

    @staticmethod
    def _parse_events(stdout: str) -> tuple[str, dict]:
        """Extract assistant text and token usage from the JSON event stream.

        ``--format json`` emits one JSON object per line. The schema is not
        stable across OpenCode releases, so unknown events are skipped and
        the raw stdout is returned if nothing parses.
        """
        chunks = []
        usage = {"input_tokens": 0, "output_tokens": 0}
        parsed_any = False

        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed_any = True

            part = event.get("text")
            if isinstance(part, str):
                chunks.append(part)

            tokens = event.get("tokens") or event.get("usage") or {}
            if isinstance(tokens, dict):
                for src, dst in (
                    ("input", "input_tokens"),
                    ("output", "output_tokens"),
                    ("input_tokens", "input_tokens"),
                    ("output_tokens", "output_tokens"),
                    ("prompt_tokens", "input_tokens"),
                    ("completion_tokens", "output_tokens"),
                ):
                    value = tokens.get(src)
                    if isinstance(value, int):
                        usage[dst] = value

        if not parsed_any or not chunks:
            return stdout.strip(), usage

        return "".join(chunks), usage
