"""Shared types and interface for auto-apply agent backends."""
from __future__ import annotations
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AgentEvent:
    """One normalized event emitted by an agent CLI."""
    kind: str
    text: str = ""
    stats: dict = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)

@dataclass
class AgentResult:
    """Final normalized result from an agent CLI."""
    output: str
    returncode: int
    stats: dict

class AgentBackend(ABC):
    """Interface implemented by each supported browser-agent CLI."""
    name: str
    cli: str
    label: str
    install_hint: str
    default_model: str | None

    @classmethod
    def is_available(cls) -> bool:
        return shutil.which(cls.cli) is not None

    def prepare_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        return dict(env if env is not None else os.environ)

    @abstractmethod
    def build_command(self, *, prompt_path_or_stdin: str, model: str | None, mcp: dict,
                      worker_dir: Path, run_dir: Path) -> list[str]:
        """Build the CLI argv for one agent run."""

    @abstractmethod
    def parse_stream(self, line: str) -> AgentEvent | None:
        """Normalize one JSONL line, returning None for unknown shapes."""

    def finalize(self, run_dir: Path, text_parts: list[str]) -> str:
        return "\n".join(text_parts)
