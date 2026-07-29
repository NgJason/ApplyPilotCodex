"""Shared types and interface for auto-apply agent backends."""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

# Event kinds returned by AgentBackend.parse_stream()
#   "text"   -- assistant prose; appended to the output scanned for RESULT: codes
#   "tool"   -- a tool/command invocation; shown on the dashboard and worker log
#   "usage"  -- token/cost accounting for the run
#   "ignore" -- a well-formed event this backend deliberately drops
EVENT_KINDS = ("text", "tool", "usage", "ignore")


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
                      worker_dir: Path, run_dir: Path, worker_id: int) -> list[str]:
        """Build the CLI argv for one agent run.

        Backends may write per-run support files (Claude Code needs its MCP
        config on disk); `run_dir` and `worker_id` scope those to one worker.
        """

    def debug_command(self, **kwargs) -> list[str]:
        """Argv printed by `applypilot apply --gen`.

        Defaults to the exact runtime argv so the manual command reproduces a
        real run. Override only if a flag is genuinely unusable interactively.
        """
        return self.build_command(**kwargs)

    @abstractmethod
    def parse_stream(self, line: str) -> AgentEvent | None:
        """Normalize one line of the CLI's JSONL stream.

        Return an AgentEvent for any line that parses as a known envelope --
        including `AgentEvent("ignore")` for events this backend drops. Return
        None ONLY when the line is not parseable, which tells the caller to
        treat it as raw passthrough text.
        """

    def finalize(self, run_dir: Path, text_parts: list[str]) -> str:
        return "\n".join(text_parts)
