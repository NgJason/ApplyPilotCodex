"""Registry and resolution helpers for auto-apply agent backends."""
from __future__ import annotations
import os
from applypilot.apply.agents.base import AgentBackend
from applypilot.apply.agents.claude import ClaudeBackend
from applypilot.apply.agents.codex import CodexBackend

BACKENDS: dict[str, type[AgentBackend]] = {"claude": ClaudeBackend, "codex": CodexBackend}
BACKEND_ORDER = ("claude", "codex")

def available_backends() -> list[str]:
    return [name for name in BACKEND_ORDER if BACKENDS[name].is_available()]

def resolve_backend(name: str | None = None) -> AgentBackend:
    requested = name if name not in (None, "auto") else os.environ.get("APPLYPILOT_AGENT")
    if requested == "auto":
        requested = None
    if requested:
        normalized = requested.lower()
        if normalized not in BACKENDS:
            raise ValueError(f"Unknown agent backend '{requested}'. Valid backends: {', '.join(BACKENDS)}.")
        backend_type = BACKENDS[normalized]
        if not backend_type.is_available():
            raise ValueError(f"{backend_type.label} is not installed or not on PATH. Install it from {backend_type.install_hint}.")
        return backend_type()
    installed = available_backends()
    if installed:
        return BACKENDS[installed[0]]()
    hints = " or ".join(f"{backend.label} ({backend.install_hint})" for backend in BACKENDS.values())
    raise ValueError(f"No supported agent CLI found. Install {hints}.")

__all__ = ["BACKENDS", "BACKEND_ORDER", "AgentBackend", "ClaudeBackend", "CodexBackend",
           "available_backends", "resolve_backend"]
