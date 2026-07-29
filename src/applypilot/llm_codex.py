"""LLM provider backed by the locally installed Codex CLI.

Lets the scoring, tailoring, cover-letter and extraction stages run through
`codex exec` instead of an HTTP LLM API, so ApplyPilot needs no API key when
Codex is already installed and signed in.

Selected when LLM_PROVIDER=codex, or automatically when no other provider is
configured and `codex` is on PATH. Exposes the same chat()/ask()/close()
surface as llm.LLMClient so callers don't care which one they got.

Environment:
  LLM_PROVIDER=codex     Force this provider even if API keys are present.
  CODEX_MODEL            Model passed to `codex exec -m` (default: Codex's own).
  CODEX_LLM_TIMEOUT      Per-request timeout in seconds (default: 300).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from applypilot import config

log = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_WAIT = 5
_DEFAULT_TIMEOUT = 300

# Codex is an agent, not a completion endpoint. Without this it will happily
# go read files, run shell commands, or narrate what it is about to do -- none
# of which the callers can parse.
_DIRECTIVE = """You are being used as a text-generation endpoint, not as a coding agent.

Rules for this request:
- Do NOT use tools, run shell commands, read or write files, or search the web.
- Answer from the text below only.
- Output ONLY the requested content. No preamble, no explanation of your
  reasoning, no follow-up questions, no summary of what you did.
- If the request asks for JSON, output raw JSON and nothing else.
"""


def is_available() -> bool:
    """True when the Codex CLI can be found on PATH."""
    return shutil.which("codex") is not None


def _flatten(messages: list[dict]) -> str:
    """Collapse OpenAI-style messages into a single Codex prompt."""
    system = [m.get("content", "") for m in messages if m.get("role") == "system"]
    turns = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            continue
        label = "REQUEST" if role == "user" else "PREVIOUS ANSWER"
        turns.append(f"== {label} ==\n{msg.get('content', '')}")

    parts = [_DIRECTIVE]
    if system:
        parts.append("== SYSTEM INSTRUCTIONS ==\n" + "\n\n".join(system))
    parts.extend(turns)
    return "\n\n".join(parts)


class CodexCLIClient:
    """Drives `codex exec` as a single-shot text generator.

    Each call runs in its own throwaway working directory with an ephemeral
    session, so concurrent pipeline workers never share state.
    """

    def __init__(self, model: str | None = None, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.model = model
        self.timeout = timeout
        self.base_url = "codex-cli"  # so log lines/diagnostics read uniformly

    def _build_command(self, out_path: Path, work_dir: Path) -> list[str]:
        command = [
            "codex", "exec",
            "--color", "never",
            "--skip-git-repo-check",
            "--ephemeral",              # don't litter ~/.codex with one session per job
            "-s", "read-only",          # text generation needs no write access
            "-c", 'approval_policy="never"',
            "-c", "mcp_servers={}",     # no browser/gmail servers for this path
            "-C", str(work_dir),
            "-o", str(out_path),
        ]
        if self.model:
            command.extend(["-m", self.model])
        command.append("-")             # read the prompt from stdin
        return command

    def chat(self, messages: list[dict], temperature: float = 0.0,
             max_tokens: int = 4096) -> str:
        """Send messages to Codex and return the final assistant text.

        temperature and max_tokens are accepted for interface compatibility but
        the Codex CLI exposes no equivalent knobs, so they are ignored.
        """
        prompt = _flatten(messages)
        config.ensure_dirs()

        last_error = ""
        for attempt in range(_MAX_RETRIES):
            work_dir = Path(tempfile.mkdtemp(prefix="req-", dir=str(config.LLM_RUN_DIR)))
            out_path = work_dir / "response.txt"
            try:
                proc = subprocess.run(
                    self._build_command(out_path, work_dir),
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=str(work_dir),
                )
                if proc.returncode != 0:
                    last_error = (proc.stderr or proc.stdout or "").strip()[:400]
                    log.warning("Codex CLI exited %d (attempt %d/%d): %s",
                                proc.returncode, attempt + 1, _MAX_RETRIES, last_error)
                elif out_path.exists():
                    answer = out_path.read_text(encoding="utf-8").strip()
                    if answer:
                        return answer
                    last_error = "Codex returned an empty response"
                else:
                    last_error = "Codex wrote no response file"

            except subprocess.TimeoutExpired:
                last_error = f"Codex CLI timed out after {self.timeout}s"
                log.warning("%s (attempt %d/%d)", last_error, attempt + 1, _MAX_RETRIES)
            except OSError as exc:
                raise RuntimeError(f"Could not launch the Codex CLI: {exc}") from exc
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

            if attempt < _MAX_RETRIES - 1:
                time.sleep(_RETRY_BASE_WAIT * (2 ** attempt))

        raise RuntimeError(f"Codex CLI request failed after {_MAX_RETRIES} attempts: {last_error}")

    def ask(self, prompt: str, **kwargs) -> str:
        """Convenience: single user prompt -> assistant response."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def close(self) -> None:
        """No persistent resources to release."""


def check_auth() -> tuple[bool, str]:
    """Best-effort check that Codex is installed and signed in.

    Returns (ok, detail). Never raises -- used by `applypilot doctor`.
    """
    if not is_available():
        return False, "codex not found on PATH"
    try:
        proc = subprocess.run(
            ["codex", "login", "status"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not query Codex login status ({exc})"

    detail = (proc.stdout or proc.stderr or "").strip().splitlines()
    summary = detail[0][:80] if detail else ""
    if proc.returncode != 0:
        return False, summary or "not signed in — run 'codex login'"
    return True, summary or "signed in"


__all__ = ["CodexCLIClient", "check_auth", "is_available"]
