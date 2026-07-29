"""Codex CLI backend for the auto-apply agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from applypilot.apply.agents.base import AgentBackend, AgentEvent


def _toml_value(value: str | list[str]) -> str:
    """Serialize values using JSON syntax, which is valid TOML for these types."""
    return json.dumps(value)


def _first_present(mapping: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value: Any = mapping
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                value = None
                break
            value = value[part]
        if value is not None:
            return value
    return None


def _arguments(payload: dict) -> dict:
    value = _first_present(payload, ("arguments", "args", "input", "item.arguments", "item.args"))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"text": value}
    return value if isinstance(value, dict) else {}


class CodexBackend(AgentBackend):
    name = "codex"
    cli = "codex"
    label = "Codex"
    install_hint = "https://developers.openai.com/codex/cli"
    default_model = None

    def build_command(self, *, prompt_path_or_stdin: str, model: str | None, mcp: dict,
                      worker_dir: Path, run_dir: Path, worker_id: int) -> list[str]:
        command = [
            self.cli, "exec", "--json", "--skip-git-repo-check", "--ephemeral",
            # Auto-apply must drive a real Chrome over CDP and read the generated
            # resume/cover-letter PDFs out of ~/.applypilot, so the sandbox and
            # approval prompts are bypassed on purpose. Same trust level as
            # Claude Code's --permission-mode bypassPermissions.
            "--dangerously-bypass-approvals-and-sandbox", "-C", str(worker_dir),
            "-o", str(run_dir / "last_message.txt"),
        ]
        if model:
            command.extend(["-m", model])
        for name, server in mcp.items():
            command.extend(["-c", f"mcp_servers.{name}.command={_toml_value(server['command'])}"])
            command.extend(["-c", f"mcp_servers.{name}.args={_toml_value(server.get('args', []))}"])
            # Per-server tool allowlist -- Codex's equivalent of Claude's
            # --disallowedTools. Keeps gmail read/send available and every
            # destructive gmail tool out of the model's reach.
            allowed = server.get("enabled_tools")
            if allowed:
                command.extend(["-c", f"mcp_servers.{name}.enabled_tools={_toml_value(list(allowed))}"])
        command.append(prompt_path_or_stdin)
        return command

    def parse_stream(self, line: str) -> AgentEvent | None:
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(raw, dict):
            return None
        payload = raw.get("msg") if isinstance(raw.get("msg"), dict) else raw
        event_type = str(payload.get("type") or payload.get("item_type") or "").lower().replace("/", ".")
        item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
        item_type = str(item.get("type") or item.get("item_type") or "").lower().replace("/", ".")

        # Deltas are dropped: the completed message arrives as its own event and
        # finalize() appends last_message.txt, so keeping deltas would repeat the
        # agent's answer two or three times in the job log.
        if event_type == "agent_message_delta":
            return AgentEvent("ignore")

        text_types = {"agent_message", "assistant_message"}
        if event_type in text_types or (event_type == "item.completed" and item_type in text_types):
            text = _first_present(payload, ("text", "message", "item.text"))
            if isinstance(text, dict):
                text = _first_present(text, ("text", "content"))
            if isinstance(text, list):
                text = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in text)
            return AgentEvent("text", str(text or ""))

        tool_types = {"exec_command_begin", "mcp_tool_call_begin", "command_execution",
                      "mcp_tool_call", "function_call", "tool_call"}
        is_tool_item = any(token in item_type for token in ("function_call", "tool_call", "command_execution"))
        if event_type in tool_types or is_tool_item:
            server = _first_present(payload, ("server_name", "server", "item.server_name", "item.server"))
            tool = _first_present(payload, ("tool_name", "tool", "name", "item.tool_name", "item.tool", "item.name"))
            command = _first_present(payload, ("command", "item.command"))
            name = ":".join(str(part) for part in (server, tool) if part) or command
            detail = _first_present(_arguments(payload), ("url", "element", "text"))
            description = str(name or item_type or event_type)
            if detail is not None:
                description = f"{description} {str(detail)[:60]}"
            return AgentEvent("tool", description)

        usage_types = {"token_count", "turn.completed", "usage"}
        if event_type in usage_types or item_type in usage_types:
            usage = _first_present(payload, ("usage", "token_usage", "item.usage"))
            usage = usage if isinstance(usage, dict) else payload
            return AgentEvent("usage", stats={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read": usage.get("cached_input_tokens", 0),
                # Codex does not report dollar cost, so the dashboard total
                # stays at $0.000 for Codex runs.
                "cache_create": 0, "cost_usd": 0.0,
                "turns": usage.get("turns", 1 if event_type == "turn.completed" else 0),
            })

        # Recognized envelope with nothing to surface (session metadata, task
        # lifecycle, reasoning traces). Dropped rather than dumped into the log.
        return AgentEvent("ignore")

    def finalize(self, run_dir: Path, text_parts: list[str]) -> str:
        output = super().finalize(run_dir, text_parts)
        last_message_path = run_dir / "last_message.txt"
        if last_message_path.exists():
            last_message = last_message_path.read_text(encoding="utf-8").strip()
            if last_message and last_message not in output:
                output = f"{output}\n{last_message}" if output else last_message
        return output
