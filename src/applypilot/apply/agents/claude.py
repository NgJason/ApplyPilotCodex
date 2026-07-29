"""Claude Code backend for the auto-apply agent."""

from __future__ import annotations

import json
from pathlib import Path

from applypilot import config
from applypilot.apply.agents.base import AgentBackend, AgentEvent

# Gmail tools the agent must never reach. send_email/read_email/search_emails
# stay available on purpose: the prompt uses them for email-only applications
# and for retrieving login verification codes.
_DISALLOWED_GMAIL_TOOLS = (
    "mcp__gmail__draft_email,mcp__gmail__modify_email,"
    "mcp__gmail__delete_email,mcp__gmail__download_attachment,"
    "mcp__gmail__batch_modify_emails,mcp__gmail__batch_delete_emails,"
    "mcp__gmail__create_label,mcp__gmail__update_label,"
    "mcp__gmail__delete_label,mcp__gmail__get_or_create_label,"
    "mcp__gmail__list_email_labels,mcp__gmail__create_filter,"
    "mcp__gmail__list_filters,mcp__gmail__get_filter,mcp__gmail__delete_filter"
)

# Keys of the shared MCP spec that belong in Claude's on-disk config. Anything
# else (e.g. the per-server tool allowlist Codex consumes) is dropped.
_MCP_CONFIG_KEYS = ("command", "args", "env")


class ClaudeBackend(AgentBackend):
    name = "claude"
    cli = "claude"
    label = "Claude Code"
    install_hint = "https://claude.ai/code"
    default_model = "haiku"

    def build_command(self, *, prompt_path_or_stdin: str, model: str | None, mcp: dict,
                      worker_dir: Path, run_dir: Path, worker_id: int) -> list[str]:
        """Build the argv, writing this worker's MCP config file as a side effect."""
        servers = {
            name: {k: v for k, v in server.items() if k in _MCP_CONFIG_KEYS}
            for name, server in mcp.items()
        }
        mcp_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
        mcp_path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")

        command = [self.cli]
        if model:
            command.extend(["--model", model])
        command.extend([
            "-p", "--mcp-config", str(mcp_path),
            "--permission-mode", "bypassPermissions", "--no-session-persistence",
            "--disallowedTools", _DISALLOWED_GMAIL_TOOLS,
            "--output-format", "stream-json", "--verbose", prompt_path_or_stdin,
        ])
        return command

    def prepare_env(self, env: dict[str, str] | None = None) -> dict[str, str]:
        prepared = super().prepare_env(env)
        prepared.pop("CLAUDECODE", None)
        prepared.pop("CLAUDE_CODE_ENTRYPOINT", None)
        return prepared

    def parse_stream(self, line: str) -> AgentEvent | None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(message, dict):
            return None

        msg_type = message.get("type")
        if msg_type == "assistant":
            texts, tools = [], []
            for block in message.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tools.append(self._describe_tool(block))
            if texts:
                return AgentEvent("text", "\n".join(texts), tools=tools)
            if tools:
                return AgentEvent("tool", "\n".join(tools))
        elif msg_type == "result":
            usage = message.get("usage", {})
            return AgentEvent("text", message.get("result", ""), {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_create": usage.get("cache_creation_input_tokens", 0),
                "cost_usd": message.get("total_cost_usd", 0),
                "turns": message.get("num_turns", 0),
            })

        # Well-formed envelope we don't surface: system/init, hook events,
        # rate_limit_event, and the tool_result "user" turns whose payloads
        # carry entire page snapshots. Dropping them keeps those out of the
        # job log and out of the text scanned for RESULT: codes.
        return AgentEvent("ignore")

    @staticmethod
    def _describe_tool(block: dict) -> str:
        name = block.get("name", "").replace("mcp__playwright__", "").replace("mcp__gmail__", "gmail:")
        inp = block.get("input", {})
        if "url" in inp:
            return f"{name} {str(inp['url'])[:60]}"
        if "ref" in inp:
            return f"{name} {inp.get('element', inp.get('text', ''))}"[:50]
        if "fields" in inp:
            return f"{name} ({len(inp['fields'])} fields)"
        if "paths" in inp:
            return f"{name} upload"
        return name
