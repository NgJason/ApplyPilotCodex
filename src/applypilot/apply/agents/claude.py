"""Claude Code backend for the auto-apply agent."""
from __future__ import annotations
import json
from pathlib import Path
from applypilot import config
from applypilot.apply.agents.base import AgentBackend, AgentEvent

_DISALLOWED_GMAIL_TOOLS = (
    "mcp__gmail__draft_email,mcp__gmail__modify_email,"
    "mcp__gmail__delete_email,mcp__gmail__download_attachment,"
    "mcp__gmail__batch_modify_emails,mcp__gmail__batch_delete_emails,"
    "mcp__gmail__create_label,mcp__gmail__update_label,"
    "mcp__gmail__delete_label,mcp__gmail__get_or_create_label,"
    "mcp__gmail__list_email_labels,mcp__gmail__create_filter,"
    "mcp__gmail__list_filters,mcp__gmail__get_filter,mcp__gmail__delete_filter"
)

class ClaudeBackend(AgentBackend):
    name = "claude"
    cli = "claude"
    label = "Claude Code"
    install_hint = "https://claude.ai/code"
    default_model = "haiku"

    def build_command(self, *, prompt_path_or_stdin: str, model: str | None, mcp: dict,
                      worker_dir: Path, run_dir: Path) -> list[str]:
        worker_id = worker_dir.name.rsplit("-", 1)[-1]
        mcp_path = config.APP_DIR / f".mcp-apply-{worker_id}.json"
        mcp_path.write_text(json.dumps({"mcpServers": mcp}), encoding="utf-8")
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
        if message.get("type") == "assistant":
            texts, tools = [], []
            for block in message.get("message", {}).get("content", []):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    name = block.get("name", "").replace("mcp__playwright__", "").replace("mcp__gmail__", "gmail:")
                    inp = block.get("input", {})
                    if "url" in inp:
                        desc = f"{name} {str(inp['url'])[:60]}"
                    elif "ref" in inp:
                        desc = f"{name} {inp.get('element', inp.get('text', ''))}"[:50]
                    elif "fields" in inp:
                        desc = f"{name} ({len(inp['fields'])} fields)"
                    elif "paths" in inp:
                        desc = f"{name} upload"
                    else:
                        desc = name
                    tools.append(desc)
            if texts:
                return AgentEvent("text", "\n".join(texts), tools=tools)
            if tools:
                return AgentEvent("tool", "\n".join(tools))
        elif message.get("type") == "result":
            usage = message.get("usage", {})
            return AgentEvent("text", message.get("result", ""), {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_create": usage.get("cache_creation_input_tokens", 0),
                "cost_usd": message.get("total_cost_usd", 0),
                "turns": message.get("num_turns", 0),
            })
        return None
