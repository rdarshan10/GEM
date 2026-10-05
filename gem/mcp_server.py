"""GEM MCP server — dependency-aware memory as tools for Claude Code, Claude Desktop, Cursor, etc.

Same tool surface as Supermemory's MCP (add_memory save/forget, search_memory, list_memories,
container_tag spaces), plus get_stale and why. Runs locally over stdio; memory persists per the
GEM_STORE setting (JSON files under ~/.gem by default). Needs `pip install gem-memory[mcp]`.

    claude mcp add gem -- python -m gem.mcp_server
or in an MCP client config:
    {"mcpServers": {"gem": {"command": "python", "args": ["-m", "gem.mcp_server"],
                            "env": {"GEM_LLM": "groq", "GROQ_MODEL": "openai/gpt-oss-120b"}}}}
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys

from mcp.server.mcpserver import MCPServer

from .llm import load_dotenv
from .tools import GemTools

INSTRUCTIONS = (
    "Long-term memory that keeps derived facts consistent. Save durable facts about the user or "
    "project with add_memory; when you store a fact you worked out from others, pass their ids in "
    "derived_from. When a save or forget returns 'invalidated' facts, those are no longer reliable: "
    "do not act on them, and mention the ones that matter. search_memory puts out-of-date matches "
    "in a separate 'stale' list: confirm those with the user instead of using them."
)

server = MCPServer("gem", instructions=INSTRUCTIONS, log_level="WARNING")
_tools = GemTools()


def _run(name: str, **args) -> str:
    # stdout carries the MCP protocol; anything the engine prints must go to stderr instead
    with contextlib.redirect_stdout(sys.stderr):
        result = _tools.call(name, {k: v for k, v in args.items() if v is not None})
    return json.dumps(result, ensure_ascii=False)


@server.tool(description="Save a fact to memory (action='save') or retract one (action='forget'). "
                         "Returns facts the change corrected ('revised') and facts it made unreliable "
                         "('invalidated'), including ones derived from it several steps away.")
def add_memory(content: str, action: str = "save", fact_id: str | None = None,
               derived_from: list[str] | None = None, container_tag: str | None = None) -> str:
    return _run("add_memory", content=content, action=action, fact_id=fact_id,
                derived_from=derived_from, container_tag=container_tag)


@server.tool(description="Find stored facts relevant to a query. 'results' are still valid; "
                         "'stale' are matching facts now out of date (ask the user, don't use them).")
def search_memory(query: str, limit: int = 5, include_stale: bool = False,
                  container_tag: str | None = None) -> str:
    return _run("search_memory", query=query, limit=limit, include_stale=include_stale,
                container_tag=container_tag)


@server.tool(description="List stored facts in a memory space with their status.")
def list_memories(include_stale: bool = False, container_tag: str | None = None) -> str:
    return _run("list_memories", include_stale=include_stale, container_tag=container_tag)


@server.tool(description="Summary of what memory holds about the user or project (from still-valid "
                         "facts), plus facts to reconfirm. Call at the start of a conversation.")
def get_profile(container_tag: str | None = None) -> str:
    return _run("get_profile", container_tag=container_tag)


@server.resource("gem://profile", description="Profile of the default memory space.",
                 mime_type="application/json")
def profile_resource() -> str:
    return _run("get_profile")


@server.tool(description="List facts that went stale because something they depended on changed. "
                         "Reconfirm these before acting on them.")
def get_stale(container_tag: str | None = None) -> str:
    return _run("get_stale", container_tag=container_tag)


@server.tool(description="Show which stored facts a fact was derived from.")
def why(fact_id: str, container_tag: str | None = None) -> str:
    return _run("why", fact_id=fact_id, container_tag=container_tag)


def main() -> None:
    load_dotenv()
    # keep stderr readable in MCP client logs: no per-request HTTP logs or progress bars
    for name in ("typesafe_sdk", "httpx", "httpx2", "sentence_transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)
    os.environ.setdefault("TQDM_DISABLE", "1")
    server.run("stdio")


if __name__ == "__main__":
    main()
