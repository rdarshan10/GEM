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
from typing import Annotated, Literal

from pydantic import Field

from .tools import TOOLS, GemTools

INSTRUCTIONS = (
    "Long-term memory that keeps derived facts consistent. Save durable facts about the user or "
    "project with add_memory; when you store a fact you worked out from others, pass their ids in "
    "derived_from. When a save or forget returns 'invalidated' facts, those are no longer reliable: "
    "do not act on them, and mention the ones that matter. search_memory puts out-of-date matches "
    "in a separate 'stale' list: ask the user instead of using them. Save the answer with "
    "add_memory(answer, resolves=[stale id]); if the user says the old fact still holds, call "
    "add_memory(action='confirm', fact_id=stale id). Then ask about anything in 'reconfirm'."
)

server = MCPServer("gem", instructions=INSTRUCTIONS, log_level="WARNING")
_tools = GemTools()


def _run(name: str, **args) -> str:
    # stdout carries the MCP protocol; anything the engine prints must go to stderr instead
    with contextlib.redirect_stdout(sys.stderr):
        result = _tools.call(name, {k: v for k, v in args.items() if v is not None})
    return json.dumps(result, ensure_ascii=False)


# Descriptions (tool and per-parameter) come from gem.tools.TOOLS, the one source of truth, so
# MCP clients and agents using the Python tool definitions see the same text.
_DEF = {t["name"]: t for t in TOOLS}


def _desc(tool: str) -> str:
    return _DEF[tool]["description"]


def _p(tool: str, param: str):
    return Field(description=_DEF[tool]["input_schema"]["properties"][param].get("description", ""))


@server.tool(description=_desc("add_memory"))
def add_memory(content: Annotated[str, _p("add_memory", "content")] = "",
               action: Annotated[Literal["save", "forget", "confirm"], _p("add_memory", "action")] = "save",
               fact_id: Annotated[str | None, _p("add_memory", "fact_id")] = None,
               derived_from: Annotated[list[str] | None, _p("add_memory", "derived_from")] = None,
               resolves: Annotated[list[str] | None, _p("add_memory", "resolves")] = None,
               container_tag: Annotated[str | None, _p("add_memory", "container_tag")] = None) -> str:
    return _run("add_memory", content=content, action=action, fact_id=fact_id,
                derived_from=derived_from, resolves=resolves, container_tag=container_tag)


@server.tool(description=_desc("search_memory"))
def search_memory(query: Annotated[str, _p("search_memory", "query")],
                  limit: Annotated[int, _p("search_memory", "limit")] = 5,
                  include_stale: Annotated[bool, _p("search_memory", "include_stale")] = False,
                  container_tag: Annotated[str | None, _p("search_memory", "container_tag")] = None) -> str:
    return _run("search_memory", query=query, limit=limit, include_stale=include_stale,
                container_tag=container_tag)


@server.tool(description=_desc("list_memories"))
def list_memories(include_stale: Annotated[bool, _p("list_memories", "include_stale")] = False,
                  container_tag: Annotated[str | None, _p("list_memories", "container_tag")] = None) -> str:
    return _run("list_memories", include_stale=include_stale, container_tag=container_tag)


@server.tool(description=_desc("get_profile"))
def get_profile(container_tag: Annotated[str | None, _p("get_profile", "container_tag")] = None) -> str:
    return _run("get_profile", container_tag=container_tag)


@server.resource("gem://profile", description="Profile of the default memory space.",
                 mime_type="application/json")
def profile_resource() -> str:
    return _run("get_profile")


@server.tool(description=_desc("get_stale"))
def get_stale(container_tag: Annotated[str | None, _p("get_stale", "container_tag")] = None) -> str:
    return _run("get_stale", container_tag=container_tag)


@server.tool(description=_desc("why"))
def why(fact_id: Annotated[str, _p("why", "fact_id")],
        container_tag: Annotated[str | None, _p("why", "container_tag")] = None) -> str:
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
