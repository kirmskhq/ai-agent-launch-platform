"""Harbourline Payments sandbox over MCP (Model Context Protocol).

Any MCP client (Claude Desktop, Claude Code, an IDE, another agent framework) can use the
sandbox's tools through this server. Tools that write are marked as such in their
annotations, taken from config/tools.yaml, so clients can ask before calling them.

Run over stdio (clients usually start it themselves):
    uv run python -m sandbox.mcp_server

Add to Claude Code, for example:
    claude mcp add harbourline -- uv --directory <repo> run python -m sandbox.mcp_server
"""

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from agent_platform.tools import _function, tools_config


def build_server() -> MCPServer:
    server = MCPServer(
        name="harbourline-sandbox",
        instructions="Fictional payments company: invoices, bank statement lines, orders and "
        "refund requests. Data is made up and lives in memory.",
    )
    for name, spec in tools_config().tools.items():
        if not spec.function.startswith("sandbox."):
            continue
        server.add_tool(
            _function(spec),
            name=name,
            description=spec.description,
            annotations=ToolAnnotations(
                read_only_hint=not spec.writes, destructive_hint=spec.writes
            ),
        )
    return server


if __name__ == "__main__":
    build_server().run()
