"""Codemode with mcp-use (standalone MCP agent library).

pip install mcp-use

Uses MCPClient with LangChainAdapter to load tools from MCP servers.

Usage:
    python3 examples/with_mcp_use.py "what meetings do I have today"
"""

import asyncio
import json
import os
import sys
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python_codemode import codemode


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else None
    if not query:
        print("Usage: python3 examples/with_mcp_use.py 'your query'")
        return

    api_key = os.environ.get("OPENAI_API_KEY")

    # mcp-use — standalone MCP agent library
    from mcp_use import MCPClient
    from mcp_use.adapters.langchain_adapter import LangChainAdapter

    # Config format matches Claude Desktop / mcp config
    config = {
        "mcpServers": {
            "calendar": {
                "transport": "sse",
                "url": "http://localhost:3001/sse",
            },
            "serper": {
                "transport": "sse",
                "url": "http://localhost:3002/sse",
            },
            "github": {
                "transport": "sse",
                "url": "http://localhost:3003/sse",
            },
        }
    }

    client = MCPClient(config)
    adapter = LangChainAdapter()
    tools = await adapter.create_tools(client)
    print(f"[mcp-use] {len(tools)} tools loaded")
    print(f"[mcp-use] Returns: list[BaseTool] via LangChainAdapter")
    print(f"[mcp-use] Tools: {[t.name for t in tools[:5]]}...\n")

    try:
        # Pass tools to codemode
        cm = codemode(tools=tools, api_key=api_key, max_retries=3, timeout=60)
        result = await cm.run(query, verbose=True)

        print("\n" + "=" * 60)
        print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))
        print("=" * 60)
    finally:
        if hasattr(client, "close"):
            await client.close()


if __name__ == "__main__":
    asyncio.run(main())
