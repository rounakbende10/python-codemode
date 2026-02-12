"""Codemode with langchain-mcp-adapters (official LangChain MCP package).

pip install langchain-mcp-adapters

Uses MultiServerMCPClient to connect to MCP servers and load tools
as LangChain BaseTool objects.

Usage:
    python3 examples/with_langchain_mcp_adapters.py "what meetings do I have today"
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
        print("Usage: python3 examples/with_langchain_mcp_adapters.py 'your query'")
        return

    api_key = os.environ.get("OPENAI_API_KEY")

    # langchain-mcp-adapters — official LangChain MCP package
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient({
        "calendar": {"transport": "sse", "url": "http://localhost:3001/sse"},
        "serper":   {"transport": "sse", "url": "http://localhost:3002/sse"},
        "github":   {"transport": "sse", "url": "http://localhost:3003/sse"},
    })

    tools = await client.get_tools()
    print(f"[langchain-mcp-adapters] {len(tools)} tools loaded")
    print(f"[langchain-mcp-adapters] Returns: list[BaseTool] → raw MCP content blocks")
    print(f"[langchain-mcp-adapters] Tools: {[t.name for t in tools[:5]]}...\n")

    # Pass LangChain tools directly — codemode accepts list[BaseTool]
    cm = codemode(tools=tools, api_key=api_key, max_retries=3, timeout=60)
    result = await cm.run(query, verbose=True)

    print("\n" + "=" * 60)
    print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
