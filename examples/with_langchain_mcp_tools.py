"""Codemode with langchain-mcp-tools (community package).

pip install langchain-mcp-tools

Uses convert_mcp_to_langchain_tools() to connect to MCP servers
and convert their tools to LangChain BaseTool objects.

Usage:
    python3 examples/with_langchain_mcp_tools.py "what meetings do I have today"
"""

import asyncio
import json
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from python_codemode import codemode


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else None
    if not query:
        print("Usage: python3 examples/with_langchain_mcp_tools.py 'your query'")
        return

    api_key = os.environ.get("OPENAI_API_KEY")

    # langchain-mcp-tools — community package
    from langchain_mcp_tools import convert_mcp_to_langchain_tools

    server_configs = {
        "calendar": {"transport": "sse", "url": "http://localhost:3001/sse"},
        "serper":   {"transport": "sse", "url": "http://localhost:3002/sse"},
        "github":   {"transport": "sse", "url": "http://localhost:3003/sse"},
    }

    tools, cleanup = await convert_mcp_to_langchain_tools(server_configs)
    print(f"[langchain-mcp-tools] {len(tools)} tools loaded")
    print(f"[langchain-mcp-tools] Tools: {[t.name for t in tools[:5]]}...\n")

    try:
        cm = codemode(tools=tools, api_key=api_key, max_retries=3, timeout=60)
        result = await cm.run(query, verbose=True)

        print("\n" + "=" * 60)
        print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))
        print("=" * 60)
    finally:
        await cleanup()


if __name__ == "__main__":
    asyncio.run(main())
