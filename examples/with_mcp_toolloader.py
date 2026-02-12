"""Codemode with our built-in MCPToolLoader.

Returns clean parsed dicts (best for codemode code generation).
Supports SSE and stdio transports, auto-reconnect on session expiry.

Usage:
    python3 examples/with_mcp_toolloader.py "what meetings do I have today"
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
from python_codemode.mcp_adapter import MCPToolLoader


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else None
    if not query:
        print("Usage: python3 examples/with_mcp_toolloader.py 'your query'")
        return

    api_key = os.environ.get("OPENAI_API_KEY")

    # MCPToolLoader — our built-in loader
    async with MCPToolLoader() as loader:
        loader.add_sse_server("calendar", "http://localhost:3001/sse")
        loader.add_sse_server("serper", "http://localhost:3002/sse")
        loader.add_sse_server("github", "http://localhost:3003/sse")

        tools = await loader.load_tools()
        print(f"[MCPToolLoader] {len(tools)} tools loaded")
        print(f"[MCPToolLoader] Returns: dict[str, callable] → clean parsed dicts")
        print(f"[MCPToolLoader] Tools: {list(tools.keys())[:5]}...\n")

        cm = codemode(tools=tools, api_key=api_key, max_retries=3, timeout=60)
        result = await cm.run(query, verbose=True)

        print("\n" + "=" * 60)
        print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
