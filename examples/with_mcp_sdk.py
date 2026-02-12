"""Codemode with the official Anthropic MCP Python SDK.

pip install mcp

Uses the low-level MCP SDK directly — connects via SSE, loads tools,
and wraps them as plain async callables for codemode.

Usage:
    python3 examples/with_mcp_sdk.py "what meetings do I have today"
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


async def load_tools_from_mcp_sdk(server_url: str) -> dict:
    """Connect to an MCP server using the official SDK and return tools as callables."""
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools_response = await session.list_tools()
            tools = {}

            for tool_def in tools_response.tools:
                name = tool_def.name
                description = tool_def.description or ""
                input_schema = tool_def.inputSchema or {}

                # Create a callable wrapper for each tool
                def make_callable(tool_name, sess):
                    async def call_tool(**kwargs):
                        result = await sess.call_tool(tool_name, arguments=kwargs)
                        # Parse MCP content blocks into clean dicts
                        if result.content:
                            for block in result.content:
                                if hasattr(block, "text"):
                                    try:
                                        return json.loads(block.text)
                                    except (json.JSONDecodeError, TypeError):
                                        return block.text
                        return None

                    call_tool.__name__ = tool_name
                    call_tool.__doc__ = description
                    # Attach schema for code generator
                    call_tool._mcp_schema = {
                        "name": tool_name,
                        "description": description,
                        "inputSchema": input_schema,
                    }
                    return call_tool

                tools[name] = make_callable(name, session)

            return tools


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else None
    if not query:
        print("Usage: python3 examples/with_mcp_sdk.py 'your query'")
        return

    api_key = os.environ.get("OPENAI_API_KEY")

    # Official MCP SDK — low-level, manual wrapping
    # Note: SSE client is a context manager, so tools only work within scope
    from mcp.client.session import ClientSession
    from mcp.client.sse import sse_client

    server_urls = [
        ("calendar", "http://localhost:3001/sse"),
        ("serper", "http://localhost:3002/sse"),
        ("github", "http://localhost:3003/sse"),
    ]

    # Since the MCP SDK uses context managers, we need to keep sessions alive
    # while codemode runs. Connect to each server and collect tools.
    print("[mcp SDK] Connecting to servers...")

    all_tools = {}
    sessions = []
    contexts = []

    for name, url in server_urls:
        sse_ctx = sse_client(url)
        read, write = await sse_ctx.__aenter__()
        contexts.append(sse_ctx)

        session = ClientSession(read, write)
        await session.__aenter__()
        sessions.append(session)

        await session.initialize()
        tools_response = await session.list_tools()

        for tool_def in tools_response.tools:
            tool_name = tool_def.name
            description = tool_def.description or ""
            input_schema = tool_def.inputSchema or {}

            def make_callable(tn, sess, desc, schema):
                async def call_tool(**kwargs):
                    result = await sess.call_tool(tn, arguments=kwargs)
                    if result.content:
                        for block in result.content:
                            if hasattr(block, "text"):
                                try:
                                    return json.loads(block.text)
                                except (json.JSONDecodeError, TypeError):
                                    return block.text
                    return None

                call_tool.__name__ = tn
                call_tool.__doc__ = desc
                call_tool._mcp_schema = {
                    "name": tn,
                    "description": desc,
                    "inputSchema": schema,
                }
                return call_tool

            all_tools[tool_name] = make_callable(tool_name, session, description, input_schema)

    print(f"[mcp SDK] {len(all_tools)} tools loaded")
    print(f"[mcp SDK] Returns: dict[str, callable] → manually wrapped")
    print(f"[mcp SDK] Tools: {list(all_tools.keys())[:5]}...\n")

    try:
        cm = codemode(tools=all_tools, api_key=api_key, max_retries=3, timeout=60)
        result = await cm.run(query, verbose=True)

        print("\n" + "=" * 60)
        print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))
        print("=" * 60)
    finally:
        # Clean up — ignore errors from context manager edge cases
        for session in sessions:
            try:
                await session.__aexit__(None, None, None)
            except Exception:
                pass
        for ctx in contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(main())
