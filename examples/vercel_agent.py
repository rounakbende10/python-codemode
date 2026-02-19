"""Vercel AI SDK style agent with codemode.

Shows how to use cm.as_vercel_tool() for Vercel AI SDK integration.
Since Vercel AI SDK is JavaScript-native, this example shows how to
expose the tool as an HTTP endpoint that a Vercel AI SDK frontend
can call, plus a standalone Python demo.

Usage:
    python3 examples/vercel_agent.py "what meetings do I have today"
    python3 examples/vercel_agent.py "list my github repos"
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Load .env
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from python_codemode import codemode
from python_codemode.mcp_adapter import MCPToolLoader

MCP_SERVERS = {
    "calendar": "http://localhost:3001/sse",
    "serper":   "http://localhost:3002/sse",
    "github":   "http://localhost:3003/sse",
}


async def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = positional[0] if positional else None

    if not query:
        print("Usage: python3 examples/vercel_agent.py 'your query'")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY in .env")
        return

    # 1. Load MCP tools
    async with MCPToolLoader() as loader:
        for name, url in MCP_SERVERS.items():
            loader.add_sse_server(name, url)

        mcp_tools = await loader.load_tools()
        print(f"[setup] {len(mcp_tools)} MCP tools loaded")

        # 2. Create codemode and get Vercel tool
        cm = codemode(
            tools=mcp_tools,
            backend="pyodide-wasm",
            code_model="gpt-5.2-codex",
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )

        vercel_tool = cm.as_vercel_tool()
        print(f"[setup] Vercel tool ready")
        print(f"[setup] description: {vercel_tool['description'][:60]}...")
        print(f"[setup] parameters: {list(vercel_tool['parameters']['properties'].keys())}")

        # 3. Execute via Vercel tool's execute function
        #    In a real Vercel AI SDK app, this would be called by the SDK.
        #    Here we call it directly to demonstrate.
        print(f"[agent] {query}\n")

        result_str = await vercel_tool["execute"]({"task": query})
        result = json.loads(result_str)

        # 4. Use orchestrator to interpret results
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.responses.create(
            model="gpt-5-mini",
            input=(
                f"The user asked: {query}\n\n"
                f"The codemode tool returned:\n{json.dumps(result, indent=2, default=str)}\n\n"
                f"Provide a clear, concise answer based on these results."
            ),
        )

        print("\n" + "=" * 60)
        print(response.output_text)
        print("=" * 60)

        # 5. Show the Vercel tool schema (for integration reference)
        print("\n--- Vercel AI SDK Tool Schema (for frontend integration) ---")
        schema = {
            "description": vercel_tool["description"],
            "parameters": vercel_tool["parameters"],
        }
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
