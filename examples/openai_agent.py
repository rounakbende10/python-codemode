"""OpenAI Responses API agent with codemode.

Uses the OpenAI Responses API directly (no LangChain).
Codemode is registered as an OpenAI function/tool.

Usage:
    python3 examples/openai_agent.py "what meetings do I have today"
    python3 examples/openai_agent.py "search for AI conferences and create a github issue"
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

ORCHESTRATOR_MODEL = "gpt-5-mini"


async def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = positional[0] if positional else None

    if not query:
        print("Usage: python3 examples/openai_agent.py 'your query'")
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

        # 2. Create codemode and get OpenAI tool schema
        cm = codemode(
            tools=mcp_tools,
            backend="pyodide",
            code_model="gpt-5.2-codex",
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )

        openai_tool = cm.as_openai_tool()
        print(f"[setup] OpenAI tool: {openai_tool['function']['name']}")

        # 3. OpenAI Responses API agent loop
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        print(f"[agent] model={ORCHESTRATOR_MODEL} | query={query}\n")

        # Initial call with codemode as a tool
        response = client.responses.create(
            model=ORCHESTRATOR_MODEL,
            input=query,
            tools=[{
                "type": "function",
                "name": openai_tool["function"]["name"],
                "description": openai_tool["function"]["description"],
                "parameters": openai_tool["function"]["parameters"],
            }],
            instructions=(
                "You are a helpful assistant. Use the codemode tool to accomplish tasks. "
                "Call it ONCE, then interpret the results and respond clearly."
            ),
        )

        # Process tool calls
        while response.output:
            tool_calls = [item for item in response.output if item.type == "function_call"]

            if not tool_calls:
                break

            for tool_call in tool_calls:
                if tool_call.name == "codemode":
                    args = json.loads(tool_call.arguments)
                    task = args.get("task", query)
                    print(f"[agent] Calling codemode: {task[:80]}...")

                    result = await cm.run(task, verbose=True)
                    result_str = json.dumps(result.get("output", result.get("error")), indent=2, default=str)

                    # Feed result back to orchestrator
                    response = client.responses.create(
                        model=ORCHESTRATOR_MODEL,
                        previous_response_id=response.id,
                        input=[{
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": result_str,
                        }],
                    )

        # 4. Print final answer
        print("\n" + "=" * 60)
        print(response.output_text)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
