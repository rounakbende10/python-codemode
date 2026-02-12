"""LangChain agent using cm.as_langchain_tool().

The simplest way: create a CodeMode instance, call .as_langchain_tool(),
and pass it to a standard LangChain agent.

Usage:
    python3 examples/langchain_agent.py "what meetings do I have today"
    python3 examples/langchain_agent.py "search for AI conferences"
"""

import asyncio
import logging
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

# Enable codemode logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
# Quiet down noisy libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)

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
        print("Usage: python3 examples/langchain_agent.py 'your query'")
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

        # 2. Create codemode instance and get LangChain tool
        cm = codemode(
            tools=mcp_tools,
            backend="pyodide-wasm",
            code_model="gpt-5.2-codex",
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )

        codemode_tool = cm.as_langchain_tool()
        print(f"[setup] codemode tool: {codemode_tool.name}")
        print(f"[setup] description: {codemode_tool.description[:80]}...")

        # 3. Standard LangChain agent setup
        from langchain_openai import ChatOpenAI
        from langchain.agents import create_agent as lc_create_agent

        llm = ChatOpenAI(model="gpt-5-mini", api_key=api_key)

        agent = lc_create_agent(
            llm,
            tools=[codemode_tool],
            system_prompt=(
                "You are a helpful assistant with a codemode tool. "
                "Call codemode ONCE to accomplish the task. "
                "After receiving results, provide a clear answer. "
                "Do NOT call codemode again after getting results."
            ),
        )

        # 4. Run
        print(f"[agent] {query}\n")

        result = await agent.ainvoke(
            {"messages": [("user", query)]}
        )

        print("\n" + "=" * 60)
        print(result["messages"][-1].content)
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
