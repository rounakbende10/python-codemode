"""Direct codemode usage — no orchestrator, no framework.

The simplest way to use codemode: create it, run a task, get results.
Also shows run_code() for pre-written code and batch_run() for parallel.

Usage:
    python3 examples/direct_codemode.py "what meetings do I have today"
    python3 examples/direct_codemode.py --code   # run pre-written code
    python3 examples/direct_codemode.py --batch   # parallel batch execution
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


async def run_with_llm(cm, query: str):
    """LLM generates code from natural language task."""
    print(f"[mode] LLM code generation | query: {query}\n")
    result = await cm.run(query, verbose=True)
    print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))


async def run_prewritten_code(cm):
    """Execute pre-written code directly (no LLM)."""
    print("[mode] Pre-written code | no LLM needed\n")

    result = await cm.run_code("""
async def main():
    time = await tools['get-current-time']()
    cals = await tools['list-calendars']()
    cal_list = next((v for v in cals.values() if isinstance(v, list)), [])
    primary = next((c for c in cal_list if c.get('primary')), cal_list[0] if cal_list else None)
    return {'time': time, 'primary_calendar': primary}
""")
    print(json.dumps(result.get("output", result.get("error")), indent=2, default=str))


async def run_batch(cm):
    """Execute multiple independent tasks in parallel."""
    print("[mode] Batch execution | 3 tasks in parallel\n")

    result = await cm.batch_run([
        "get the current time and timezone",
        "list all available calendars",
        "search for python tutorials on the web",
    ], verbose=True)

    print(f"\nBatch result: {result['succeeded']}/{result['total']} succeeded")
    for task_result in result["tasks"]:
        status = "OK" if task_result.get("success") else "FAIL"
        print(f"  [{status}] {task_result.get('task', '')[:60]}")


async def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = positional[0] if positional else None
    use_code = "--code" in sys.argv
    use_batch = "--batch" in sys.argv

    if not query and not use_code and not use_batch:
        print("Usage:")
        print("  python3 examples/direct_codemode.py 'your query'    # LLM generates code")
        print("  python3 examples/direct_codemode.py --code           # pre-written code")
        print("  python3 examples/direct_codemode.py --batch          # parallel batch")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not use_code:
        print("Error: Set OPENAI_API_KEY in .env (not needed for --code)")
        return

    async with MCPToolLoader() as loader:
        for name, url in MCP_SERVERS.items():
            loader.add_sse_server(name, url)

        mcp_tools = await loader.load_tools()
        print(f"[setup] {len(mcp_tools)} MCP tools loaded")

        cm = codemode(
            tools=mcp_tools,
            backend="pyodide",
            code_model="gpt-5.2-codex",
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )

        if use_code:
            await run_prewritten_code(cm)
        elif use_batch:
            await run_batch(cm)
        else:
            await run_with_llm(cm, query)


if __name__ == "__main__":
    asyncio.run(main())
