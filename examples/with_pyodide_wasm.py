"""Codemode with actual Pyodide WASM backend.

Runs LLM-generated Python code inside a real Pyodide WebAssembly
sandbox via Node.js — true memory isolation.

Requires:
    - Node.js installed: brew install node
    - Pyodide npm package: npm install pyodide

Usage:
    python3 examples/with_pyodide_wasm.py "what meetings do I have today"
    python3 examples/with_pyodide_wasm.py --compare "get current time"   # compare both backends
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

from python_codemode import codemode
from python_codemode.mcp_adapter import MCPToolLoader

MCP_SERVERS = {
    "calendar": "http://localhost:3001/sse",
    "serper":   "http://localhost:3002/sse",
    "github":   "http://localhost:3003/sse",
}


async def run_single(query: str, backend: str):
    """Run query on a single backend."""
    api_key = os.environ.get("OPENAI_API_KEY")

    async with MCPToolLoader() as loader:
        for name, url in MCP_SERVERS.items():
            loader.add_sse_server(name, url)

        tools = await loader.load_tools()
        print(f"[setup] {len(tools)} MCP tools loaded")

        cm = codemode(
            tools=tools,
            backend=backend,
            api_key=api_key,
            max_retries=3,
            timeout=60,
        )
        print(f"[setup] Backend: {cm._backend.get_name()}")
        print(f"[setup] Query: {query}\n")

        result = await cm.run(query, verbose=True)

        print("\n" + "=" * 60)
        if result["success"]:
            print(json.dumps(result["output"], indent=2, default=str))
        else:
            print(f"ERROR: {result.get('error')}")
        print("=" * 60)

        if result.get("metrics"):
            m = result["metrics"]
            print(f"\nTokens: {m.get('total_tokens', 0)} | "
                  f"Tools: {m.get('total_tool_calls', 0)} | "
                  f"Retries: {m.get('retry_count', 0)}")


async def run_compare(query: str):
    """Run the same query on both backends and compare."""
    api_key = os.environ.get("OPENAI_API_KEY")

    async with MCPToolLoader() as loader:
        for name, url in MCP_SERVERS.items():
            loader.add_sse_server(name, url)

        tools = await loader.load_tools()
        print(f"[setup] {len(tools)} MCP tools loaded")
        print(f"[compare] Running on both backends...\n")

        # Pre-written code so both backends run the exact same thing
        code = """
async def main():
    time = await tools['get-current-time']()
    cals = await tools['list-calendars']()
    cal_list = next((v for v in cals.values() if isinstance(v, list)), [])
    primary = next((c for c in cal_list if c.get('primary')), cal_list[0] if cal_list else None)
    cal_id = primary.get('id') if primary else 'primary'

    import datetime
    time_str = next((v for v in time.values() if isinstance(v, str) and 'T' in v), '')
    now = datetime.datetime.fromisoformat(time_str) if time_str else datetime.datetime.now()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + datetime.timedelta(days=1)

    events = await tools['list-events'](
        calendarId=cal_id, timeMin=day_start.isoformat(), timeMax=day_end.isoformat()
    )
    ev_list = next((v for v in events.values() if isinstance(v, list)), [])
    return {
        'backend': 'TBD',
        'time': time_str,
        'calendar': cal_id,
        'event_count': len(ev_list),
        'events': [e.get('summary', 'no title') for e in ev_list[:5]],
    }
"""

        results = {}
        for backend in ["pyodide", "pyodide-wasm"]:
            print(f"--- {backend} ---")
            cm = codemode(tools=tools, backend=backend, timeout=60)
            import time
            start = time.monotonic()
            result = await cm.run_code(code)
            duration = time.monotonic() - start
            results[backend] = {
                "success": result["success"],
                "duration": round(duration, 2),
                "output": result.get("output"),
                "error": result.get("error"),
            }
            status = "OK" if result["success"] else "FAIL"
            print(f"  [{status}] {duration:.2f}s")
            if result["success"]:
                out = result["output"]
                print(f"  Events: {out.get('event_count', '?')} — {out.get('events', [])}")
            else:
                print(f"  Error: {result.get('error', '')[:100]}")
            print()

        # Summary
        print("=" * 60)
        print(f"{'Backend':<20} {'Status':<8} {'Duration':<10} {'Events':<8}")
        print("-" * 60)
        for backend, r in results.items():
            status = "OK" if r["success"] else "FAIL"
            events = r["output"].get("event_count", "?") if r.get("output") else "?"
            print(f"{backend:<20} {status:<8} {r['duration']:<10.2f} {events:<8}")
        print("=" * 60)


async def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    query = positional[0] if positional else None
    compare = "--compare" in sys.argv

    if not query:
        print("Usage:")
        print("  python3 examples/with_pyodide_wasm.py 'your query'")
        print("  python3 examples/with_pyodide_wasm.py --compare 'get current time'")
        return

    if compare:
        await run_compare(query)
    else:
        await run_single(query, "pyodide-wasm")


if __name__ == "__main__":
    asyncio.run(main())
