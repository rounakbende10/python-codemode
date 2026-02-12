# python-codemode

A Python port of [Cloudflare's Codemode](https://blog.cloudflare.com/code-mode/) — wraps your tools, lets LLMs generate Python code, and executes it safely in a sandbox.

Instead of calling tools one at a time, the LLM writes code that calls multiple tools, chains results, runs parallel operations, and processes data — all in a single execution.

## How It Works

```
User: "what meetings do I have today"
  │
  ▼
Orchestrator LLM (gpt-5-mini)
  │ decides to call codemode
  ▼
CodeMode
  ├── gpt-5.2-codex generates Python code
  ├── Code runs in sandbox (restricted exec or Pyodide WASM)
  ├── tools['get-current-time']()  ──→  MCP Server (Calendar)
  ├── tools['list-calendars']()    ──→  MCP Server (Calendar)
  ├── tools['list-events'](...)    ──→  MCP Server (Calendar)
  └── Returns structured results
  │
  ▼
Orchestrator interprets results
  │
  ▼
"You have 1 meeting today: ET Data Science Bi-Weekly at 11:00 AM"
```

## Prerequisites

- **Python 3.11+**
- **OpenAI API key** — for LLM code generation (gpt-5.2-codex / gpt-4o)
- **MCP servers running** — the tools codemode calls (calendar, search, github, etc.)
- **Node.js** (optional) — only for `pyodide-wasm` backend

## Install

```bash
# 1. Clone the repo
git clone https://github.com/rounakbende10/python-codemode.git
cd python-codemode

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install with LangChain support
pip install -e ".[langchain]"

# 4. Set your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 5. (Optional) For Pyodide WASM backend
npm install pyodide

# 6. (Optional) For running tests
pip install -r requirements-dev.txt
pytest
```

### What each dependency does

| Package | Why | Installed by |
|---|---|---|
| `openai` | LLM code generation via Responses API | `pip install -e .` |
| `aiohttp` | MCP SSE connections (MCPToolLoader) | `pip install -e .` |
| `pydantic` | Schema validation | `pip install -e .` |
| `langchain`, `langchain-openai` | LangChain agent integration | `pip install -e ".[langchain]"` |
| `langchain-mcp-adapters` | Official LangChain MCP loader | `pip install -e ".[langchain]"` |
| `pyodide` (npm) | Actual WASM sandbox | `npm install pyodide` |
| `pytest`, `pytest-asyncio` | Testing | `pip install -r requirements-dev.txt` |

### Start MCP servers

Codemode needs MCP servers running for the tools. Example with npx:

```bash
# Google Calendar MCP (port 3001)
npx -y @anthropic/google-calendar-mcp --port 3001

# Serper search MCP (port 3002)
SERPER_API_KEY=your-key npx -y @anthropic/serper-mcp --port 3002

# GitHub MCP (port 3003)
GITHUB_TOKEN=your-token npx -y @anthropic/github-mcp --port 3003
```

Adjust server commands and ports to match your MCP setup. The examples default to ports 3001/3002/3003.

### Verify everything works

```bash
# Run tests (no MCP servers needed)
pytest

# Test with MCP servers running
python3 examples/langchain_agent.py "what meetings do I have today"
```

## Quick Start

### Direct usage

```python
from python_codemode import codemode

async def search(query: str) -> list:
    return [{"title": f"Result for {query}"}]

cm = codemode(tools={"search": search}, backend="pyodide")

# LLM generates code from natural language
result = await cm.run("search for python tutorials")

# Or run pre-written code (no LLM needed)
result = await cm.run_code("""
async def main():
    results = await tools['search']('python tutorials')
    return {'results': results}
""")
```

### With MCP servers

```python
from python_codemode import codemode
from python_codemode.mcp_adapter import MCPToolLoader

async with MCPToolLoader() as loader:
    loader.add_sse_server("calendar", "http://localhost:3001/sse")
    loader.add_sse_server("serper",   "http://localhost:3002/sse")
    loader.add_sse_server("github",   "http://localhost:3003/sse")

    tools = await loader.load_tools()  # 40 tools discovered

    cm = codemode(tools=tools, backend="pyodide")
    result = await cm.run("what meetings do I have today")
```

### LangChain agent

```python
from python_codemode import codemode
from python_codemode.mcp_adapter import MCPToolLoader
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

async with MCPToolLoader() as loader:
    loader.add_sse_server("calendar", "http://localhost:3001/sse")
    tools = await loader.load_tools()

    cm = codemode(tools=tools, backend="pyodide-wasm")
    codemode_tool = cm.as_langchain_tool()

    llm = ChatOpenAI(model="gpt-5-mini")
    agent = create_agent(llm, tools=[codemode_tool], system_prompt="...")
    result = await agent.ainvoke({"messages": [("user", "what meetings today")]})
```

### With LangChain @tool

```python
from langchain_core.tools import tool
from python_codemode import codemode

@tool
async def search(query: str) -> list:
    """Search the web."""
    return [{"title": "Result"}]

@tool
async def notify(message: str) -> dict:
    """Send a notification."""
    return {"sent": True}

cm = codemode(tools=[search, notify])
result = await cm.run("search for AI news and notify me")
```

### OpenAI Responses API

```python
from python_codemode import codemode
from openai import OpenAI

cm = codemode(tools=mcp_tools)
openai_tool = cm.as_openai_tool()

client = OpenAI()
response = client.responses.create(
    model="gpt-5-mini",
    input="what meetings do I have today",
    tools=[openai_tool],
)
```

### Vercel AI SDK

```python
cm = codemode(tools=mcp_tools)
vercel_tool = cm.as_vercel_tool()

# vercel_tool has: description, parameters, execute
result = await vercel_tool["execute"]({"task": "list my events"})
```

### Batch execution (parallel)

```python
result = await cm.batch_run([
    "search for AI conferences",
    "list my calendar events",
    "check open github issues",
])
# 3 tasks run in parallel, each with own code generation
```

## Architecture

```
python-codemode/
├── python_codemode/
│   ├── __init__.py              # codemode(), create_agent(), create_codemode_tool()
│   ├── codemode.py              # CodeMode class — self-healing retry loop
│   ├── agent.py                 # LangChain/LangGraph integration
│   ├── generator.py             # LLM code generation (OpenAI Responses API)
│   ├── proxy.py                 # Tool call routing + logging + None stripping
│   ├── schema.py                # JSON Schema ↔ Python type conversion
│   ├── metrics.py               # Per-call token tracking + execution metrics
│   ├── mcp_adapter.py           # MCP SSE/stdio client with auto-reconnect
│   ├── backends/
│   │   ├── base.py              # SandboxBackend abstract class
│   │   ├── pyodide_backend.py   # Restricted exec() sandbox (default)
│   │   ├── pyodide_wasm_backend.py  # Actual Pyodide WASM via Node.js
│   │   ├── pyodide_runner.js    # Node.js script for WASM execution
│   │   ├── docker_backend.py    # Docker container sandbox
│   │   └── nsjail_backend.py    # nsjail/gVisor sandbox (Linux)
│   └── integrations/
│       ├── langchain.py         # cm.as_langchain_tool()
│       ├── openai.py            # cm.as_openai_tool() / cm.as_openai_function()
│       └── vercel.py            # cm.as_vercel_tool()
├── examples/                    # Working examples for every integration
├── tests/                       # 204 tests
└── docs/
```

## Backends

| Backend | `backend=` | Isolation | Speed | Requirements |
|---|---|---|---|---|
| Restricted exec | `"pyodide"` | AST-checked imports + restricted builtins | Fast (sub-ms) | None |
| Pyodide WASM | `"pyodide-wasm"` | True WASM memory isolation | ~0.7s boot | Node.js + `npm install pyodide` |
| Docker | `"docker"` | Container isolation | ~2s boot | Docker running |
| nsjail | `"nsjail"` | Syscall-level (Google-grade) | Fast | Linux + nsjail binary |

```python
cm = codemode(tools=tools, backend="pyodide")       # default — fast
cm = codemode(tools=tools, backend="pyodide-wasm")   # true isolation
cm = codemode(tools=tools, backend="docker")          # container
cm = codemode(tools=tools, backend="nsjail")          # syscall sandbox
```

Falls back to restricted exec if the requested backend isn't available.

## MCP Tool Loaders

Works with any MCP client library:

```python
# Built-in MCPToolLoader (recommended — returns clean dicts)
from python_codemode.mcp_adapter import MCPToolLoader
async with MCPToolLoader() as loader:
    loader.add_sse_server("calendar", "http://localhost:3001/sse")
    tools = await loader.load_tools()

# langchain-mcp-adapters (official LangChain)
from langchain_mcp_adapters.client import MultiServerMCPClient
client = MultiServerMCPClient({"cal": {"transport": "sse", "url": "..."}})
tools = await client.get_tools()

# langchain-mcp-tools (community)
from langchain_mcp_tools import convert_mcp_to_langchain_tools
tools, cleanup = await convert_mcp_to_langchain_tools(config)

# mcp-use
from mcp_use import MCPClient
from mcp_use.adapters.langchain_adapter import LangChainAdapter
tools = await LangChainAdapter().create_tools(MCPClient(config))

# All work with codemode:
cm = codemode(tools=tools)
```

## Self-Healing Retry Loop

When generated code fails, codemode feeds the error + actual tool responses back to the LLM:

```
Attempt 1: LLM guesses response format → KeyError: 'items'
           Tool responses captured by proxy

Attempt 2: LLM sees actual data:
           "tools['list-events']() returned dict with keys: ['events', 'totalCount']"
           → fixes code → succeeds
```

Empty output detection: if code "succeeds" but returns `{}` while tools were called, triggers a retry with tool response data.

Previous code is included in retry prompt (like Cloudflare) so the LLM sees what already ran and writes idempotent code.

## Metrics & Logging

```python
import logging
logging.getLogger("codemode").setLevel(logging.INFO)

result = await cm.run("task", verbose=True)

# Per-call token usage
result["metrics"]["llm_calls"]
# [{"attempt": 1, "model": "gpt-5.2-codex", "input_tokens": 7895, "output_tokens": 952, ...}]

# Cumulative
result["metrics"]["total_tokens"]       # 8847
result["metrics"]["total_input_tokens"] # 7895

# Tool breakdown
result["metrics"]["tool_breakdown"]
# {"get-current-time": {"calls": 1, "successes": 1, "total_time": 0.51}, ...}
```

Verbose output:

```
╔══════════════════════════════════════════════════════════╗
║               CODEMODE METRICS                         ║
╠══════════════════════════════════════════════════════════╣
║ LLM Calls
║   Attempt 1 | gpt-5.2-codex | in=7895 out=952 total=8847 | 17.56s
║   Cumulative: in=7895 out=952 total=8847 | 17.56s
╠──────────────────────────────────────────────────────────
║ Sandbox Executions
║   Total: 1 | OK: 1 | Failed: 0 | Retries: 0
╠──────────────────────────────────────────────────────────
║ Tool Calls
║   Total: 5 | OK: 5 | Failed: 0
║
║   Name                      Calls   OK Fail     Time
║   ────────────────────────────────────────────────
║   get-current-time              1    1    0   0.513s
║   list-calendars                1    1    0   0.543s
║   list-events                   3    3    0   2.605s
╚══════════════════════════════════════════════════════════╝
```

## Examples

```bash
# Agent examples
python3 examples/langchain_agent.py "what meetings do I have today"
python3 examples/openai_agent.py "search for AI conferences"
python3 examples/vercel_agent.py "list my github repos"

# Direct codemode
python3 examples/direct_codemode.py "what meetings today"
python3 examples/direct_codemode.py --code    # pre-written code
python3 examples/direct_codemode.py --batch   # parallel tasks

# MCP loader examples
python3 examples/with_mcp_toolloader.py "query"
python3 examples/with_langchain_mcp_adapters.py "query"
python3 examples/with_langchain_mcp_tools.py "query"
python3 examples/with_mcp_use.py "query"
python3 examples/with_mcp_sdk.py "query"

# Pyodide WASM backend
python3 examples/with_pyodide_wasm.py "query"
python3 examples/with_pyodide_wasm.py --compare "query"  # compare backends
```

## Configuration

```python
cm = codemode(
    tools=tools,                    # dict or list of tools
    backend="pyodide",              # sandbox backend
    code_model="gpt-5.2-codex",     # code generation model
    model="gpt-5-mini",             # orchestrator model
    api_key="sk-...",               # OpenAI API key
    max_retries=3,                  # max code generation retries
    timeout=60,                     # sandbox execution timeout (seconds)
)
```

## Comparison with Cloudflare's Codemode

| | Cloudflare | python-codemode |
|---|---|---|
| Language | TypeScript / JavaScript | Python |
| Sandbox | V8 isolates (WASM) | Restricted exec, Pyodide WASM, Docker, nsjail |
| Tool schemas | TypeScript interfaces from `inputSchema` + `outputSchema` | Python type stubs from `inputSchema` |
| Structured output | `generateObject({ schema })` | `responses.create(text={format: json_schema})` |
| Parallel calls | `Promise.all()` | `asyncio.gather()` |
| Retry | Previous code + error | Previous code + error + tool responses + key listing |
| MCP responses | Raw content blocks (`JSON.parse(content[0].text)`) | Pre-parsed dicts (MCPToolLoader) |
| Integrations | Vercel AI SDK | LangChain, OpenAI, Vercel |

## License

MIT
