# python-codemode

A Python port of [Cloudflare's Codemode](https://github.com/cloudflare/codemode) -- wrap your tools, let LLMs generate Python code, and execute it safely in a sandbox.

## Architecture

```
+-------------------+       +------------------+       +------------------+
|                   |       |                  |       |                  |
|   User / LLM      +------>   CodeMode API    +------>   Code Generator |
|   (task string)   |       |   (codemode.py)  |       |   (generator.py)|
|                   |       |                  |       |                  |
+-------------------+       +--------+---------+       +------------------+
                                     |
                                     v
                            +--------+---------+
                            |                  |
                            |   Tool Proxy     |
                            |   (proxy.py)     |
                            |                  |
                            +--------+---------+
                                     |
                    +----------------+----------------+
                    |                |                 |
           +--------+------+ +------+-------+ +------+-------+
           |               | |              | |              |
           | Pyodide       | | Docker       | | nsjail       |
           | Backend       | | Backend      | | Backend      |
           | (restricted   | | (container)  | | (Linux jail) |
           |  exec)        | |              | |              |
           +---------------+ +--------------+ +--------------+
```

### Components

| Module | Description |
|--------|-------------|
| `src/codemode.py` | Main public API -- `codemode()` factory and `CodeMode` class |
| `src/generator.py` | LLM-based code generation with validation and fallback |
| `src/proxy.py` | Routes sandbox tool calls to real tool implementations |
| `src/schema.py` | Converts Python callables to JSON Schema for LLM prompts |
| `src/metrics.py` | Tracks execution counts, durations, retries, and token usage |
| `src/backends/base.py` | Abstract `SandboxBackend` base class and `ExecutionResult` |
| `src/backends/pyodide_backend.py` | Default backend using restricted `exec()` with AST validation |
| `src/main.py` | CLI entry point with argparse |

## Installation

```bash
# From source
pip install -e .

# With optional dependencies
pip install -e ".[openai]"     # OpenAI code generation
pip install -e ".[docker]"     # Docker sandbox backend
pip install -e ".[all]"        # Everything
```

## Quick Start

### Python API

```python
import asyncio
from src import codemode

# Define your tools
async def search_web(query: str) -> list:
    return [{"title": "Result", "url": "https://example.com"}]

async def create_event(title: str, date: str) -> dict:
    return {"id": "evt_1", "title": title, "date": date}

# Create a CodeMode instance
cm = codemode(
    tools={
        "search_web": search_web,
        "create_event": create_event,
    },
    backend="pyodide",       # sandbox backend
    model="gpt-4o-mini",     # LLM for code generation
    max_retries=3,           # retry on failure
    timeout=30,              # execution timeout (seconds)
)

# Run a task
result = asyncio.run(cm.run("Search for Python conferences and create calendar events"))
print(result)
# {
#   "success": True,
#   "output": {...},
#   "duration": 1.23,
#   "backend": "pyodide",
#   "tool_calls": [...]
# }
```

### Execute Pre-written Code

```python
result = asyncio.run(cm.run_code("""
async def main():
    results = await tools['search_web']('Python tutorials')
    return {'results': results}
"""))
```

### CLI

```bash
# Run a task with default settings
python -m src "Search for Python tutorials"

# Use a specific backend with verbose output
python -m src "Create a GitHub issue" --backend pyodide --verbose

# Compare backends
python -m src "Search the web" --compare

# Set timeout and retries
python -m src "Complex task" --timeout 60 --max-retries 5
```

## Sandbox Security

The Pyodide backend enforces safety by:

1. **AST-level import checking** -- blocks `os`, `subprocess`, `sys`, `shutil`, `socket`, `signal`, `ctypes`, and `importlib`
2. **Restricted builtins** -- only safe builtins are exposed (`print`, `len`, `range`, math operations, etc.)
3. **Timeout enforcement** -- configurable execution timeout via `asyncio.wait_for`
4. **Tool isolation** -- tools are accessed only through the proxy, which logs every call

### Forbidden Modules

The following modules are blocked from import in sandboxed code:

- `os` - filesystem and process access
- `subprocess` - shell command execution
- `sys` - interpreter internals
- `shutil` - file operations
- `socket` - network access
- `signal` - process signals
- `ctypes` - C library access
- `importlib` - dynamic imports

## Code Generation

The generator creates sandbox-safe Python code from natural language. It:

1. Formats tool schemas into a prompt
2. Calls the configured LLM (defaults to `gpt-4o-mini`)
3. Extracts code from the response (handles markdown fences)
4. Validates the code via AST analysis
5. Falls back to template-based generation if the LLM is unavailable

### Validation Rules

- Code must contain an `async def main()` entry point
- No forbidden module imports
- Must be valid Python syntax

## Metrics

Access execution metrics through the `CodeMode.metrics` property:

```python
cm = codemode(tools={...})
await cm.run("some task")

# Get summary
print(cm.metrics.summary())
# {
#   "total_executions": 1,
#   "successful_executions": 1,
#   "total_tool_calls": 2,
#   "total_tokens": 0,
#   "retries": 0,
#   "avg_execution_time": 0.045,
# }

# Pretty table
print(cm.metrics.format_table())
```

## Backends

| Backend | Platform | Isolation Level | Description |
|---------|----------|----------------|-------------|
| `pyodide` | Any | Medium | Restricted `exec()` with AST checks (default) |
| `docker` | Any with Docker | High | Full container isolation |
| `nsjail` | Linux only | Very High | Linux namespace jail |

Backends are selected at construction time and fall back to `pyodide` if unavailable.

## Framework Integrations

The `CodeMode` instance can be exported for use with popular frameworks:

```python
cm = codemode(tools={...})

# LangChain
tool = cm.as_langchain_tool()

# OpenAI function calling
fn_def = cm.as_openai_function()
tool_def = cm.as_openai_tool()

# Vercel AI SDK
vercel_tool = cm.as_vercel_tool()
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing
```

## License

MIT
