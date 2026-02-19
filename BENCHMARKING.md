# Benchmarking with MCP-Bench

MCP-Bench measures how well an agent uses MCP tools to complete tasks. It connects to real MCP servers, runs tasks, and scores results using an LLM judge + programmatic metrics.

This document covers setup, how the benchmark works, scoring, results, and how to customize it.

## Table of Contents

- [Setup](#setup)
- [Running Benchmarks](#running-benchmarks)
- [How It Works](#how-it-works)
- [Agent Modes](#agent-modes)
- [Scoring System](#scoring-system)
- [Results](#results)
- [Customization](#customization)
- [Task File Format](#task-file-format)
- [Configuration Reference](#configuration-reference)

## Setup

### Prerequisites

- Python 3.11+
- Node.js (for some MCP servers)
- OpenAI API key
- [mcp-bench](https://github.com/Accenture/mcp-bench) repo cloned alongside python-codemode

### Directory structure

```
Code-mode/
├── python-codemode/          # this repo
└── mcp-bench/                # Accenture's benchmark framework
    ├── run_benchmark.py      # entry point
    ├── agent/
    │   ├── executor.py       # default agent (upstream)
    │   └── codemode_executor.py  # codemode agent (local addition)
    ├── benchmark/
    │   ├── runner.py         # orchestrates everything
    │   ├── evaluator.py      # LLM judge scoring
    │   ├── results_aggregator.py  # averages scores across tasks
    │   └── results_formatter.py   # displays metrics
    ├── config/
    │   └── benchmark_config.yaml  # all tunable parameters
    ├── mcp_servers/
    │   ├── commands.json     # how to start each server
    │   └── api_key           # API keys for servers that need them
    ├── tasks/                # task definition files
    └── cache/                # SQLite cache for tool call results
```

### Install

```bash
# 1. Install python-codemode
cd python-codemode
pip install -e ".[dev]"

# 2. Set up mcp-bench
cd ../mcp-bench
pip install -r requirements.txt

# 3. Configure API keys
#    Edit .env — uncomment and fill in your keys:
cp .env.example .env   # if needed
```

**.env file:**
```bash
OPENAI_API_KEY=sk-...              # required — for LLM + judge

# Optional — for specific MCP servers:
# GOOGLE_MAPS_API_KEY=your-key     # Google Maps server
# NASA_API_KEY=your-key            # NASA Data server
# NPS_API_KEY=your-key             # National Parks server
# HF_TOKEN=your-token              # Hugging Face server
# NCI_API_KEY=your-key             # BioMCP server
```

**Important:** The `.env` file does not use `export`. You must source it with `set -a`:
```bash
set -a && source .env && set +a
```

### Verify servers

```bash
# Check which models are available
python3 run_benchmark.py --list-models

# Quick test with 2 tasks
python3 run_benchmark.py \
  --models gpt-5-mini \
  --tasks-file tasks/test_2tasks.json \
  --agent codemode \
  --distraction-count 0
```

## Running Benchmarks

### Basic commands

```bash
cd mcp-bench
set -a && source .env && set +a

# Codemode agent — single-shot code generation
python3 run_benchmark.py \
  --models gpt-5-mini \
  --tasks-file tasks/test_2tasks.json \
  --agent codemode \
  --distraction-count 0

# Default agent — traditional multi-round tool calling
python3 run_benchmark.py \
  --models gpt-5 \
  --tasks-file tasks/test_2tasks.json \
  --agent default \
  --distraction-count 0

# Full benchmark (56 tasks, 28 servers, 10 distractions)
python3 run_benchmark.py \
  --models gpt-5-mini \
  --tasks-file tasks/mcpbench_tasks_single_runner_format.json \
  --agent codemode \
  --distraction-count 10

# Multiple models
python3 run_benchmark.py --models gpt-5-mini gpt-4o gpt-5

# All available models
python3 run_benchmark.py --models all
```

### Command-line flags

| Flag | Description | Default |
|---|---|---|
| `--models MODEL [...]` | Models to test (or `all`) | `o4-mini` |
| `--list-models` | List available models and exit | — |
| `--tasks-file FILE` | Task JSON file(s), comma-separated | All task files |
| `--agent {default\|codemode}` | Agent executor type | `default` |
| `--distraction-count N` | Number of irrelevant servers to add | `10` |
| `--disable-fuzzy` | Show detailed task descriptions to agent | Fuzzy enabled |
| `--disable-judge-stability` | Run judge once instead of 5 times | 5 runs |
| `--disable-filter-problematic-tools` | Include known buggy tools | Filtered |
| `--enable-cache` | Cache tool call results in SQLite | Enabled |
| `--cache-ttl HOURS` | Cache time-to-live (0 = permanent) | `0` |
| `--output FILE` | Output JSON filename | Auto-timestamped |
| `--verbose` / `-v` | Debug logging | Off |

### Output

Results are saved to `benchmark_results_YYYYMMDD_HHMMSS.json`:

```json
{
  "task_completion_score": 5.2,
  "tool_selection_score": 5.9,
  "planning_effectiveness_and_efficiency_score": 4.6,
  "task_fulfillment": 4.6,
  "grounding": 5.7,
  "tool_appropriateness": 6.6,
  "parameter_accuracy": 5.6,
  "dependency_awareness": 3.9,
  "parallelism_and_efficiency": 4.4,
  "input_schema_compliance": 1.0,
  "valid_tool_name_rate": 1.0,
  "tool_call_success_rate": 0.0,
  "avg_execution_time": 92.9,
  "avg_agent_execution_time": 91.4,
  "avg_total_rounds": 2.3,
  "avg_total_tokens": 15926
}
```

## How It Works

### Full flow diagram

```
run_benchmark.py
│
▼
BenchmarkRunner.run_benchmark()
│
├── Load task file (JSON)
├── Load server configs (mcp_servers/commands.json)
├── Load model configs (check env vars for API keys)
├── Init tool cache (SQLite)
│
└── For each model:
    └── For each task:
        │
        ├── 1. PREPARE
        │   ├── Read task_id, fuzzy_description (→ agent)
        │   ├── Read concrete_task_description (→ judge only)
        │   ├── Map server_name → server config
        │   ├── Add resident server (Time MCP)
        │   └── Add N distraction servers
        │
        ├── 2. CONNECT
        │   ├── Start MCP servers (stdio subprocesses)
        │   ├── Initialize sessions
        │   └── Discover all tools
        │
        ├── 3. EXECUTE
        │   │
        │   ├── [default agent]                    [codemode agent]
        │   │    │                                  │
        │   │    ├── For round 1..20:               ├── Generate Python code
        │   │    │   ├── LLM plans actions          │   (gpt-5.2-codex)
        │   │    │   ├── Execute tool calls         │
        │   │    │   ├── Accumulate results         ├── Run in sandbox
        │   │    │   │   in context                 │   (all tools in one pass)
        │   │    │   └── Continue?                  │
        │   │    │                                  ├── If error → retry
        │   │    └── Synthesize final answer        │   (up to 3 attempts)
        │   │                                       │
        │   │    Context grows each round           Code stays in sandbox
        │   │    ~5K → 25K → 80K → 400K+ tokens    ~16K tokens total
        │   │
        │   └── Returns: solution, tool_calls, tokens, rounds
        │
        ├── 4. EVALUATE
        │   │
        │   ├── LLM Judge (5 randomized evaluations)
        │   │   ├── Scores 6 subdimensions (1-10 each)
        │   │   ├── Compares agent output vs concrete reference
        │   │   └── Averages 5 runs for stability
        │   │
        │   ├── Programmatic metrics
        │   │   ├── input_schema_compliance (jsonschema validation)
        │   │   ├── valid_tool_name_rate (tool exists?)
        │   │   └── execution_success_rate (tool returned OK?)
        │   │
        │   └── Server utilization metrics
        │
        ├── 5. RECORD
        │   └── Store scores, tokens, duration
        │
        └── 6. CLEANUP
            └── Close MCP server connections

Final: Aggregate all task scores → JSON output
```

### What the agent sees vs what the judge sees

| | Agent | Judge |
|---|---|---|
| Task description | `fuzzy_description` (natural language) | Both fuzzy + `concrete_task_description` (step-by-step) |
| Tool schemas | All discovered tools (required + resident + distractions) | Same tool list |
| Dependency analysis | Not shown | Shown as reference |
| Prior results | Accumulated (default) or in sandbox vars (codemode) | Full execution log |

The judge compares the agent's output against the concrete reference. This means the agent might solve the task correctly but in a different way than prescribed, resulting in a lower score.

### Tool call flow

```
Agent decides to call tool
  │
  ▼
ToolProxy.call() (python-codemode/proxy.py)
  ├── Strip None kwargs
  ├── Unwrap 'arguments' dict pattern
  │
  ▼
server_manager.call_tool(tool_id, params)
  ├── Check cache (SQLite) → return if hit
  ├── Call MCP server via stdio session
  ├── Cache result (if not error)
  └── Return result
  │
  ▼
ToolProxy post-processing
  ├── Unwrap tuple responses (LangChain artifacts)
  ├── Unwrap MCP content blocks ([{type:"text", text:"..."}])
  ├── Parse JSON strings → dicts
  ├── Detect validation errors → raise ToolExecutionError
  └── Return processed result
```

### Caching

Tool call results are cached in `cache/tool_cache.db` (SQLite). Cache behavior:

- **Key**: `MD5(server_name:tool_name:sorted_params_json)`
- **TTL**: permanent by default (`--cache-ttl 0`)
- **Not cached**: error responses, validation errors, rate limit errors, `isError=True` results
- **Cleared manually**: delete `cache/tool_cache.db` or run with `--cache-ttl 1`

Caching prevents redundant API calls across benchmark runs. The same `get_current_time(timezone="UTC")` call returns the cached result instead of hitting the MCP server again.

## Agent Modes

### Default agent (`--agent default`)

The original mcp-bench agent from Accenture's repo. Multi-round planning with native tool calling.

```
Round 1: LLM plans → call tools → results added to context
Round 2: LLM plans (sees round 1 results) → call tools → results added
Round 3: LLM plans (sees rounds 1+2 results) → call tools → ...
...
Round N: LLM decides to stop → synthesize final answer
```

**Key characteristics:**
- Up to 20 rounds per task
- Full conversation history resent each round (context grows)
- Native tool calling — LLM sees tool schemas as function definitions
- Gets parameter names right on first try (~8.4/10 parameter accuracy)
- Can iteratively refine based on tool outputs
- Token usage grows quadratically: ~500K tokens/task average

**Code location:** `mcp-bench/agent/executor.py` (upstream, unmodified)

### Codemode agent (`--agent codemode`)

Plugs python-codemode into the benchmark. Single-shot code generation.

```
Attempt 1: LLM generates Python code → sandbox executes all tools → done
  └── If error: feed error + tool responses → Attempt 2 → ...
```

**Key characteristics:**
- 1 LLM call generates all code (retries only on failure)
- Tool results stay in Python variables (never enter LLM context)
- `asyncio.gather()` for parallel tool calls
- Relies on retry loop for parameter correction
- Token usage flat: ~16K tokens/task average
- Code generation uses `gpt-5.2-codex` (hardcoded in `codemode_executor.py`)

**Code location:** `mcp-bench/agent/codemode_executor.py` (local addition)

**Note:** The `--models` flag selects the model for the default agent and the LLM judge. Codemode always uses `gpt-5.2-codex` for code generation regardless of `--models`. To change it, set `CODEMODE_CODE_MODEL` env var or edit `codemode_executor.py:100`.

## Scoring System

### 6 subdimensions (LLM judge, 1-10 each)

Grouped into 3 top-level scores:

```
Task Completion Score    = avg(Task Fulfillment, Grounding)
Tool Selection Score     = avg(Tool Appropriateness, Parameter Accuracy)
Planning Effectiveness   = avg(Dependency Awareness, Parallelism & Efficiency)
```

| Subdimension | What It Measures | Example |
|---|---|---|
| **Task Fulfillment** | % of required steps completed correctly | Did it convert all 14 units? |
| **Grounding** | % of claims backed by tool output evidence | Are numbers from actual tool calls, not hallucinated? |
| **Tool Appropriateness** | Were the right tools selected? | Did it use `convert_temperature` for temperature, not `convert_length`? |
| **Parameter Accuracy** | Were tool parameters correct? | Did it use `source_timezone` or `from_tz`? |
| **Dependency Awareness** | Were dependent calls chained correctly? | Did it get the calendar ID before listing events? |
| **Parallelism & Efficiency** | Were independent calls parallelized? Minimal redundancy? | Did it call 3 independent tools with `gather()` or sequentially? |

### Score calibration

The judge maps defect rates to scores:

| Defect Rate | Score | Meaning |
|---|---|---|
| 0-10% | 9-10 | Nearly perfect |
| 10-30% | 7-9 | Good with minor issues |
| 30-50% | 5-7 | Partial completion |
| 50-70% | 3-5 | Significant gaps |
| 70-100% | 0-3 | Mostly failed |

Default expected score is **4-5** (average). Scores of **8+** require strong evidence.

### Judge stability

Each task is evaluated **5 times** with randomized ordering:
- Dimension order shuffled each run
- Subdimension order shuffled
- Criteria order shuffled
- Final score = average of 5 runs

This reduces positional bias in the LLM judge. Disable with `--disable-judge-stability` (faster, less reliable).

### Programmatic metrics (calculated, not judged)

| Metric | How It's Calculated |
|---|---|
| `input_schema_compliance` | `jsonschema.validate(params, tool.inputSchema)` for each call |
| `valid_tool_name_rate` | `tool_name in available_tools` for each call |
| `tool_call_success_rate` | `result.success == True` count / total calls |

### Token counting

- **Default agent**: sum of all `prompt_tokens + completion_tokens` across all planning rounds + final synthesis
- **Codemode agent**: sum of code generation tokens across attempts (1-3 attempts). Does NOT include an orchestrator — in production there would be ~1-2K additional tokens for the outer LLM
- **Source**: `results_aggregator.py` averages per-task token counts (upstream code, unmodified)

## Results

### Codemode vs Default — Same 3 Tasks

Apples-to-apples comparison on identical tasks (unit_converter_000, unit_converter_001, wikipedia_000), 0 distraction servers. Default used `gpt-5`, codemode used `gpt-5.2-codex`.

#### Efficiency

| Task | Default Time | Codemode Time | Default Tokens | Codemode Tokens | Default Cost | Codemode Cost |
|---|---|---|---|---|---|---|
| unit_converter_000 | 466.8s | 74.7s | 58K | 8K | $0.29 | $0.08 |
| unit_converter_001 | 203.6s | 101.8s | 30K | 20K | $0.15 | $0.14 |
| wikipedia_000 | 4,284.7s (71 min) | 97.6s | 1,407K | 19K | $2.30 | $0.13 |
| **Total (3 tasks)** | **4,955s** | **274s** | **1,495K** | **48K** | **$2.74** | **$0.35** |

Costs based on OpenAI API pricing: gpt-5 at $1.25/1M input + $10/1M output, gpt-5.2-codex at $1.75/1M input + $14/1M output.

**At 1,000 tasks**: default ~$913, codemode ~$118 (7.8x cheaper).

#### Quality

| Task | Metric | Default (gpt-5) | Codemode (gpt-5.2-codex) |
|---|---|---|---|
| unit_converter_000 | Task / Tool / Plan | 8.4 / 8.3 / 8.1 | 5.5 / 3.7 / 4.4 |
| unit_converter_001 | Task / Tool / Plan | 8.5 / 7.6 / 8.2 | 7.3 / 8.0 / 5.4 |
| wikipedia_000 | Task / Tool / Plan | 6.4 / 7.8 / 5.5 | 3.2 / 6.1 / 3.9 |
| **Average** | **Task / Tool / Plan** | **7.8 / 7.9 / 7.3** | **5.3 / 5.9 / 4.6** |

#### Context growth (why default is slow)

```
Default agent — context accumulates each round (wikipedia_000 actual data):
  Round  1:  prompt_tokens ~5,000       (task + tool schemas)
  Round  5:  prompt_tokens ~25,000      (+ 4 rounds of results)
  Round 10:  prompt_tokens ~80,000      (growing)
  Round 20:  prompt_tokens ~1,300,000   (1.3M — full history of all 20 rounds)
  Total:     1,407K tokens for one task

Codemode — flat, no accumulation:
  Attempt 1: prompt_tokens ~5,000       (task + tool schemas)
  Attempt 2: prompt_tokens ~7,000       (+ error context)
  Attempt 3: prompt_tokens ~8,000       (+ error context)
  Total:     ~16K tokens for the same task
```

#### Tradeoffs

| | Default Agent | Codemode Agent |
|---|---|---|
| **Strengths** | Higher scores (~1.5x); native tool calling; iterative refinement | ~31x fewer tokens; ~7.8x cheaper; ~18x faster; parallel execution |
| **Weaknesses** | Context growth; slow on complex tasks; ~$0.91/task | Lower scores; relies on retry; code gen can fail |
| **Best for** | Quality-critical, low-volume | Cost-sensitive, high-throughput |

## Customization

### Writing new tasks

Create a JSON file following the [task file format](#task-file-format) and run:

```bash
python3 run_benchmark.py \
  --tasks-file tasks/my_tasks.json \
  --agent codemode \
  --distraction-count 0
```

### Adding a new MCP server

1. Clone/install the server in `mcp_servers/`:
   ```bash
   cd mcp_servers
   git clone https://github.com/user/my-mcp-server
   cd my-mcp-server && npm install  # or pip install
   ```

2. Add to `mcp_servers/commands.json`:
   ```json
   "My Server": {
     "cmd": "node dist/index.js",
     "env": ["MY_API_KEY"],
     "cwd": "../my-mcp-server"
   }
   ```

3. If it needs an API key, add to `mcp_servers/api_key`:
   ```
   MY_API_KEY = your-key-here
   ```

4. Write tasks that use it in a task file.

### Changing the code generation model

```bash
# Via environment variable
export CODEMODE_CODE_MODEL=gpt-4o
python3 run_benchmark.py --agent codemode ...

# Or edit codemode_executor.py line 100:
code_model=os.environ.get("CODEMODE_CODE_MODEL", "gpt-5.2-codex"),
```

### Changing the judge model

Edit `benchmark/runner.py` — find where `_judge_provider` is created and change the model name. Currently uses `gpt-5-mini`.

### Tuning execution parameters

Edit `config/benchmark_config.yaml`:

```yaml
execution:
  task_timeout: 5000        # seconds before task is killed
  task_retry_max: 3         # retries on timeout/crash
  max_execution_rounds: 20  # max planning rounds (default agent)

benchmark:
  distraction_servers_default: 10  # distraction count when not specified
  resident_servers:
    - "Time MCP"            # always-included servers

cache:
  enabled: true
  ttl_hours: 0              # 0 = permanent

evaluation:
  judge_stability_runs: 5   # number of judge evaluations per task
```

### Tweaking codemode behavior

These are in `python-codemode/python_codemode/`:

| What | File | Line | Default |
|---|---|---|---|
| Code generation model | `codemode_executor.py` | 100 | `gpt-5.2-codex` |
| Max retries | `codemode_executor.py` | 103 | `3` |
| Sandbox timeout | `codemode_executor.py` | 104 | `120s` |
| System prompt | `generator.py` | 20-53 | Rules for code generation |
| Forbidden imports | `generator.py` | 15-18 | `os, subprocess, sys, ...` |
| Validation error detection | `proxy.py` | 99-103 | Pattern matching on result strings |
| SSE retry | `proxy.py` | 84-93 | Retry once on `BrokenResourceError` |
| Content block unwrapping | `proxy.py` | 95-101 | `list[ContentBlock]` → dict |

### Adding a new agent type

1. Create `mcp-bench/agent/my_executor.py`:
   ```python
   class MyExecutor:
       def __init__(self, llm_provider, server_manager, concurrent_summarization):
           self.llm = llm_provider
           self.server_manager = server_manager
           self.total_tokens = 0
           self.total_prompt_tokens = 0
           self.total_output_tokens = 0

       async def execute(self, task_description):
           # Your execution logic here
           # Must return:
           return {
               "solution": "final answer string",
               "total_rounds": 1,
               "execution_results": [
                   {"tool": "name", "parameters": {}, "result": {}, "status": "success"}
               ],
               "accumulated_information": "summary for judge",
               "total_output_tokens": self.total_output_tokens,
               "total_prompt_tokens": self.total_prompt_tokens,
               "total_tokens": self.total_tokens,
           }
   ```

2. Register in `benchmark/runner.py`:
   ```python
   from agent.my_executor import MyExecutor

   # In the execution section:
   if self.agent_type == "my_agent":
       executor = MyExecutor(llm_provider, conn_mgr.server_manager, ...)
   ```

3. Add to argparse choices:
   ```python
   choices=['default', 'codemode', 'my_agent']
   ```

### Filtering problematic tools

Some tools have known issues (rate limits, crashes). They're filtered by default. To include them:

```bash
python3 run_benchmark.py --disable-filter-problematic-tools
```

To add tools to the filter list, edit `config/benchmark_config.yaml`:

```yaml
execution:
  problematic_tools:
    - "Paper Search:search_semantic"
    - "My Server:buggy_tool"
```

## Task File Format

```json
{
  "generation_info": {
    "status": "completed"
  },
  "server_tasks": [
    {
      "server_name": "Time MCP",
      "tasks": [
        {
          "task_id": "time_mcp_000",
          "task_description": "Step 1: Call get_current_time with timezone='UTC'...",
          "fuzzy_description": "What time is it in different cities?",
          "dependency_analysis": "get_current_time → convert_time (needs result)",
          "distraction_servers": ["Unit Converter", "Wikipedia"]
        }
      ]
    }
  ]
}
```

### Field reference

| Field | Shown to Agent | Shown to Judge | Purpose |
|---|---|---|---|
| `task_id` | No | Yes | Unique identifier |
| `task_description` | Only if `--disable-fuzzy` | Yes (as concrete reference) | Step-by-step instructions with exact tool names, params, output format |
| `fuzzy_description` | Yes (default) | Yes | Natural language task — lets agent choose its own approach |
| `dependency_analysis` | No | Yes (reference) | Tool call dependency graph — used by judge for dependency_awareness scoring |
| `distraction_servers` | Indirectly (their tools appear) | No | Servers to add as noise — tests tool selection |

### Multi-server tasks

For tasks requiring multiple servers, use `+` in the server name:

```json
{
  "server_name": "Wikipedia+Google Maps",
  "tasks": [...]
}
```

Task files exist for different server combinations:
- `mcpbench_tasks_single_runner_format.json` — 1 server per task (56 tasks)
- `mcpbench_tasks_multi_2server_runner_format.json` — 2 servers per task
- `mcpbench_tasks_multi_3server_runner_format.json` — 3 servers per task

## Configuration Reference

### benchmark_config.yaml — key parameters

| Section | Parameter | Default | What It Controls |
|---|---|---|---|
| `execution.task_timeout` | `5000` | Seconds before a task execution is killed |
| `execution.task_retry_max` | `3` | Retries on task-level timeout/crash |
| `execution.max_execution_rounds` | `20` | Max planning rounds for default agent |
| `execution.server_semaphore_limit` | `20` | Max concurrent tool calls per server |
| `benchmark.distraction_servers_default` | `10` | Default distraction count |
| `benchmark.resident_servers` | `["Time MCP"]` | Servers always included |
| `benchmark.enable_judge_stability` | `true` | Run judge 5 times and average |
| `benchmark.use_fuzzy_descriptions` | `true` | Show fuzzy descriptions to agent |
| `benchmark.filter_problematic_tools` | `true` | Remove known buggy tools |
| `cache.enabled` | `true` | Cache tool call results |
| `cache.ttl_hours` | `0` | Cache lifetime (0 = permanent) |
| `evaluation.judge_stability_runs` | `5` | Number of judge evaluations |
| `llm.planning_tokens` | `12000` | Max tokens for planning LLM calls |

### Environment variables

| Variable | Used By | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | LLM calls + judge | Required for all runs |
| `CODEMODE_CODE_MODEL` | `codemode_executor.py` | Override code generation model |
| `GOOGLE_MAPS_API_KEY` | Google Maps MCP server | Optional |
| `NASA_API_KEY` | NASA Data MCP server | Optional |
| `NPS_API_KEY` | National Parks MCP server | Optional |
| `HF_TOKEN` | Hugging Face MCP server | Optional |
| `NCI_API_KEY` | BioMCP server | Optional |

### Available MCP servers (28 total)

Servers that need no API key: OpenAPI Explorer, Unit Converter, Wikipedia, Bibliomantic, Call for Papers, Car Price Evaluator, Context7, DEX Paprika, FruityVice, Game Trends, Huge Icons, Math MCP, NixOS, OSINT Intelligence, Reddit, Metropolitan Museum, Movie Recommender, OKX Exchange, Paper Search, Scientific Computing, Weather Data, Time MCP, Medical Calculator.

Servers that need API keys: Google Maps, NASA Data, National Parks, Hugging Face, BioMCP.
