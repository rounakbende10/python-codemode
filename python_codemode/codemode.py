"""Main codemode() function - the public API for python-codemode."""
import json
import logging
import time
from typing import Callable

from .backends.pyodide_backend import PyodideBackend
from .generator import CodeGenerator
from .proxy import ToolProxy
from .schema import callable_to_schema
from .metrics import MetricsCollector

logger = logging.getLogger("codemode")


class CodeMode:
    """Main CodeMode class that wraps tools and executes LLM-generated code.

    Uses two models:
    - code_model: Generates Python code for the sandbox (default: gpt-5.2-codex)
    - model: Used for orchestration/planning if needed (default: gpt-5-mini)
    """

    def __init__(
        self,
        tools: dict[str, Callable],
        backend: str = "pyodide-wasm",
        model: str = "gpt-5-mini",
        code_model: str = "gpt-5.2-codex",
        api_key: str = None,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.tools = tools
        self.backend_name = backend
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._proxy = ToolProxy(tools)
        self._generator = CodeGenerator(model=code_model, api_key=api_key)
        self._metrics = MetricsCollector()
        self._backend = self._create_backend(backend)

        logger.info(
            "CodeMode initialized | backend=%s | code_model=%s | tools=%d | max_retries=%d | timeout=%ds",
            backend, code_model, len(tools), max_retries, timeout,
        )
        logger.debug("Available tools: %s", list(tools.keys()))

    def _create_backend(self, backend: str):
        """Create the appropriate sandbox backend."""
        if backend == "pyodide":
            return PyodideBackend()
        elif backend == "pyodide-wasm":
            try:
                from .backends.pyodide_wasm_backend import PyodideWasmBackend
                b = PyodideWasmBackend()
                if b.is_available():
                    logger.info("Using actual Pyodide WASM backend (Node.js)")
                    return b
                else:
                    logger.warning("Pyodide WASM not available (need Node.js + npm pyodide), falling back to restricted exec")
            except ImportError:
                logger.warning("Pyodide WASM backend import failed, falling back to restricted exec")
            return PyodideBackend()
        elif backend == "docker":
            try:
                from .backends.docker_backend import DockerBackend
                b = DockerBackend()
                if b.is_available():
                    return b
            except ImportError:
                pass
            return PyodideBackend()
        elif backend == "nsjail":
            try:
                from .backends.nsjail_backend import NsjailBackend
                b = NsjailBackend()
                if b.is_available():
                    return b
            except ImportError:
                pass
            return PyodideBackend()
        else:
            return PyodideBackend()

    async def run(self, task: str, verbose: bool = False) -> dict:
        """Execute a task by generating and running code.

        Self-healing loop:
        1. Generate code from task description
        2. Execute in sandbox
        3. If execution fails OR output is empty (while tools were called),
           feed the actual tool responses back to the LLM and retry
        """
        run_start = time.monotonic()
        logger.info("=" * 60)
        logger.info("RUN START | task: %s", task[:100] + ("..." if len(task) > 100 else ""))

        tool_schemas = {}
        for name, fn in self.tools.items():
            if hasattr(fn, '_mcp_schema'):
                tool_schemas[name] = fn._mcp_schema
            elif hasattr(fn, 'args_schema') and hasattr(fn, 'description'):
                # LangChain BaseTool object — extract schema from it
                tool_schemas[name] = {
                    "name": name,
                    "description": getattr(fn, "description", ""),
                    "inputSchema": fn.args_schema.model_json_schema() if fn.args_schema else {},
                }
            else:
                try:
                    tool_schemas[name] = callable_to_schema(fn)
                except (TypeError, ValueError):
                    tool_schemas[name] = {"name": name, "description": "", "parameters": {}}

        last_error = None
        last_code = None
        last_tool_results = ""
        for attempt in range(self.max_retries):
            logger.info("-" * 40)
            logger.info("ATTEMPT %d/%d", attempt + 1, self.max_retries)

            # Build prompt
            if last_error:
                logger.warning("Retrying after error: %s", last_error[:200])
                feedback = f"\n\nPREVIOUS ATTEMPT FAILED. You must fix the error.\n"
                if last_code:
                    feedback += (
                        f"\nPrevious code that failed:\n"
                        f"```python\n{last_code}\n```\n"
                    )
                feedback += f"\nError message:\n{last_error}\n"
                if last_tool_results:
                    feedback += (
                        f"\nACTUAL tool responses (use these EXACT keys):\n\n"
                        f"{last_tool_results}\n"
                    )
                feedback += (
                    "\nAnalyze the error and generate corrected code.\n"
                    "The previous code already ran — some tool calls may have "
                    "already created resources (events, repos, issues, etc). "
                    "Before creating any resource, CHECK if it already exists. "
                    "If it does, use the existing one instead of creating a duplicate."
                )
                task_with_context = task + feedback
            else:
                task_with_context = task

            # Generate code
            logger.info("GENERATE | model=%s | prompt_length=%d", self._generator.model, len(task_with_context))
            gen_start = time.monotonic()
            code = await self._generator.generate(task_with_context, tool_schemas)
            gen_duration = time.monotonic() - gen_start
            last_code = code

            # Record LLM call with token usage
            usage = getattr(self._generator, "last_usage", {})
            self._metrics.record_llm_call(
                model=self._generator.model,
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                duration=gen_duration,
                attempt=attempt + 1,
            )
            logger.info(
                "GENERATE | done in %.2fs | code_length=%d | tokens: in=%d out=%d total=%d",
                gen_duration, len(code),
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
                usage.get("total_tokens", 0),
            )
            logger.debug("Generated code:\n%s", code)

            if verbose:
                import sys
                print(f"\n--- Generated code (attempt {attempt + 1}) ---", file=sys.stderr)
                print(code, file=sys.stderr)
                print("--- end ---", file=sys.stderr)

            # Validate
            valid, msg = self._generator.validate_code(code)
            if not valid:
                logger.warning("VALIDATE | FAILED: %s", msg)
                last_error = msg
                self._metrics.record_retry()
                continue
            logger.info("VALIDATE | passed")

            # Execute
            self._proxy.clear_log()
            sandbox_tools = self._proxy.as_sandbox_globals()
            logger.info("EXECUTE | backend=%s | timeout=%ds", self._backend.get_name(), self.timeout)
            exec_start = time.monotonic()
            result = await self._backend.execute(code, sandbox_tools, timeout=self.timeout)
            exec_duration = time.monotonic() - exec_start
            call_log = self._proxy.get_call_log()
            logger.info(
                "EXECUTE | %s in %.2fs | tool_calls=%d",
                "SUCCESS" if result.success else "FAILED",
                exec_duration, len(call_log),
            )

            # Log each tool call
            for entry in call_log:
                status = "OK" if entry.get("success") else "FAIL"
                logger.info(
                    "  TOOL %s | %s | %.3fs | args=%s",
                    status, entry["name"], entry.get("duration", 0),
                    str(entry.get("kwargs", entry.get("args", "")))[:100],
                )
                if not entry.get("success") and entry.get("error"):
                    logger.warning("  TOOL ERROR | %s: %s", entry["name"], entry["error"][:200])

            self._metrics.record_execution(
                backend=self._backend.get_name(),
                duration=result.duration,
                success=result.success,
            )

            # Record tool call metrics
            for entry in call_log:
                self._metrics.record_tool_call(
                    name=entry["name"],
                    duration=entry.get("duration", 0),
                    success=entry.get("success", False),
                )

            if result.success:
                # Check for empty output
                if self._is_empty_output(result.output) and call_log:
                    logger.warning("EMPTY OUTPUT | tools called but result is empty — retrying")
                    last_error = (
                        "Code returned empty/null output but tools were called "
                        "successfully. The code is probably using wrong keys "
                        "to access the tool response data."
                    )
                    last_tool_results = self._format_tool_results(call_log)
                    self._metrics.record_retry()
                    continue

                total_duration = time.monotonic() - run_start
                logger.info("=" * 60)
                logger.info(
                    "RUN COMPLETE | attempts=%d | total=%.2fs | gen=%.2fs | exec=%.2fs | tools=%d",
                    attempt + 1, total_duration, gen_duration, exec_duration, len(call_log),
                )
                logger.info(self._metrics.format_table())

                if verbose:
                    import sys
                    print(f"\n{self._metrics.format_table()}", file=sys.stderr)

                return {
                    "success": True,
                    "output": result.output,
                    "duration": total_duration,
                    "backend": self._backend.get_name(),
                    "tool_calls": call_log,
                    "metrics": self._metrics.summary(),
                }
            else:
                logger.error("EXECUTION FAILED | %s", result.error[:200] if result.error else "unknown")
                last_error = result.error
                last_tool_results = self._format_tool_results(call_log)
                self._metrics.record_retry()

        total_duration = time.monotonic() - run_start
        logger.error(
            "RUN FAILED | all %d attempts exhausted in %.2fs | last_error: %s",
            self.max_retries, total_duration, last_error[:200] if last_error else "unknown",
        )
        logger.info(self._metrics.format_table())

        if verbose:
            import sys
            print(f"\n{self._metrics.format_table()}", file=sys.stderr)

        return {
            "success": False,
            "error": last_error,
            "retries": self.max_retries,
            "backend": self._backend.get_name(),
            "metrics": self._metrics.summary(),
        }

    @staticmethod
    def _is_empty_output(output) -> bool:
        """Check if execution output is effectively empty."""
        if output is None:
            return True
        if isinstance(output, dict):
            if not output:
                return True
            return all(
                v is None or v == [] or v == {} or v == ""
                for v in output.values()
            )
        if isinstance(output, (list, str)) and len(output) == 0:
            return True
        return False

    @staticmethod
    def _format_tool_results(call_log: list[dict]) -> str:
        """Format tool results for retry prompt, with explicit key listings."""
        snippets = []
        for entry in call_log:
            if entry.get("success") and entry.get("result") is not None:
                result = entry["result"]
                name = entry["name"]
                try:
                    result_str = json.dumps(result, indent=2, default=str)
                    if len(result_str) > 500:
                        result_str = result_str[:500] + "..."

                    if isinstance(result, dict):
                        keys = list(result.keys())
                        snippets.append(
                            f"tools['{name}']() returned dict with keys: {keys}\n"
                            f"Use EXACTLY these keys (e.g. result['{keys[0]}']).\n"
                            f"Response:\n{result_str}"
                        )
                    elif isinstance(result, list) and result:
                        sample = result[0]
                        item_keys = list(sample.keys()) if isinstance(sample, dict) else []
                        snippets.append(
                            f"tools['{name}']() returned list of {len(result)} items.\n"
                            f"Each item keys: {item_keys}\n"
                            f"Response:\n{result_str}"
                        )
                    else:
                        snippets.append(f"tools['{name}']() returned:\n{result_str}")
                except (TypeError, ValueError):
                    snippets.append(
                        f"tools['{name}']() returned: {repr(result)[:300]}"
                    )
        return "\n\n".join(snippets)

    async def run_code(self, code: str) -> dict:
        """Execute pre-written code directly (no LLM generation)."""
        logger.info("RUN_CODE | code_length=%d", len(code))
        valid, msg = self._generator.validate_code(code)
        if not valid:
            logger.warning("RUN_CODE VALIDATE | FAILED: %s", msg)
            return {
                "success": False,
                "error": msg,
                "backend": self._backend.get_name(),
            }

        sandbox_tools = self._proxy.as_sandbox_globals()
        result = await self._backend.execute(code, sandbox_tools, timeout=self.timeout)

        self._metrics.record_execution(
            backend=self._backend.get_name(),
            duration=result.duration,
            success=result.success,
        )

        logger.info("RUN_CODE | %s in %.2fs", "SUCCESS" if result.success else "FAILED", result.duration)

        if result.success:
            return {
                "success": True,
                "output": result.output,
                "duration": result.duration,
                "backend": self._backend.get_name(),
                "tool_calls": self._proxy.get_call_log(),
            }
        else:
            return {
                "success": False,
                "error": result.error,
                "backend": self._backend.get_name(),
            }

    async def batch_run(self, tasks: list[str], verbose: bool = False) -> dict:
        """Execute multiple tasks in parallel."""
        import asyncio

        logger.info("BATCH_RUN | tasks=%d", len(tasks))

        async def _run_task(i: int, task: str) -> dict:
            result = await self.run(task, verbose=verbose)
            return {"index": i, "task": task, **result}

        results = await asyncio.gather(
            *[_run_task(i, task) for i, task in enumerate(tasks)],
            return_exceptions=True,
        )

        task_results = []
        for r in results:
            if isinstance(r, Exception):
                task_results.append({"success": False, "error": str(r)})
            else:
                task_results.append(r)

        succeeded = sum(1 for t in task_results if t.get("success"))
        logger.info("BATCH_RUN | done | %d/%d succeeded", succeeded, len(tasks))

        return {
            "success": all(t.get("success") for t in task_results),
            "tasks": task_results,
            "total": len(tasks),
            "succeeded": succeeded,
        }

    def as_langchain_tool(self):
        """Return a LangChain Tool wrapping this CodeMode instance."""
        from .integrations.langchain import as_langchain_tool
        return as_langchain_tool(self)

    def as_openai_function(self) -> dict:
        """Return an OpenAI function definition."""
        from .integrations.openai import as_openai_function
        return as_openai_function(self)

    def as_openai_tool(self) -> dict:
        """Return an OpenAI tool definition."""
        from .integrations.openai import as_openai_tool
        return as_openai_tool(self)

    def as_vercel_tool(self) -> dict:
        """Return a Vercel AI SDK tool definition."""
        from .integrations.vercel import as_vercel_tool
        return as_vercel_tool(self)

    @property
    def metrics(self) -> MetricsCollector:
        """Access the metrics collector."""
        return self._metrics


def _normalize_tool_list(tools_list: list) -> dict[str, callable]:
    """Convert a list of tools (LangChain @tool, adapters, callables) to a dict."""
    result = {}
    for t in tools_list:
        if hasattr(t, "name"):
            name = t.name
            # Try multiple ways to get a callable
            fn = getattr(t, "coroutine", None) or getattr(t, "func", None)
            if fn is None and hasattr(t, "ainvoke"):
                # LangChain BaseTool / adapter — wrap ainvoke
                _tool = t
                async def _ainvoke_wrapper(_t=_tool, **kwargs):
                    return await _t.ainvoke(kwargs)
                _ainvoke_wrapper.__name__ = name
                _ainvoke_wrapper.__doc__ = getattr(t, "description", "")
                if hasattr(t, "args_schema") and t.args_schema:
                    _ainvoke_wrapper._mcp_schema = {
                        "name": name,
                        "description": getattr(t, "description", ""),
                        "inputSchema": t.args_schema.model_json_schema(),
                    }
                fn = _ainvoke_wrapper
            if fn is None:
                fn = t
            result[name] = fn
        elif callable(t):
            result[getattr(t, "__name__", str(t))] = t
    return result


def codemode(
    tools,
    backend: str = "pyodide-wasm",
    **kwargs,
) -> CodeMode:
    """Create a CodeMode instance.

    Args:
        tools: Tools for the codemode sandbox. Accepts:
            - ``dict[str, callable]`` — name-to-function mapping
            - ``list`` — list of LangChain @tool functions or plain callables
        backend: Sandbox backend to use ('pyodide', 'docker', 'nsjail').
        **kwargs: Additional options:
            model: Orchestrator model (default: gpt-5-mini)
            code_model: Code generation model (default: gpt-5.2-codex)
            api_key: OpenAI API key
            max_retries: Max retry attempts (default: 3)
            timeout: Execution timeout in seconds (default: 30)

    Returns:
        A CodeMode instance with run() method and integration helpers.

    Example::

        from langchain_core.tools import tool
        from python_codemode import codemode

        @tool
        async def search(query: str) -> list:
            '''Search the web.'''
            return [...]

        cm = codemode(tools=[search], backend='pyodide')
    """
    if isinstance(tools, list):
        tools = _normalize_tool_list(tools)

    return CodeMode(tools=tools, backend=backend, **kwargs)
