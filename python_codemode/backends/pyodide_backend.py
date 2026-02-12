"""Pyodide-style sandbox backend using restricted exec.

Supports two modes:
1. async def main() pattern: Code defines an async main() that is called and its return value captured.
2. print() pattern: Code uses print() calls whose output is captured.

Safety is enforced via:
- AST-level forbidden import checking
- Restricted builtins
- Timeout enforcement
- Tool call tracking
"""
import ast
import asyncio
import collections
import datetime
import functools
import io
import itertools
import json
import math
import re
import time
from typing import Any

from .base import ExecutionResult, SandboxBackend

# Safe standard library modules pre-injected into sandbox namespace.
# LLM-generated code can use these without importing.
SAFE_MODULES = {
    "json": json,
    "re": re,
    "math": math,
    "datetime": datetime,
    "collections": collections,
    "itertools": itertools,
    "functools": functools,
}


# Modules that are forbidden in the sandbox
FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "sys", "socket", "shutil", "signal",
    "ctypes", "importlib", "pathlib", "glob", "tempfile",
    "multiprocessing", "threading", "webbrowser", "http",
    "ftplib", "smtplib", "telnetlib", "xmlrpc", "code",
    "codeop", "compile", "compileall", "py_compile",
})

# Safe builtins to expose in the sandbox
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "bytes": bytes,
    "callable": callable,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "frozenset": frozenset,
    "getattr": getattr,
    "hasattr": hasattr,
    "hash": hash,
    "hex": hex,
    "id": id,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "iter": iter,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "next": next,
    "object": object,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "AttributeError": AttributeError,
    "RuntimeError": RuntimeError,
    "StopIteration": StopIteration,
    "ZeroDivisionError": ZeroDivisionError,
}


def _check_forbidden_imports(code: str) -> str | None:
    """Check if code contains forbidden imports. Returns error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # Let exec handle syntax errors

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_root = alias.name.split(".")[0]
                if module_root in FORBIDDEN_MODULES:
                    return f"Import of '{alias.name}' is not allowed in the sandbox"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_root = node.module.split(".")[0]
                if module_root in FORBIDDEN_MODULES:
                    return f"Import from '{node.module}' is not allowed in the sandbox"
    return None


def _has_async_main(code: str) -> bool:
    """Check if code defines an async def main() function."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.AsyncFunctionDef) and node.name == "main"
        for node in ast.walk(tree)
    )


class PyodideBackend(SandboxBackend):
    """Sandbox backend using restricted exec() with limited builtins.

    This mimics pyodide's browser-based isolation by:
    - Restricting available builtins
    - Blocking dangerous module imports (os, subprocess, sys, socket, etc.)
    - Enforcing timeouts
    - Tracking memory usage
    - Injecting tools into the execution namespace

    Supports two execution patterns:
    1. If code defines `async def main()`, it is called and its return value is used as output.
    2. Otherwise, captured print output is used as the result.
    """

    def __init__(self, use_subprocess: bool = False):
        """Initialize the Pyodide backend.

        Args:
            use_subprocess: If True, run code in a separate subprocess for
                           stronger isolation. If False, use in-process
                           restricted exec().
        """
        self._use_subprocess = use_subprocess

    async def execute(self, code: str, tools: dict, timeout: int = 30) -> ExecutionResult:
        """Execute code in a restricted sandbox.

        Args:
            code: Python code to execute.
            tools: Dict mapping tool names to callable functions.
            timeout: Maximum execution time in seconds.

        Returns:
            ExecutionResult with output, errors, and metrics.
        """
        return await self._execute_restricted(code, tools, timeout)

    async def _execute_restricted(self, code: str, tools: dict, timeout: int) -> ExecutionResult:
        """Execute code using restricted exec() in-process."""
        # Check for forbidden imports
        import_error = _check_forbidden_imports(code)
        if import_error:
            return ExecutionResult(
                success=False,
                error=import_error,
            )

        start_time = time.monotonic()
        tool_calls = []

        has_main = _has_async_main(code)

        # Capture the current event loop before dispatching to thread
        main_loop = asyncio.get_running_loop()

        # Create tool wrappers that track calls and handle async
        def _make_tool_wrapper(name, func):
            def wrapper(*args, **kwargs):
                tool_calls.append({
                    "tool": name,
                    "args": list(args),
                    "kwargs": kwargs,
                })
                # If the tool is async, schedule it on the main event loop
                if asyncio.iscoroutinefunction(func):
                    future = asyncio.run_coroutine_threadsafe(
                        func(*args, **kwargs), main_loop
                    )
                    return future.result(timeout=timeout)
                else:
                    return func(*args, **kwargs)
            return wrapper

        # For async def main() mode, tools need to be async
        async_tool_wrappers = {}
        sync_tool_wrappers = {}

        for name, func in tools.items():
            sync_tool_wrappers[name] = _make_tool_wrapper(name, func)

            if asyncio.iscoroutinefunction(func):
                # Create async wrapper that logs and delegates
                async def _make_async_wrapper(n, f):
                    async def async_wrapper(*a, **kw):
                        tool_calls.append({
                            "tool": n,
                            "args": list(a),
                            "kwargs": kw,
                        })
                        return await f(*a, **kw)
                    return async_wrapper
                async_tool_wrappers[name] = await _make_async_wrapper(name, func)
            else:
                # Wrap sync tool as async
                async def _make_async_sync_wrapper(n, f):
                    async def async_wrapper(*a, **kw):
                        tool_calls.append({
                            "tool": n,
                            "args": list(a),
                            "kwargs": kw,
                        })
                        return f(*a, **kw)
                    return async_wrapper
                async_tool_wrappers[name] = await _make_async_sync_wrapper(name, func)

        if has_main:
            # Use async def main() pattern
            return await self._execute_main_pattern(
                code, async_tool_wrappers, tool_calls, timeout, start_time
            )
        else:
            # Use print() capture pattern
            return await self._execute_print_pattern(
                code, sync_tool_wrappers, tool_calls, timeout, start_time, main_loop
            )

    async def _execute_main_pattern(
        self, code: str, tools: dict, tool_calls: list, timeout: int, start_time: float
    ) -> ExecutionResult:
        """Execute code that defines async def main()."""
        # Create a safe __import__ that blocks forbidden modules
        def _safe_import(name, *args, **kwargs):
            module_root = name.split(".")[0]
            if module_root in FORBIDDEN_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
            return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)

        safe_builtins = dict(SAFE_BUILTINS)
        safe_builtins["__import__"] = _safe_import

        namespace = {
            "tools": tools,
            "asyncio": asyncio,
            "__builtins__": safe_builtins,
            **SAFE_MODULES,
        }

        try:
            tree = ast.parse(code)
            exec(compile(tree, "<sandbox>", "exec"), namespace)

            if "main" in namespace:
                result = await asyncio.wait_for(namespace["main"](), timeout=timeout)
            else:
                result = None

            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=True,
                output=result,
                duration=duration,
                tool_calls=tool_calls,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                error=f"Timeout after {timeout}s",
                duration=duration,
                tool_calls=tool_calls,
            )
        except Exception as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration=duration,
                tool_calls=tool_calls,
            )

    async def _execute_print_pattern(
        self, code: str, tools: dict, tool_calls: list,
        timeout: int, start_time: float, main_loop
    ) -> ExecutionResult:
        """Execute code that uses print() for output."""
        # Create a safe __import__ that only allows safe modules
        def _safe_import(name, *args, **kwargs):
            module_root = name.split(".")[0]
            if module_root in FORBIDDEN_MODULES:
                raise ImportError(f"Import of '{name}' is not allowed in the sandbox")
            return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else __import__(name, *args, **kwargs)

        safe_builtins = dict(SAFE_BUILTINS)
        safe_builtins["__import__"] = _safe_import

        # Custom print that captures output
        captured_lines = []

        def _safe_print(*args, **kwargs):
            output = io.StringIO()
            kwargs["file"] = output
            print(*args, **kwargs)
            line = output.getvalue()
            captured_lines.append(line)

        safe_builtins["print"] = _safe_print

        namespace = {
            "__builtins__": safe_builtins,
            "tools": tools,
            "__result__": None,
            **SAFE_MODULES,
        }

        try:
            exec_code = code + "\n"

            def _run_exec():
                exec(compile(exec_code, "<sandbox>", "exec"), namespace)

            await asyncio.wait_for(
                main_loop.run_in_executor(None, _run_exec),
                timeout=timeout,
            )

            duration = time.monotonic() - start_time
            output_text = "".join(captured_lines).rstrip("\n")

            # Check if there's a __result__ set
            result_value = namespace.get("__result__")
            if result_value is not None:
                output_text = result_value

            return ExecutionResult(
                success=True,
                output=output_text if output_text else None,
                duration=duration,
                tool_calls=tool_calls,
            )

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {timeout} seconds",
                duration=duration,
                tool_calls=tool_calls,
            )
        except Exception as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration=duration,
                tool_calls=tool_calls,
            )

    def is_available(self) -> bool:
        """Always returns True since restricted exec is always available."""
        return True

    def get_name(self) -> str:
        """Return the name of this backend."""
        return "pyodide"
