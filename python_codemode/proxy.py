"""Tool call routing and logging proxy."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable


class ToolNotFoundError(Exception):
    """Raised when a requested tool name is not registered."""


class ToolExecutionError(Exception):
    """Raised when a registered tool raises during invocation."""


def _json_safe(value: Any) -> Any:
    """Attempt to make *value* JSON-serialisable; fall back to repr."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


class ToolProxy:
    """Proxy layer that routes tool calls to real implementations.

    Records every invocation for later inspection and handles both sync
    and async callables transparently.

    Args:
        tools: Mapping of tool-name -> callable.
    """

    def __init__(self, tools: dict[str, Callable]) -> None:
        self._tools: dict[str, Callable] = dict(tools)
        self._call_log: list[dict[str, Any]] = []

    # ---- Core call routing ----

    async def call(self, _tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke the tool identified by *_tool_name*.

        Handles both sync and async callables. Logs the call with
        serialised args, result, timing, and success status.

        The parameter is named ``_tool_name`` (not ``name``) to avoid
        conflicts when the tool itself accepts a ``name`` kwarg.

        Raises:
            ToolNotFoundError: If *_tool_name* is not in the registered tools.
            ToolExecutionError: If the underlying tool raises an exception.
        """
        name = _tool_name
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool not found: {name}")

        fn = self._tools[name]
        start = time.monotonic()
        success = True
        result: Any = None
        error_msg: str | None = None

        # Strip None values from kwargs — LLM-generated code often passes
        # sha=None, branch=None etc. which tools reject as invalid
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        # Unwrap 'arguments' dict — LLM sometimes generates
        # tools['name'](arguments={...}) instead of tools['name'](key=val)
        if "arguments" in kwargs and isinstance(kwargs["arguments"], dict) and len(kwargs) == 1:
            kwargs = {k: v for k, v in kwargs["arguments"].items() if v is not None}
        # Also handle single positional dict arg: tools['name']({'key': 'val'})
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            kwargs = {k: v for k, v in args[0].items() if v is not None}
            args = ()

        try:
            try:
                if asyncio.iscoroutinefunction(fn):
                    result = await fn(*args, **kwargs)
                else:
                    result = fn(*args, **kwargs)
            except Exception as call_exc:
                # Retry once on SSE/transport connection errors
                exc_str = str(call_exc)
                if "TaskGroup" in exc_str or "BrokenResource" in exc_str:
                    await asyncio.sleep(0.1)
                    if asyncio.iscoroutinefunction(fn):
                        result = await fn(*args, **kwargs)
                    else:
                        result = fn(*args, **kwargs)
                else:
                    raise

            # Unwrap tuple responses — some LangChain tools return
            # (content, artifact) tuples. Extract the content part.
            if isinstance(result, tuple) and len(result) >= 1:
                result = result[0]
            # Unwrap MCP content blocks — langchain-mcp-adapters and
            # other clients return [{type:"text", text:"..."}] lists.
            if isinstance(result, list) and result:
                item = result[0]
                text = (item.get("text") if isinstance(item, dict)
                        else getattr(item, "text", None))
                if text and isinstance(text, str):
                    try:
                        result = json.loads(text)
                    except (ValueError, TypeError):
                        result = text
            # Parse JSON string responses into dicts
            if isinstance(result, str):
                try:
                    import json as _json
                    result = _json.loads(result)
                except (ValueError, TypeError):
                    pass
            # Detect MCP validation errors returned as plain strings
            if isinstance(result, str) and any(kw in result.lower() for kw in (
                "validation error", "required property", "missing required",
                "invalid argument", "is not valid",
            )):
                raise ToolExecutionError(f"Tool '{name}' validation error: {result}")
        except Exception as exc:
            success = False
            error_msg = str(exc)
            raise ToolExecutionError(
                f"Tool '{name}' raised: {exc}"
            ) from exc
        finally:
            duration = time.monotonic() - start
            self._call_log.append({
                "name": name,
                "args": _json_safe(args),
                "kwargs": _json_safe(kwargs),
                "result": _json_safe(result) if success else None,
                "error": error_msg,
                "duration": round(duration, 6),
                "success": success,
            })

        return result

    # ---- Sandbox helpers ----

    def as_sandbox_globals(self) -> dict:
        """Return a dict suitable for injecting into a sandbox's global namespace.

        The returned mapping is ``{"tool_name": wrapped_callable, ...}``.
        Each wrapped callable calls through :meth:`call` so that logging
        and error handling are applied automatically.
        """
        sandbox: dict[str, Callable] = {}
        for name in self._tools:
            # Capture *name* in closure
            async def _wrapper(*a, _n=name, **kw):
                return await self.call(_n, *a, **kw)
            _wrapper.__name__ = name
            _wrapper.__qualname__ = name
            sandbox[name] = _wrapper
        return sandbox

    # ---- Inspection ----

    def get_call_log(self) -> list[dict[str, Any]]:
        """Return the list of recorded tool call entries (copies)."""
        return list(self._call_log)

    def clear_log(self) -> None:
        """Clear the call log."""
        self._call_log.clear()

    @property
    def tool_names(self) -> list[str]:
        """List of registered tool names."""
        return list(self._tools.keys())
