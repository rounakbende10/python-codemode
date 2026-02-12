"""Abstract base class for sandbox backends and shared execution types."""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExecutionResult:
    """Result of executing code in a sandbox backend.

    Attributes:
        success: Whether the execution completed without error.
        output: The return value or captured output from execution.
        error: Error message if execution failed, None otherwise.
        duration: Wall-clock execution time in seconds.
        memory_used: Peak memory usage in bytes (0 if not measured).
        tool_calls: List of tool call records from the execution.
    """

    success: bool
    output: Any = None
    error: Optional[str] = None
    duration: float = 0.0
    memory_used: int = 0
    tool_calls: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a plain dictionary."""
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
            "memory_used": self.memory_used,
            "tool_calls": self.tool_calls,
        }


class TimeoutContext:
    """Async context manager that enforces a timeout on the wrapped block.

    Usage:
        async with TimeoutContext(seconds=10):
            await long_running_operation()
    """

    def __init__(self, seconds: int) -> None:
        self.seconds = seconds
        self._start: float = 0.0

    async def __aenter__(self) -> "TimeoutContext":
        self._start = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    @property
    def elapsed(self) -> float:
        """Return elapsed time since entering the context."""
        return time.monotonic() - self._start


def parse_result(raw_output: Any, error: Optional[str] = None, duration: float = 0.0, memory_used: int = 0) -> ExecutionResult:
    """Build an ExecutionResult from raw execution output.

    Args:
        raw_output: The value returned or printed by the executed code.
        error: An error string if execution failed.
        duration: Execution duration in seconds.
        memory_used: Peak memory in bytes.

    Returns:
        A properly populated ExecutionResult.
    """
    if error is not None:
        return ExecutionResult(
            success=False,
            output=None,
            error=error,
            duration=duration,
            memory_used=memory_used,
        )
    return ExecutionResult(
        success=True,
        output=raw_output,
        error=None,
        duration=duration,
        memory_used=memory_used,
    )


class TimeoutError(Exception):
    """Raised when execution exceeds timeout."""
    pass


async def run_with_timeout(coro, timeout: int) -> Any:
    """Run a coroutine with a timeout."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        raise TimeoutError(f"Execution timed out after {timeout} seconds")


def parse_execution_output(raw_output: str) -> Any:
    """Parse raw output from sandbox execution."""
    try:
        return json.loads(raw_output)
    except (json.JSONDecodeError, TypeError):
        return raw_output


class SandboxBackend(ABC):
    """Abstract base class that all sandbox backends must implement.

    A sandbox backend provides a secure environment for executing
    LLM-generated Python code with access to a set of registered tools.
    """

    @abstractmethod
    async def execute(
        self, code: str, tools: dict, timeout: int = 30
    ) -> ExecutionResult:
        """Execute Python code inside the sandbox.

        Args:
            code: The Python source code to execute.
            tools: A mapping of tool-name -> callable to inject into the sandbox.
            timeout: Maximum execution time in seconds.

        Returns:
            An ExecutionResult describing the outcome.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend is ready to execute code.

        Backends may check for required binaries, running services, etc.
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this backend."""
        ...

    async def _execute_with_timeout(
        self, coro, timeout: int
    ) -> ExecutionResult:
        """Helper: run *coro* with a wall-clock timeout.

        Returns an ExecutionResult with success=False on timeout.
        """
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(coro, timeout=timeout)
            return result
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            return ExecutionResult(
                success=False,
                output=None,
                error=f"Execution timed out after {timeout}s",
                duration=elapsed,
                memory_used=0,
            )
