"""Tests for sandbox backends: base classes, pyodide, docker, nsjail, and executor."""

import asyncio
import platform
import sys

import pytest

from python_codemode.backends.base import (
    ExecutionResult,
    SandboxBackend,
    TimeoutContext,
    TimeoutError,
    parse_result,
    run_with_timeout,
    parse_execution_output,
)
from python_codemode.backends.pyodide_backend import PyodideBackend, _check_forbidden_imports
from python_codemode.backends.docker_backend import DockerBackend
from python_codemode.backends.nsjail_backend import NsjailBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    """Check if Docker daemon is accessible."""
    try:
        import docker
        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _nsjail_available() -> bool:
    """Check if nsjail is available (Linux only)."""
    if platform.system() != "Linux":
        return False
    import shutil
    return shutil.which("nsjail") is not None or shutil.which("runsc") is not None


# ---------------------------------------------------------------------------
# ExecutionResult dataclass (agent_1 + agent_2 tests)
# ---------------------------------------------------------------------------

class TestExecutionResult:
    def test_success_result(self):
        r = ExecutionResult(success=True, output=42, duration=0.5, memory_used=1024)
        assert r.success is True
        assert r.output == 42
        assert r.error is None
        assert r.duration == 0.5
        assert r.memory_used == 1024

    def test_failure_result(self):
        r = ExecutionResult(success=False, error="boom", duration=1.0)
        assert r.success is False
        assert r.output is None
        assert r.error == "boom"

    def test_defaults(self):
        r = ExecutionResult(success=True)
        assert r.output is None
        assert r.error is None
        assert r.duration == 0.0
        assert r.memory_used == 0
        assert r.tool_calls == []

    def test_to_dict(self):
        result = ExecutionResult(
            success=True,
            output="hello",
            error=None,
            duration=0.5,
            memory_used=1024,
            tool_calls=[{"tool": "test", "args": []}],
        )
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == "hello"
        assert d["error"] is None
        assert d["duration"] == 0.5
        assert d["memory_used"] == 1024
        assert len(d["tool_calls"]) == 1

    def test_tool_calls_not_shared(self):
        """Ensure tool_calls default list is not shared between instances."""
        r1 = ExecutionResult(success=True)
        r2 = ExecutionResult(success=True)
        r1.tool_calls.append("x")
        assert r2.tool_calls == []


# ---------------------------------------------------------------------------
# parse_result helper (agent_1)
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_success(self):
        r = parse_result(raw_output={"key": "val"}, duration=0.1, memory_used=512)
        assert r.success is True
        assert r.output == {"key": "val"}
        assert r.error is None

    def test_error(self):
        r = parse_result(raw_output=None, error="fail", duration=0.2)
        assert r.success is False
        assert r.output is None
        assert r.error == "fail"


# ---------------------------------------------------------------------------
# parse_execution_output (agent_2)
# ---------------------------------------------------------------------------

class TestParseExecutionOutput:
    def test_parse_json_object(self):
        result = parse_execution_output('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_array(self):
        result = parse_execution_output('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_parse_plain_string(self):
        result = parse_execution_output("hello world")
        assert result == "hello world"

    def test_parse_none(self):
        result = parse_execution_output(None)
        assert result is None

    def test_parse_number_string(self):
        result = parse_execution_output("42")
        assert result == 42


# ---------------------------------------------------------------------------
# TimeoutContext (agent_1)
# ---------------------------------------------------------------------------

class TestTimeoutContext:
    async def test_context_tracks_elapsed(self):
        async with TimeoutContext(seconds=5) as ctx:
            await asyncio.sleep(0.05)
        assert ctx.elapsed >= 0.04

    async def test_context_does_not_swallow_exceptions(self):
        with pytest.raises(ValueError):
            async with TimeoutContext(seconds=5):
                raise ValueError("oops")


# ---------------------------------------------------------------------------
# run_with_timeout (agent_2)
# ---------------------------------------------------------------------------

class TestRunWithTimeout:
    async def test_completes_within_timeout(self):
        async def fast():
            return 42
        result = await run_with_timeout(fast(), timeout=5)
        assert result == 42

    async def test_raises_on_timeout(self):
        async def slow():
            await asyncio.sleep(10)
        with pytest.raises(TimeoutError, match="timed out after 1 seconds"):
            await run_with_timeout(slow(), timeout=1)


# ---------------------------------------------------------------------------
# SandboxBackend abstract class (agent_1)
# ---------------------------------------------------------------------------

class _DummyBackend(SandboxBackend):
    """Minimal concrete subclass for testing the ABC."""

    async def execute(self, code, tools, timeout=30):
        return ExecutionResult(success=True, output=eval(code), duration=0.0)

    def is_available(self):
        return True

    def get_name(self):
        return "dummy"


class TestSandboxBackend:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SandboxBackend()

    def test_concrete_subclass(self):
        b = _DummyBackend()
        assert b.is_available() is True
        assert b.get_name() == "dummy"

    async def test_execute(self):
        b = _DummyBackend()
        r = await b.execute("1+1", {})
        assert r.success is True
        assert r.output == 2

    async def test_execute_with_timeout_ok(self):
        async def fast_coro():
            await asyncio.sleep(0.01)
            return ExecutionResult(success=True, output="done", duration=0.01)

        b = _DummyBackend()
        r = await b._execute_with_timeout(fast_coro(), timeout=5)
        assert r.success is True
        assert r.output == "done"

    async def test_execute_with_timeout_exceeded(self):
        async def slow_coro():
            await asyncio.sleep(10)
            return ExecutionResult(success=True, output="never")

        b = _DummyBackend()
        r = await b._execute_with_timeout(slow_coro(), timeout=0.1)
        assert r.success is False
        assert "timed out" in r.error.lower()


# ---------------------------------------------------------------------------
# Pyodide Backend tests (agent_2)
# ---------------------------------------------------------------------------

class TestPyodideBackend:
    @pytest.fixture
    def backend(self):
        return PyodideBackend(use_subprocess=False)

    async def test_pyodide_basic_execution(self, backend):
        """Test basic Python code execution."""
        result = await backend.execute("print(2 + 3)", tools={})
        assert result.success is True
        assert str(result.output) == "5"
        assert result.error is None
        assert result.duration > 0

    async def test_pyodide_multiline(self, backend):
        """Test multi-line code execution."""
        code = """\
x = 10
y = 20
print(x + y)
"""
        result = await backend.execute(code, tools={})
        assert result.success is True
        assert str(result.output) == "30"

    async def test_pyodide_multiple_prints(self, backend):
        """Test capturing multiple print statements."""
        code = """\
print("hello")
print("world")
"""
        result = await backend.execute(code, tools={})
        assert result.success is True
        assert "hello" in result.output
        assert "world" in result.output

    async def test_pyodide_tool_calls(self, backend, mock_tools):
        """Test that tools can be called from sandbox code."""
        code = """\
result = tools['search_web']("python")
print(result[0]['title'])
"""
        result = await backend.execute(code, mock_tools)
        assert result.success is True
        assert "Result for python" in str(result.output)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["tool"] == "search_web"

    async def test_pyodide_tool_calls_with_kwargs(self, backend, mock_tools):
        """Test tool calls with keyword arguments."""
        code = """\
result = tools['create_event'](title="Meeting", date="2024-01-01")
print(result['title'])
"""
        result = await backend.execute(code, mock_tools)
        assert result.success is True
        assert "Meeting" in str(result.output)
        assert len(result.tool_calls) == 1

    async def test_pyodide_isolation_os(self, backend):
        """Test that os module is blocked."""
        result = await backend.execute("import os", tools={})
        assert result.success is False
        assert "os" in result.error
        assert "not allowed" in result.error

    async def test_pyodide_isolation_subprocess(self, backend):
        """Test that subprocess module is blocked."""
        result = await backend.execute("import subprocess", tools={})
        assert result.success is False
        assert "subprocess" in result.error
        assert "not allowed" in result.error

    async def test_pyodide_isolation_sys(self, backend):
        """Test that sys module is blocked."""
        result = await backend.execute("import sys", tools={})
        assert result.success is False
        assert "sys" in result.error
        assert "not allowed" in result.error

    async def test_pyodide_isolation_socket(self, backend):
        """Test that socket module is blocked."""
        result = await backend.execute("import socket", tools={})
        assert result.success is False
        assert "socket" in result.error
        assert "not allowed" in result.error

    async def test_pyodide_isolation_from_import(self, backend):
        """Test that 'from os import ...' is also blocked."""
        result = await backend.execute("from os import path", tools={})
        assert result.success is False
        assert "os" in result.error

    async def test_pyodide_syntax_error(self, backend):
        """Test that syntax errors are reported."""
        result = await backend.execute("def foo(", tools={})
        assert result.success is False
        assert "SyntaxError" in result.error

    async def test_pyodide_runtime_error(self, backend):
        """Test that runtime errors are reported."""
        result = await backend.execute("1/0", tools={})
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    async def test_pyodide_safe_imports_allowed(self, backend):
        """Test that safe modules can be imported."""
        code = """\
import json
import math
print(math.pi)
"""
        result = await backend.execute(code, tools={})
        assert result.success is True
        assert "3.14" in str(result.output)

    async def test_pyodide_no_output(self, backend):
        """Test code that produces no output."""
        result = await backend.execute("x = 42", tools={})
        assert result.success is True

    async def test_pyodide_is_available(self, backend):
        """Test that pyodide backend reports as available."""
        assert backend.is_available() is True

    async def test_pyodide_get_name(self, backend):
        """Test backend name."""
        assert backend.get_name() == "pyodide"

    async def test_pyodide_async_main_pattern(self, backend):
        """Test async def main() execution pattern."""
        code = """\
async def main():
    return {'value': 42}
"""
        result = await backend.execute(code, tools={})
        assert result.success is True
        assert result.output == {"value": 42}

    async def test_pyodide_async_main_with_tools(self, backend):
        """Test async def main() with async tool calls."""
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        code = """\
async def main():
    msg = await tools['greet']('World')
    return {'message': msg}
"""
        result = await backend.execute(code, {"greet": greet})
        assert result.success is True
        assert result.output["message"] == "Hello, World!"

    async def test_pyodide_forbidden_import_in_main(self, backend):
        """Test that forbidden imports are blocked in async main pattern."""
        code = "import os\nasync def main(): pass"
        result = await backend.execute(code, tools={})
        assert result.success is False
        assert "os" in result.error


class TestForbiddenImportChecker:
    """Tests for the _check_forbidden_imports helper."""

    def test_allows_safe_imports(self):
        assert _check_forbidden_imports("import json") is None
        assert _check_forbidden_imports("import math") is None
        assert _check_forbidden_imports("from json import dumps") is None

    def test_blocks_os(self):
        result = _check_forbidden_imports("import os")
        assert result is not None
        assert "os" in result

    def test_blocks_subprocess(self):
        result = _check_forbidden_imports("import subprocess")
        assert result is not None

    def test_blocks_from_import(self):
        result = _check_forbidden_imports("from os import path")
        assert result is not None

    def test_blocks_os_path_dotted(self):
        result = _check_forbidden_imports("import os.path")
        assert result is not None

    def test_syntax_error_passes_through(self):
        """Syntax errors should be handled by exec, not import checker."""
        assert _check_forbidden_imports("def foo(") is None


# ---------------------------------------------------------------------------
# Docker Backend tests (agent_2)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _docker_available(), reason="Docker is not available")
class TestDockerBackend:
    @pytest.fixture
    def backend(self):
        return DockerBackend()

    async def test_docker_basic_execution(self, backend):
        result = await backend.execute("print(2 + 3)", tools={})
        assert result.success is True
        assert result.output == "5"

    async def test_docker_tool_calls(self, backend):
        def add(a: int, b: int) -> int:
            return a + b

        code = """\
result = tools['add'](3, 4)
print(result)
"""
        result = await backend.execute(code, {"add": add})
        assert result.success is True
        assert result.output == 7
        assert len(result.tool_calls) == 1

    async def test_docker_isolation(self, backend):
        result = await backend.execute("print('isolated')", tools={})
        assert result.success is True
        assert result.output == "isolated"

    async def test_docker_error_handling(self, backend):
        result = await backend.execute("1/0", tools={})
        assert result.success is False
        assert "ZeroDivisionError" in result.error

    def test_docker_is_available(self, backend):
        assert backend.is_available() is True

    def test_docker_get_name(self, backend):
        assert backend.get_name() == "docker"


# ---------------------------------------------------------------------------
# nsjail Backend tests (agent_2)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _nsjail_available(), reason="nsjail is not available (requires Linux)")
class TestNsjailBackend:
    @pytest.fixture
    def backend(self):
        return NsjailBackend()

    async def test_nsjail_basic_execution(self, backend):
        result = await backend.execute("print(2 + 3)", tools={})
        assert result.success is True
        assert result.output == "5"

    def test_nsjail_get_name(self, backend):
        name = backend.get_name()
        assert "nsjail" in name


class TestNsjailAvailability:
    def test_nsjail_not_available_on_non_linux(self):
        if platform.system() != "Linux":
            backend = NsjailBackend()
            assert backend.is_available() is False

    async def test_nsjail_returns_error_when_unavailable(self):
        if platform.system() != "Linux":
            backend = NsjailBackend()
            result = await backend.execute("print('hello')", tools={})
            assert result.success is False
            assert "not available" in result.error


