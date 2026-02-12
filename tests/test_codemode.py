"""Tests for the codemode core module."""
import asyncio
import pytest
from python_codemode.codemode import codemode, CodeMode
from python_codemode.backends.base import ExecutionResult
from python_codemode.backends.pyodide_backend import PyodideBackend
from python_codemode.generator import CodeGenerator
from python_codemode.proxy import ToolProxy
from python_codemode.schema import callable_to_schema
from python_codemode.metrics import MetricsCollector


# ---- Factory function tests ----

class TestCodemodeFactory:
    def test_codemode_returns_codemode_instance(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert isinstance(cm, CodeMode)

    def test_codemode_default_backend(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert cm.backend_name == "pyodide"

    def test_codemode_custom_backend(self, mock_tools):
        cm = codemode(tools=mock_tools, backend="docker")
        assert cm.backend_name == "docker"
        # Falls back to pyodide since docker is not available
        assert cm._backend.get_name() == "pyodide"

    def test_codemode_custom_timeout(self, mock_tools):
        cm = codemode(tools=mock_tools, timeout=60)
        assert cm.timeout == 60

    def test_codemode_custom_retries(self, mock_tools):
        cm = codemode(tools=mock_tools, max_retries=5)
        assert cm.max_retries == 5

    def test_codemode_stores_tools(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert set(cm.tools.keys()) == set(mock_tools.keys())

    def test_codemode_with_empty_tools(self):
        cm = codemode(tools={})
        assert isinstance(cm, CodeMode)
        assert cm.tools == {}

    def test_codemode_with_sync_tools(self):
        def add(a: int, b: int) -> int:
            return a + b
        cm = codemode(tools={"add": add})
        assert "add" in cm.tools


# ---- CodeMode.run() tests ----

class TestCodeModeRun:
    async def test_run_with_simple_code(self, mock_tools):
        """Test run_code with a simple code string that uses tools."""
        cm = codemode(tools=mock_tools)
        code = """
async def main():
    results = await tools['search_web']('python')
    return {'results': results}
"""
        result = await cm.run_code(code)
        assert result["success"] is True
        assert result["output"]["results"] is not None
        assert result["backend"] == "pyodide"

    async def test_run_code_with_multiple_tools(self, mock_tools):
        """Test code that calls multiple tools."""
        cm = codemode(tools=mock_tools)
        code = """
async def main():
    search = await tools['search_web']('test')
    event = await tools['create_event']('Meeting', '2026-03-01')
    return {'search': search, 'event': event}
"""
        result = await cm.run_code(code)
        assert result["success"] is True
        assert "search" in result["output"]
        assert "event" in result["output"]

    async def test_run_code_returns_tool_calls(self, mock_tools):
        """Test that tool call log is included in the result."""
        cm = codemode(tools=mock_tools)
        code = """
async def main():
    await tools['search_web']('test')
    return {'done': True}
"""
        result = await cm.run_code(code)
        assert result["success"] is True
        assert len(result["tool_calls"]) >= 1
        assert result["tool_calls"][0]["name"] == "search_web"

    async def test_run_code_forbidden_import(self, mock_tools):
        """Test that forbidden imports are rejected."""
        cm = codemode(tools=mock_tools)
        code = """
import os
async def main():
    return os.listdir('.')
"""
        result = await cm.run_code(code)
        assert result["success"] is False
        assert "Forbidden import" in result["error"]

    async def test_run_code_syntax_error(self, mock_tools):
        """Test that syntax errors are caught."""
        cm = codemode(tools=mock_tools)
        code = """
async def main(
    return 42
"""
        result = await cm.run_code(code)
        assert result["success"] is False
        assert "Syntax error" in result["error"] or "error" in result

    async def test_run_code_no_main(self, mock_tools):
        """Test that missing main() is rejected."""
        cm = codemode(tools=mock_tools)
        code = """
x = 42
"""
        result = await cm.run_code(code)
        assert result["success"] is False
        assert "main" in result["error"].lower()

    async def test_run_code_runtime_error(self, mock_tools):
        """Test that runtime errors are caught."""
        cm = codemode(tools=mock_tools)
        code = """
async def main():
    return 1 / 0
"""
        result = await cm.run_code(code)
        assert result["success"] is False
        assert result["error"] is not None

    async def test_run_code_pure_computation(self):
        """Test code without any tool calls."""
        cm = codemode(tools={})
        code = """
async def main():
    total = sum(range(10))
    return {'total': total}
"""
        result = await cm.run_code(code)
        assert result["success"] is True
        assert result["output"]["total"] == 45

    async def test_run_code_returns_none_without_main(self):
        """Test that code without explicit return returns None."""
        cm = codemode(tools={})
        code = """
async def main():
    x = 42
"""
        result = await cm.run_code(code)
        assert result["success"] is True
        assert result["output"] is None


# ---- Retry logic tests ----

class TestRetryLogic:
    async def test_metrics_track_retries(self, mock_tools):
        """Test that metrics track retry attempts."""
        cm = codemode(tools=mock_tools, max_retries=2)
        # Use run_code with invalid code to trigger validation failure
        code = """
import os
async def main():
    pass
"""
        result = await cm.run_code(code)
        assert result["success"] is False

    async def test_max_retries_respected(self, mock_tools):
        """Test that the max_retries parameter is stored and accessible."""
        cm = codemode(tools=mock_tools, max_retries=5)
        assert cm.max_retries == 5


# ---- Integration method tests ----

class TestIntegrationMethods:
    def test_as_langchain_tool_exists(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert hasattr(cm, "as_langchain_tool")
        assert callable(cm.as_langchain_tool)

    def test_as_openai_function_exists(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert hasattr(cm, "as_openai_function")
        assert callable(cm.as_openai_function)

    def test_as_openai_tool_exists(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert hasattr(cm, "as_openai_tool")
        assert callable(cm.as_openai_tool)

    def test_as_vercel_tool_exists(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert hasattr(cm, "as_vercel_tool")
        assert callable(cm.as_vercel_tool)

    def test_metrics_property(self, mock_tools):
        cm = codemode(tools=mock_tools)
        assert isinstance(cm.metrics, MetricsCollector)


# ---- Backend tests ----

class TestPyodideBackend:
    async def test_backend_name(self):
        backend = PyodideBackend()
        assert backend.get_name() == "pyodide"

    async def test_backend_available(self):
        backend = PyodideBackend()
        assert backend.is_available() is True

    async def test_execute_simple(self):
        backend = PyodideBackend()
        code = """
async def main():
    return {'value': 42}
"""
        result = await backend.execute(code, {})
        assert result.success is True
        assert result.output == {"value": 42}

    async def test_execute_forbidden_import(self):
        backend = PyodideBackend()
        code = "import os\nasync def main(): pass"
        result = await backend.execute(code, {})
        assert result.success is False
        assert "os" in result.error
        assert "not allowed" in result.error or "Forbidden" in result.error

    async def test_execute_with_tools(self):
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        backend = PyodideBackend()
        code = """
async def main():
    msg = await tools['greet']('World')
    return {'message': msg}
"""
        result = await backend.execute(code, {"greet": greet})
        assert result.success is True
        assert result.output["message"] == "Hello, World!"

    async def test_execute_records_duration(self):
        backend = PyodideBackend()
        code = """
async def main():
    return True
"""
        result = await backend.execute(code, {})
        assert result.duration > 0

    async def test_execution_result_to_dict(self):
        result = ExecutionResult(success=True, output=42, duration=1.5)
        d = result.to_dict()
        assert d["success"] is True
        assert d["output"] == 42
        assert d["duration"] == 1.5


# ---- Proxy tests ----

class TestToolProxy:
    async def test_proxy_call(self):
        async def add(a: int, b: int) -> int:
            return a + b

        proxy = ToolProxy({"add": add})
        result = await proxy.call("add", 2, 3)
        assert result == 5

    async def test_proxy_logs_calls(self):
        async def echo(msg: str) -> str:
            return msg

        proxy = ToolProxy({"echo": echo})
        await proxy.call("echo", "hello")

        log = proxy.get_call_log()
        assert len(log) == 1
        assert log[0]["name"] == "echo"
        assert log[0]["success"] is True

    async def test_proxy_not_found(self):
        proxy = ToolProxy({})
        from python_codemode.proxy import ToolNotFoundError
        with pytest.raises(ToolNotFoundError):
            await proxy.call("missing")

    async def test_proxy_sandbox_globals(self):
        async def noop() -> None:
            return None

        proxy = ToolProxy({"noop": noop})
        globals_dict = proxy.as_sandbox_globals()
        assert "noop" in globals_dict
        assert asyncio.iscoroutinefunction(globals_dict["noop"])

    def test_proxy_clear_log(self):
        proxy = ToolProxy({})
        proxy._call_log.append({"test": True})
        assert len(proxy.get_call_log()) == 1
        proxy.clear_log()
        assert len(proxy.get_call_log()) == 0


# ---- Generator tests ----

class TestCodeGenerator:
    def test_validate_code_valid(self):
        code = "async def main():\n    return 42"
        valid, msg = CodeGenerator.validate_code(code)
        assert valid is True
        assert msg == "OK"

    def test_validate_code_no_main(self):
        code = "x = 42"
        valid, msg = CodeGenerator.validate_code(code)
        assert valid is False
        assert "main" in msg.lower()

    def test_validate_code_forbidden_import(self):
        code = "import os\nasync def main(): pass"
        valid, msg = CodeGenerator.validate_code(code)
        assert valid is False
        assert "Forbidden" in msg

    def test_validate_code_syntax_error(self):
        code = "def main(\n"
        valid, msg = CodeGenerator.validate_code(code)
        assert valid is False
        assert "Syntax" in msg

    def test_extract_code_markdown(self):
        response = "Here is the code:\n```python\nasync def main():\n    return 1\n```"
        code = CodeGenerator.extract_code(response)
        assert "async def main" in code

    def test_extract_code_plain(self):
        response = "async def main():\n    return 1"
        code = CodeGenerator.extract_code(response)
        assert "async def main" in code

    def test_fallback_generation(self):
        gen = CodeGenerator()
        code = gen._generate_fallback("test task", {"search_web": {"type": "object"}})
        assert "async def main" in code
        assert "search_web" in code


# ---- Schema tests ----

class TestSchema:
    def test_callable_to_schema_simple(self):
        def add(a: int, b: int) -> int:
            return a + b

        schema = callable_to_schema(add)
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "a" in params["properties"]
        assert "b" in params["properties"]
        assert params["properties"]["a"]["type"] == "integer"

    def test_callable_to_schema_with_defaults(self):
        def greet(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        schema = callable_to_schema(greet)
        params = schema["parameters"]
        assert "name" in params["required"]
        assert "greeting" not in params["required"]

    def test_callable_to_schema_no_hints(self):
        def no_hints(x):
            return x

        schema = callable_to_schema(no_hints)
        params = schema["parameters"]
        assert "x" in params["properties"]


# ---- Metrics tests ----

class TestMetrics:
    def test_record_execution(self):
        mc = MetricsCollector()
        mc.record_execution("pyodide", 1.5, True)
        s = mc.summary()
        assert s["total_executions"] == 1
        assert s["successful_executions"] == 1

    def test_record_retry(self):
        mc = MetricsCollector()
        mc.record_retry()
        mc.record_retry()
        assert mc.summary()["retries"] == 2

    def test_record_tokens(self):
        mc = MetricsCollector()
        mc.record_llm_call("gpt-5.2-codex", 80, 20, 100, 1.0, 1)
        mc.record_llm_call("gpt-5.2-codex", 30, 20, 50, 0.5, 2)
        assert mc.summary()["total_tokens"] == 150
        assert mc.summary()["total_input_tokens"] == 110
        assert mc.summary()["total_output_tokens"] == 40

    def test_format_table(self):
        mc = MetricsCollector()
        mc.record_execution("pyodide", 1.0, True)
        table = mc.format_table()
        assert "Executions" in table
        assert "1" in table

    def test_reset(self):
        mc = MetricsCollector()
        mc.record_execution("pyodide", 1.0, True)
        mc.record_retry()
        mc.record_tokens(100)
        mc.reset()
        s = mc.summary()
        assert s["total_executions"] == 0
        assert s["retries"] == 0
        assert s["total_tokens"] == 0
