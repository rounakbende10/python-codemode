"""Tests for src.proxy -- ToolProxy, ToolNotFoundError, ToolExecutionError."""

import pytest

from python_codemode.proxy import ToolExecutionError, ToolNotFoundError, ToolProxy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sync_add(a: int, b: int) -> int:
    return a + b


async def _async_greet(name: str) -> str:
    return f"Hello, {name}!"


def _sync_fail():
    raise ValueError("intentional error")


async def _async_fail():
    raise RuntimeError("async boom")


# ---------------------------------------------------------------------------
# ToolProxy.call
# ---------------------------------------------------------------------------


class TestToolProxyCall:
    async def test_call_sync_tool(self):
        proxy = ToolProxy({"add": _sync_add})
        result = await proxy.call("add", 2, 3)
        assert result == 5

    async def test_call_async_tool(self):
        proxy = ToolProxy({"greet": _async_greet})
        result = await proxy.call("greet", "Alice")
        assert result == "Hello, Alice!"

    async def test_call_with_kwargs(self):
        proxy = ToolProxy({"add": _sync_add})
        result = await proxy.call("add", a=10, b=20)
        assert result == 30

    async def test_tool_not_found(self):
        proxy = ToolProxy({"add": _sync_add})
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            await proxy.call("nonexistent")

    async def test_sync_tool_execution_error(self):
        proxy = ToolProxy({"fail": _sync_fail})
        with pytest.raises(ToolExecutionError, match="intentional error"):
            await proxy.call("fail")

    async def test_async_tool_execution_error(self):
        proxy = ToolProxy({"fail": _async_fail})
        with pytest.raises(ToolExecutionError, match="async boom"):
            await proxy.call("fail")


# ---------------------------------------------------------------------------
# Call log
# ---------------------------------------------------------------------------


class TestToolProxyCallLog:
    async def test_log_recorded_on_success(self):
        proxy = ToolProxy({"add": _sync_add})
        await proxy.call("add", 1, 2)
        log = proxy.get_call_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["name"] == "add"
        assert entry["success"] is True
        assert entry["result"] == 3
        assert entry["error"] is None
        assert entry["duration"] >= 0

    async def test_log_recorded_on_failure(self):
        proxy = ToolProxy({"fail": _sync_fail})
        with pytest.raises(ToolExecutionError):
            await proxy.call("fail")
        log = proxy.get_call_log()
        assert len(log) == 1
        entry = log[0]
        assert entry["success"] is False
        assert entry["error"] == "intentional error"
        assert entry["result"] is None

    async def test_multiple_calls_logged(self):
        proxy = ToolProxy({"add": _sync_add, "greet": _async_greet})
        await proxy.call("add", 1, 2)
        await proxy.call("greet", "Bob")
        log = proxy.get_call_log()
        assert len(log) == 2
        assert log[0]["name"] == "add"
        assert log[1]["name"] == "greet"

    async def test_args_kwargs_serialized(self):
        proxy = ToolProxy({"add": _sync_add})
        await proxy.call("add", 5, b=10)
        entry = proxy.get_call_log()[0]
        assert entry["args"] == (5,)  # tuple is JSON-serializable, kept as-is
        assert entry["kwargs"] == {"b": 10}


# ---------------------------------------------------------------------------
# as_sandbox_globals
# ---------------------------------------------------------------------------


class TestAsSandboxGlobals:
    async def test_returns_dict_of_callables(self):
        proxy = ToolProxy({"add": _sync_add, "greet": _async_greet})
        sandbox = proxy.as_sandbox_globals()
        assert set(sandbox.keys()) == {"add", "greet"}
        assert callable(sandbox["add"])
        assert callable(sandbox["greet"])

    async def test_sandbox_callable_routes_through_proxy(self):
        proxy = ToolProxy({"add": _sync_add})
        sandbox = proxy.as_sandbox_globals()
        result = await sandbox["add"](3, 4)
        assert result == 7
        # Should also be logged
        assert len(proxy.get_call_log()) == 1

    async def test_sandbox_callable_preserves_name(self):
        proxy = ToolProxy({"greet": _async_greet})
        sandbox = proxy.as_sandbox_globals()
        assert sandbox["greet"].__name__ == "greet"


# ---------------------------------------------------------------------------
# tool_names property
# ---------------------------------------------------------------------------


class TestToolNames:
    def test_tool_names(self):
        proxy = ToolProxy({"a": _sync_add, "b": _async_greet})
        assert sorted(proxy.tool_names) == ["a", "b"]

    def test_empty(self):
        proxy = ToolProxy({})
        assert proxy.tool_names == []


# ---------------------------------------------------------------------------
# JSON serialization edge cases
# ---------------------------------------------------------------------------


class TestJsonSerialization:
    async def test_non_serializable_result_uses_repr(self):
        def returns_set() -> set:
            return {1, 2, 3}

        proxy = ToolProxy({"sets": returns_set})
        result = await proxy.call("sets")
        assert result == {1, 2, 3}
        # Log entry should have repr'd result
        entry = proxy.get_call_log()[0]
        assert isinstance(entry["result"], str)  # repr of the set

    async def test_non_serializable_args_uses_repr(self):
        def identity(x):
            return x

        proxy = ToolProxy({"id": identity})
        val = object()
        result = await proxy.call("id", val)
        assert result is val
        entry = proxy.get_call_log()[0]
        # args should be repr'd since object() is not JSON-serializable
        assert isinstance(entry["args"], str)
