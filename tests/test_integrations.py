"""Tests for framework integrations."""
import pytest
import json
import sys
from unittest.mock import MagicMock, patch, AsyncMock

from tests.conftest import MockCodeMode


# --- OpenAI integration tests ---

class TestOpenAIIntegration:
    def test_as_openai_function_returns_dict(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_function
        result = as_openai_function(mock_codemode)
        assert isinstance(result, dict)

    def test_as_openai_function_has_name(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_function
        result = as_openai_function(mock_codemode)
        assert result["name"] == "codemode"

    def test_as_openai_function_has_description(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_function
        result = as_openai_function(mock_codemode)
        assert "description" in result
        assert len(result["description"]) > 0

    def test_as_openai_function_has_parameters(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_function
        result = as_openai_function(mock_codemode)
        params = result["parameters"]
        assert params["type"] == "object"
        assert "task" in params["properties"]
        assert params["properties"]["task"]["type"] == "string"
        assert "task" in params["required"]

    def test_as_openai_function_includes_tool_names(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_function
        result = as_openai_function(mock_codemode)
        assert "search_web" in result["description"]
        assert "create_event" in result["description"]

    def test_as_openai_function_no_tools(self, mock_codemode_no_tools):
        from python_codemode.integrations.openai import as_openai_function
        result = as_openai_function(mock_codemode_no_tools)
        assert "none" in result["description"]

    def test_as_openai_tool_returns_dict(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_tool
        result = as_openai_tool(mock_codemode)
        assert isinstance(result, dict)

    def test_as_openai_tool_has_type_function(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_tool
        result = as_openai_tool(mock_codemode)
        assert result["type"] == "function"

    def test_as_openai_tool_contains_function(self, mock_codemode):
        from python_codemode.integrations.openai import as_openai_tool
        result = as_openai_tool(mock_codemode)
        assert "function" in result
        assert result["function"]["name"] == "codemode"
        assert "parameters" in result["function"]

    async def test_handle_openai_call_executes_task(self, mock_codemode):
        from python_codemode.integrations.openai import handle_openai_call
        function_call = {
            "name": "codemode",
            "arguments": json.dumps({"task": "do something"}),
        }
        result = await handle_openai_call(mock_codemode, function_call)
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"]["task"] == "do something"

    async def test_handle_openai_call_empty_arguments(self, mock_codemode):
        from python_codemode.integrations.openai import handle_openai_call
        function_call = {
            "name": "codemode",
            "arguments": "{}",
        }
        result = await handle_openai_call(mock_codemode, function_call)
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"]["task"] == ""

    async def test_handle_openai_call_missing_arguments(self, mock_codemode):
        from python_codemode.integrations.openai import handle_openai_call
        function_call = {
            "name": "codemode",
        }
        result = await handle_openai_call(mock_codemode, function_call)
        parsed = json.loads(result)
        assert parsed["success"] is True


# --- Vercel integration tests ---

class TestVercelIntegration:
    def test_as_vercel_tool_returns_dict(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        result = as_vercel_tool(mock_codemode)
        assert isinstance(result, dict)

    def test_as_vercel_tool_has_description(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        result = as_vercel_tool(mock_codemode)
        assert "description" in result
        assert len(result["description"]) > 0

    def test_as_vercel_tool_has_parameters(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        result = as_vercel_tool(mock_codemode)
        params = result["parameters"]
        assert params["type"] == "object"
        assert "task" in params["properties"]
        assert params["properties"]["task"]["type"] == "string"
        assert "task" in params["required"]

    def test_as_vercel_tool_has_execute(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        result = as_vercel_tool(mock_codemode)
        assert "execute" in result
        assert callable(result["execute"])

    def test_as_vercel_tool_includes_tool_names(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        result = as_vercel_tool(mock_codemode)
        assert "search_web" in result["description"]

    def test_as_vercel_tool_no_tools(self, mock_codemode_no_tools):
        from python_codemode.integrations.vercel import as_vercel_tool
        result = as_vercel_tool(mock_codemode_no_tools)
        assert "none" in result["description"]

    async def test_as_vercel_tool_execute(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        tool = as_vercel_tool(mock_codemode)
        result = await tool["execute"]({"task": "do something"})
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"]["task"] == "do something"

    async def test_as_vercel_tool_execute_empty_task(self, mock_codemode):
        from python_codemode.integrations.vercel import as_vercel_tool
        tool = as_vercel_tool(mock_codemode)
        result = await tool["execute"]({})
        parsed = json.loads(result)
        assert parsed["success"] is True
        assert parsed["output"]["task"] == ""


# --- LangChain integration tests ---

class TestLangChainIntegration:
    def test_as_langchain_tool_import_error(self, mock_codemode):
        """Test that ImportError is raised when langchain is not available."""
        import importlib
        import python_codemode.integrations.langchain as lc_mod

        # Block only the actual langchain packages, not our own src.integrations.langchain
        blocked = {
            k: None for k in list(sys.modules.keys())
            if k.startswith("langchain") and not k.startswith("src.")
        }
        blocked.update({
            "langchain_core": None, "langchain_core.tools": None,
            "langchain": None, "langchain.tools": None,
        })

        with patch.dict(sys.modules, blocked):
            importlib.reload(lc_mod)
            with pytest.raises(ImportError, match="langchain is required"):
                lc_mod.as_langchain_tool(mock_codemode)

    def test_as_langchain_tool_with_mock(self, mock_codemode):
        """Test LangChain tool creation with mocked LangChain."""
        from python_codemode.integrations.langchain import as_langchain_tool

        # Create a mock Tool class
        mock_tool_instance = MagicMock()

        class MockTool:
            def __init__(self, **kwargs):
                self.name = kwargs.get("name")
                self.func = kwargs.get("func")
                self.coroutine = kwargs.get("coroutine")
                self.description = kwargs.get("description")

        # Create mock modules
        mock_langchain_core = MagicMock()
        mock_langchain_core_tools = MagicMock()
        mock_langchain_core_tools.Tool = MockTool

        with patch.dict(sys.modules, {
            "langchain_core": mock_langchain_core,
            "langchain_core.tools": mock_langchain_core_tools,
        }):
            tool = as_langchain_tool(mock_codemode)
            assert tool.name == "codemode"
            assert callable(tool.func)
            assert callable(tool.coroutine)
            assert "search_web" in tool.description

    def test_as_langchain_tool_description_with_tools(self, mock_codemode):
        """Test that tool names appear in description."""
        from python_codemode.integrations.langchain import as_langchain_tool

        class MockTool:
            def __init__(self, **kwargs):
                self.name = kwargs.get("name")
                self.func = kwargs.get("func")
                self.coroutine = kwargs.get("coroutine")
                self.description = kwargs.get("description")

        mock_langchain_core = MagicMock()
        mock_langchain_core_tools = MagicMock()
        mock_langchain_core_tools.Tool = MockTool

        with patch.dict(sys.modules, {
            "langchain_core": mock_langchain_core,
            "langchain_core.tools": mock_langchain_core_tools,
        }):
            tool = as_langchain_tool(mock_codemode)
            assert "search_web" in tool.description
            assert "create_event" in tool.description
            assert "create_issue" in tool.description

    def test_as_langchain_tool_description_no_tools(self, mock_codemode_no_tools):
        """Test description when no tools are available."""
        from python_codemode.integrations.langchain import as_langchain_tool

        class MockTool:
            def __init__(self, **kwargs):
                self.name = kwargs.get("name")
                self.func = kwargs.get("func")
                self.coroutine = kwargs.get("coroutine")
                self.description = kwargs.get("description")

        mock_langchain_core = MagicMock()
        mock_langchain_core_tools = MagicMock()
        mock_langchain_core_tools.Tool = MockTool

        with patch.dict(sys.modules, {
            "langchain_core": mock_langchain_core,
            "langchain_core.tools": mock_langchain_core_tools,
        }):
            tool = as_langchain_tool(mock_codemode_no_tools)
            # No tools means empty tool list in description
            assert "Available tools:" in tool.description

    def test_as_langchain_tool_sync_run(self, mock_codemode):
        """Test that the sync_run wrapper works."""
        from python_codemode.integrations.langchain import as_langchain_tool

        class MockTool:
            def __init__(self, **kwargs):
                self.name = kwargs.get("name")
                self.func = kwargs.get("func")
                self.coroutine = kwargs.get("coroutine")
                self.description = kwargs.get("description")

        mock_langchain_core = MagicMock()
        mock_langchain_core_tools = MagicMock()
        mock_langchain_core_tools.Tool = MockTool

        with patch.dict(sys.modules, {
            "langchain_core": mock_langchain_core,
            "langchain_core.tools": mock_langchain_core_tools,
        }):
            tool = as_langchain_tool(mock_codemode)
            result = tool.func("test task")
            assert "TASK COMPLETE" in result
            assert "test task" in result

    def test_as_langchain_tool_fallback_import(self, mock_codemode):
        """Test that it falls back to langchain.tools if langchain_core is not available."""
        from python_codemode.integrations.langchain import as_langchain_tool

        class MockTool:
            def __init__(self, **kwargs):
                self.name = kwargs.get("name")
                self.func = kwargs.get("func")
                self.coroutine = kwargs.get("coroutine")
                self.description = kwargs.get("description")

        # Make langchain_core fail but langchain succeed
        mock_langchain = MagicMock()
        mock_langchain_tools = MagicMock()
        mock_langchain_tools.Tool = MockTool

        # Clear cached modules
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("langchain"):
                del sys.modules[mod_name]

        with patch.dict(sys.modules, {
            "langchain_core": None,
            "langchain_core.tools": None,
            "langchain": mock_langchain,
            "langchain.tools": mock_langchain_tools,
        }):
            tool = as_langchain_tool(mock_codemode)
            assert tool.name == "codemode"


# --- Cross-integration tests ---

class TestIntegrationConsistency:
    def test_openai_and_vercel_have_same_parameters(self, mock_codemode):
        """Verify OpenAI and Vercel integrations expose the same parameter schema."""
        from python_codemode.integrations.openai import as_openai_function
        from python_codemode.integrations.vercel import as_vercel_tool

        openai_params = as_openai_function(mock_codemode)["parameters"]
        vercel_params = as_vercel_tool(mock_codemode)["parameters"]
        assert openai_params == vercel_params

    def test_all_integrations_mention_tools(self, mock_codemode):
        """All integrations should mention available tools in descriptions."""
        from python_codemode.integrations.openai import as_openai_function
        from python_codemode.integrations.vercel import as_vercel_tool

        openai_desc = as_openai_function(mock_codemode)["description"]
        vercel_desc = as_vercel_tool(mock_codemode)["description"]

        for desc in [openai_desc, vercel_desc]:
            assert "search_web" in desc

    async def test_openai_and_vercel_produce_same_result(self, mock_codemode):
        """OpenAI and Vercel should produce identical results for the same task."""
        from python_codemode.integrations.openai import handle_openai_call
        from python_codemode.integrations.vercel import as_vercel_tool

        task = "test task"

        openai_result = await handle_openai_call(
            mock_codemode,
            {"name": "codemode", "arguments": json.dumps({"task": task})}
        )

        vercel_tool = as_vercel_tool(mock_codemode)
        vercel_result = await vercel_tool["execute"]({"task": task})

        assert json.loads(openai_result) == json.loads(vercel_result)
