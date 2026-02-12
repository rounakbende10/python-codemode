"""LangChain/LangGraph agent with built-in codemode support.

Provides ``create_agent`` which returns a LangGraph ``CompiledGraph``
(the modern LangChain agent) with an optional ``codemode`` flag.

When ``codemode=True``, the tools you pass are wrapped into a single
codemode tool. The orchestrator LLM generates a task description, and
gpt-5.2-codex writes Python code that calls your tools in a sandbox.

Usage:

    from python_codemode.agent import create_agent
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-5-mini")

    # codemode=True: tools run through gpt-5.2-codex sandbox
    agent = create_agent(llm=llm, tools=mcp_tools, codemode=True)

    # codemode=False: standard LangChain direct tool calling
    agent = create_agent(llm=llm, tools=mcp_tools, codemode=False)

    # Same interface either way
    result = await agent.ainvoke({"messages": [("user", "what meetings tomorrow")]})
    print(result["messages"][-1].content)
"""

import json
from typing import Callable

from .codemode import codemode as _make_codemode


def create_agent(
    tools,
    llm=None,
    model: str = "gpt-5-mini",
    codemode: bool = True,
    code_model: str = "gpt-5.2-codex",
    api_key: str = None,
    backend: str = "pyodide",
    verbose: bool = False,
    system_prompt: str = None,
    max_retries: int = 5,
    timeout: int = 60,
):
    """Create a LangGraph react agent with a codemode flag.

    When ``codemode=True``, all tools are wrapped into a single codemode
    tool. The orchestrator LLM decides when to invoke it, and
    gpt-5.2-codex generates Python code that uses the tools in a sandbox.

    When ``codemode=False``, tools are passed directly to the agent
    (standard LangChain behavior).

    Args:
        tools: Tools for the agent. Accepts:
            - ``dict[str, callable]`` (from MCPToolLoader)
            - ``list[Tool]`` (LangChain tools)
        llm: LangChain LLM instance. If None, creates ChatOpenAI(model=model).
        model: Orchestrator model if llm not provided (default: gpt-5-mini).
        codemode: When True, tools run through codemode sandbox (default: True).
        code_model: Code generation model (default: gpt-5.2-codex).
        api_key: OpenAI API key.
        backend: Sandbox backend (default: pyodide).
        verbose: Print debug output.
        system_prompt: Custom system prompt.
        max_retries: Max code generation retries.
        timeout: Sandbox execution timeout in seconds.

    Returns:
        A LangGraph ``CompiledGraph`` (react agent).

    Example::

        # With codemode
        agent = create_agent(tools=mcp_tools, codemode=True)
        result = await agent.ainvoke(
            {"messages": [("user", "what meetings do I have tomorrow")]}
        )
        print(result["messages"][-1].content)

        # Without codemode
        agent = create_agent(tools=[search, calendar], codemode=False)
    """
    from langchain.agents import create_agent as create_react_agent

    if llm is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=model, api_key=api_key)

    tool_dict = _normalize_tools(tools)
    tool_names = list(tool_dict.keys())

    # Build tool list
    if codemode:
        lc_tools = [create_codemode_tool(
            tools=tool_dict,
            code_model=code_model,
            api_key=api_key,
            backend=backend,
            max_retries=max_retries,
            timeout=timeout,
            verbose=verbose,
        )]
    else:
        if isinstance(tools, list) and all(hasattr(t, "name") for t in tools):
            lc_tools = list(tools)
        else:
            lc_tools = [_wrap_callable(n, f) for n, f in tool_dict.items()]

    # System prompt
    if system_prompt is None:
        if codemode:
            system_prompt = (
                "You are an action-oriented assistant. You have a codemode tool "
                "that executes Python code in a sandbox with access to: "
                f"{', '.join(tool_names[:15])}"
                f"{'...' if len(tool_names) > 15 else ''}. "
                "RULES: "
                "1. ALWAYS call codemode immediately. NEVER ask clarifying questions. "
                "2. Make reasonable assumptions — if the user says 'my repos', use "
                "search_repositories. If they say 'my meetings', use list-events. "
                "3. Call codemode ONCE, then interpret the results and respond. "
                "4. Do NOT ask for tokens, credentials, or preferences. Just act."
            )
        else:
            system_prompt = (
                "You are a helpful assistant with access to tools: "
                f"{', '.join(tool_names)}. "
                "Use them to answer the user's question. "
                "Provide clear, concise answers with specific details."
            )

    agent = create_react_agent(llm, lc_tools, system_prompt=system_prompt)

    # Set recursion limit to prevent infinite loops
    agent.config = {"recursion_limit": 10}

    return agent


def create_codemode_tool(
    tools,
    code_model: str = "gpt-5.2-codex",
    api_key: str = None,
    backend: str = "pyodide",
    max_retries: int = 5,
    timeout: int = 60,
    verbose: bool = False,
):
    """Create a LangChain StructuredTool wrapping codemode.

    Returns a standard ``StructuredTool`` you can add to any agent.

    Args:
        tools: dict or list of tools to make available in the sandbox.
        code_model: Code generation model (default: gpt-5.2-codex).
        api_key: OpenAI API key.
        backend: Sandbox backend (default: pyodide).
        max_retries: Max code generation retries.
        timeout: Sandbox execution timeout in seconds.
        verbose: Print generated code to stderr.

    Returns:
        A LangChain ``StructuredTool`` named "codemode".
    """
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    tool_dict = _normalize_tools(tools)

    cm = _make_codemode(
        tools=tool_dict,
        backend=backend,
        code_model=code_model,
        api_key=api_key,
        max_retries=max_retries,
        timeout=timeout,
    )

    class CodemodeInput(BaseModel):
        task: str = Field(
            description="Detailed description of what to accomplish"
        )

    async def _run(task: str) -> str:
        result = await cm.run(task, verbose=verbose)
        # Format response so the LLM knows the task is complete
        output = result.get("output", result.get("error", "No output"))
        return (
            f"TASK COMPLETE. Results:\n"
            f"{json.dumps(output, indent=2, default=str)}\n\n"
            f"Now provide your final answer to the user based on these results."
        )

    tool_names = list(tool_dict.keys())
    tool_list = ", ".join(tool_names[:15])
    suffix = "..." if len(tool_names) > 15 else ""

    return StructuredTool(
        name="codemode",
        description=(
            "Generates and executes Python code in a secure sandbox "
            "to accomplish complex multi-step tasks. The code can call "
            f"these tools: {tool_list}{suffix}. "
            "Use this for tasks that require calling tools, combining "
            "results, or multi-step operations."
        ),
        coroutine=_run,
        args_schema=CodemodeInput,
    )


# ── Internal helpers ────────────────────────────────────────────────

def _normalize_tools(tools) -> dict[str, Callable]:
    if isinstance(tools, dict):
        return tools
    if isinstance(tools, list):
        result = {}
        for t in tools:
            # @tool-decorated functions
            if hasattr(t, '_tool_name'):
                result[t._tool_name] = t
            # LangChain Tool objects
            elif hasattr(t, "name"):
                name = t.name
                fn = getattr(t, "coroutine", None) or getattr(t, "func", None)
                if fn is None and callable(t):
                    fn = t
                if fn is not None:
                    result[name] = fn
            # Plain callables
            elif callable(t):
                result[getattr(t, '__name__', str(t))] = t
        return result
    raise TypeError(f"tools must be dict or list, got {type(tools)}")


def _wrap_callable(name: str, fn: Callable):
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class ToolInput(BaseModel):
        input: str = Field(description=f"JSON arguments for {name}")

    async def _run(input: str) -> str:
        try:
            arg = json.loads(input)
        except json.JSONDecodeError:
            arg = {"query": input}
        result = await fn(arg)
        return json.dumps(result, default=str) if not isinstance(result, str) else result

    return StructuredTool(
        name=name.replace("-", "_"),
        description=fn.__doc__ or f"Call the {name} tool",
        coroutine=_run,
        args_schema=ToolInput,
    )
