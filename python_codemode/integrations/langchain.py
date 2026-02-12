"""LangChain integration for python-codemode."""


def as_langchain_tool(codemode_instance, verbose: bool = True):
    """Convert a CodeMode instance to a LangChain Tool.

    Args:
        codemode_instance: A CodeMode instance with a run() method.
        verbose: Print generated code to stderr (default: True).

    Returns:
        A LangChain Tool object.

    Raises:
        ImportError: If langchain is not installed.
    """
    try:
        from langchain_core.tools import Tool
    except ImportError:
        try:
            from langchain.tools import Tool
        except ImportError:
            raise ImportError(
                "langchain is required for LangChain integration. "
                "Install it with: pip install langchain-core"
            )

    import asyncio
    import json

    cm = codemode_instance
    _verbose = verbose

    async def async_run(task: str) -> str:
        """Run codemode and return JSON results."""
        result = await cm.run(task, verbose=_verbose)
        output = result.get("output", result.get("error", "No output"))
        return (
            f"TASK COMPLETE. Results:\n"
            f"{json.dumps(output, indent=2, default=str)}"
        )

    def sync_run(task: str) -> str:
        """Synchronous wrapper for codemode.run()."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(async_run(task))
        finally:
            loop.close()

    tool_names = list(cm.tools.keys()) if hasattr(cm, 'tools') else []
    tool_list = ", ".join(tool_names[:15])
    suffix = "..." if len(tool_names) > 15 else ""
    description = (
        f"Executes Python code that calls tools to accomplish tasks. "
        f"Available tools: {tool_list}{suffix}. "
        f"Pass a clear task description. Do NOT ask the user for clarification — "
        f"just call this tool with what you know."
    )

    return Tool(
        name="codemode",
        func=sync_run,
        coroutine=async_run,
        description=description,
    )
