"""Vercel AI SDK integration for python-codemode."""
import json


def as_vercel_tool(codemode_instance) -> dict:
    """Convert a CodeMode instance to a Vercel AI SDK compatible tool.

    Returns a dict matching the Vercel AI SDK tool format with
    description, parameters, and execute function.
    """
    tool_names = list(codemode_instance.tools.keys()) if hasattr(codemode_instance, 'tools') else []

    async def execute(args: dict) -> str:
        """Execute the codemode tool with given arguments."""
        task = args.get("task", "")
        result = await codemode_instance.run(task)
        return json.dumps(result) if isinstance(result, dict) else str(result)

    return {
        "description": (
            f"Generates and runs Python code to complete complex multi-step tasks. "
            f"Available tools: {', '.join(tool_names) if tool_names else 'none'}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language description of the task to accomplish"
                }
            },
            "required": ["task"]
        },
        "execute": execute,
    }
