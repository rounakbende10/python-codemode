"""OpenAI function/tool integration for python-codemode."""
import json


def as_openai_function(codemode_instance) -> dict:
    """Convert a CodeMode instance to an OpenAI function definition.

    Returns the legacy 'functions' format for OpenAI API.
    """
    tool_names = list(codemode_instance.tools.keys()) if hasattr(codemode_instance, 'tools') else []

    return {
        "name": "codemode",
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
        }
    }


def as_openai_tool(codemode_instance) -> dict:
    """Convert a CodeMode instance to an OpenAI tool definition.

    Returns the newer 'tools' format for OpenAI API.
    """
    return {
        "type": "function",
        "function": as_openai_function(codemode_instance)
    }


async def handle_openai_call(codemode_instance, function_call: dict) -> str:
    """Process an OpenAI function call and return result as string.

    Args:
        codemode_instance: The CodeMode instance.
        function_call: Dict with 'name' and 'arguments' keys.

    Returns:
        JSON string of the execution result.
    """
    args = json.loads(function_call.get("arguments", "{}"))
    task = args.get("task", "")
    result = await codemode_instance.run(task)
    return json.dumps(result) if isinstance(result, dict) else str(result)
