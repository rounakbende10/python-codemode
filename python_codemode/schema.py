"""JSON Schema utilities: introspect callables, convert types, generate code."""

from __future__ import annotations

import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional, get_type_hints


# ---------------------------------------------------------------------------
# Python type -> JSON Schema mapping
# ---------------------------------------------------------------------------

_PYTHON_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}

# Reverse: JSON Schema type string -> Python type string
_JSON_TO_PYTHON: dict[str, str] = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "list",
    "object": "dict",
    "null": "None",
}


def _python_type_to_json_schema(tp: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema fragment."""
    # Handle None / NoneType
    if tp is type(None):
        return {"type": "null"}

    # Basic scalar / container types
    if tp in _PYTHON_TO_JSON:
        return {"type": _PYTHON_TO_JSON[tp]}

    # typing.Optional, typing.Union, etc. from __args__
    origin = getattr(tp, "__origin__", None)

    if origin is list or origin is List:
        args = getattr(tp, "__args__", None)
        schema: dict = {"type": "array"}
        if args:
            schema["items"] = _python_type_to_json_schema(args[0])
        return schema

    if origin is dict or origin is Dict:
        args = getattr(tp, "__args__", None)
        schema = {"type": "object"}
        if args and len(args) == 2:
            schema["additionalProperties"] = _python_type_to_json_schema(args[1])
        return schema

    # typing.Optional[X] is Union[X, None]
    if origin is type(Optional[int]):  # types.UnionType on 3.10+
        pass  # fall through to Union handling below

    # Union
    import typing
    try:
        # Python 3.10+ types.UnionType (X | Y)
        import types as _types
        if isinstance(tp, _types.UnionType):
            args = tp.__args__
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return _python_type_to_json_schema(non_none[0])
            return {"anyOf": [_python_type_to_json_schema(a) for a in args]}
    except AttributeError:
        pass

    if origin is typing.Union:
        args = tp.__args__
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            # Optional[X]
            return _python_type_to_json_schema(non_none[0])
        return {"anyOf": [_python_type_to_json_schema(a) for a in args]}

    # Fallback
    return {"type": "string"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def callable_to_schema(fn: Callable) -> dict:
    """Extract a JSON Schema description from a callable's signature + type hints.

    Returns a dict shaped like a simplified OpenAI function-calling schema::

        {
            "name": "search_web",
            "description": "First line of docstring",
            "parameters": {
                "type": "object",
                "properties": { ... },
                "required": [ ... ]
            }
        }
    """
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue

        prop: dict[str, Any] = {}
        if name in hints:
            prop.update(_python_type_to_json_schema(hints[name]))
        else:
            prop["type"] = "string"  # default fallback

        # Check for default value to decide if required
        if param.default is inspect.Parameter.empty:
            required.append(name)

        properties[name] = prop

    doc = inspect.getdoc(fn) or ""
    first_line = doc.split("\n")[0].strip() if doc else ""

    return {
        "name": fn.__name__,
        "description": first_line,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def schema_to_python_type(schema: dict) -> str:
    """Convert a JSON Schema type descriptor to a Python type string.

    Supports: string, number, integer, boolean, array, object, null.
    For array with items, produces ``list[<inner>]``.
    For object with additionalProperties, produces ``dict[str, <inner>]``.
    For enum values, produces ``Literal[...]``.
    """
    # Enum values take priority — show exact valid strings
    enum = schema.get("enum")
    if enum and len(enum) <= 20:
        vals = ", ".join(repr(v) for v in enum)
        return f"Literal[{vals}]"
    elif enum:
        # Too many to inline — show type + note
        return f"str  # one of {len(enum)} values"

    json_type = schema.get("type", "string")

    if json_type == "array":
        items = schema.get("items")
        if isinstance(items, dict):
            inner = schema_to_python_type(items)
            return f"list[{inner}]"
        return "list"

    if json_type == "object":
        additional = schema.get("additionalProperties")
        if isinstance(additional, dict):
            inner = schema_to_python_type(additional)
            return f"dict[str, {inner}]"
        return "dict"

    return _JSON_TO_PYTHON.get(json_type, "str")


def generate_dataclass(name: str, schema: dict) -> str:
    """Generate Python dataclass source code from a JSON Schema object.

    Args:
        name: Class name for the generated dataclass.
        schema: A JSON Schema with ``properties`` (and optionally ``required``).

    Returns:
        A string of valid Python source code defining the dataclass.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines = [
        "from dataclasses import dataclass",
        "from typing import Optional",
        "",
        "",
        "@dataclass",
        f"class {name}:",
    ]

    if not properties:
        lines.append("    pass")
        return "\n".join(lines) + "\n"

    # Required fields first (no default), then optional ones
    required_fields: list[str] = []
    optional_fields: list[str] = []

    for prop_name, prop_schema in properties.items():
        py_type = schema_to_python_type(prop_schema)
        if prop_name in required:
            required_fields.append(f"    {prop_name}: {py_type}")
        else:
            optional_fields.append(f"    {prop_name}: Optional[{py_type}] = None")

    lines.extend(required_fields)
    lines.extend(optional_fields)

    return "\n".join(lines) + "\n"


def _sanitize_class_name(tool_name: str) -> str:
    """Convert a tool name like 'Unit Converter__convert_batch' to 'ConvertBatch'."""
    # Strip server prefix (before __)
    if "__" in tool_name:
        tool_name = tool_name.split("__", 1)[1]
    # Convert snake_case/kebab-case to PascalCase
    parts = tool_name.replace("-", "_").split("_")
    return "".join(p.capitalize() for p in parts if p)


def generate_typed_dict(name: str, schema: dict, description: str = "") -> str:
    """Generate a TypedDict class from a JSON Schema object.

    Handles nested objects by generating separate TypedDict classes.
    Returns all generated classes as a single string.
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    if not properties:
        return ""

    nested_classes: list[str] = []
    lines = []

    # Docstring with description
    if description:
        lines.append(f"class {name}(TypedDict):")
        lines.append(f'    """{description}"""')
    else:
        lines.append(f"class {name}(TypedDict):")

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            py_type = "Any"
        elif (prop_schema.get("type") == "array"
              and isinstance(prop_schema.get("items"), dict)
              and prop_schema["items"].get("type") == "object"
              and prop_schema["items"].get("properties")):
            # Nested array of objects — generate a separate TypedDict
            inner_name = f"{name}{prop_name.replace('_', ' ').title().replace(' ', '')}Item"
            inner_schema = prop_schema["items"]
            inner_desc = inner_schema.get("description", "")
            nested = generate_typed_dict(inner_name, inner_schema, inner_desc)
            if nested:
                nested_classes.append(nested)
            py_type = f"list[{inner_name}]"
        elif (prop_schema.get("type") == "object"
              and prop_schema.get("properties")):
            # Nested object — generate a separate TypedDict
            inner_name = f"{name}{prop_name.replace('_', ' ').title().replace(' ', '')}"
            inner_desc = prop_schema.get("description", "")
            nested = generate_typed_dict(inner_name, prop_schema, inner_desc)
            if nested:
                nested_classes.append(nested)
            py_type = inner_name
        else:
            py_type = schema_to_python_type(prop_schema)

        # Add field description as inline comment if available
        field_desc = prop_schema.get("description", "") if isinstance(prop_schema, dict) else ""
        comment = f"  # {field_desc}" if field_desc else ""

        if prop_name in required:
            lines.append(f"    {prop_name}: {py_type}{comment}")
        else:
            lines.append(f"    {prop_name}: NotRequired[{py_type}]{comment}")

    result = "\n".join(lines)
    if nested_classes:
        return "\n\n".join(nested_classes) + "\n\n" + result
    return result


def generate_tool_declarations(tools: dict) -> str:
    """Generate TypedDict declarations + tool signatures for all tools.

    Mirrors Cloudflare's generateTypes() — produces Python type stubs
    that the LLM uses to write correct tool calls.
    """
    type_blocks: list[str] = []
    sig_lines: list[str] = []

    for tool_name, fn_or_schema in tools.items():
        if callable(fn_or_schema):
            schema = callable_to_schema(fn_or_schema)
        elif isinstance(fn_or_schema, dict):
            schema = fn_or_schema
        else:
            schema = {"name": tool_name, "description": "", "parameters": {}}

        desc = schema.get("description", "")
        params = schema.get("inputSchema", schema.get("parameters", {}))
        props = params.get("properties", {})

        class_name = _sanitize_class_name(tool_name)

        # Generate Input TypedDict
        if props:
            input_name = f"{class_name}Input"
            input_td = generate_typed_dict(input_name, params, desc)
            if input_td:
                type_blocks.append(input_td)
            sig_lines.append(f"# tools['{tool_name}']({input_name}) -> dict")
        else:
            # No params — just show signature with description
            sig_lines.append(f"# tools['{tool_name}']() -> dict")

        if desc:
            first_line = desc.split("\n")[0].strip()
            sig_lines.append(f"#   {first_line}")
        sig_lines.append("")

    # Generate Output TypedDicts (if outputSchema exists)
    for tool_name, fn_or_schema in tools.items():
        if not isinstance(fn_or_schema, dict):
            continue
        output_schema = fn_or_schema.get("outputSchema", {})
        if output_schema.get("properties"):
            class_name = _sanitize_class_name(tool_name)
            output_name = f"{class_name}Output"
            output_td = generate_typed_dict(output_name, output_schema)
            if output_td:
                type_blocks.append(output_td)

    header = "# Type definitions for tool inputs\n"
    types_section = "\n\n".join(type_blocks) if type_blocks else ""
    sigs_section = "\n".join(sig_lines)

    if types_section:
        return f"{header}\n{types_section}\n\n# Tool signatures\n{sigs_section}"
    return f"# Tool signatures\n{sigs_section}"


def format_tools_for_prompt(tools: dict) -> str:
    """Format a dict of tools into a text block for LLM prompt injection.

    Accepts either:
    - ``{name: callable}`` -- will extract schema from each callable
    - ``{name: schema_dict}`` -- will use pre-built schemas directly

    Each tool is described with its name, description, and parameter list so the
    LLM knows how to invoke ``tools['name'](args)``.
    """
    sections: list[str] = []

    for tool_name, fn_or_schema in tools.items():
        # Determine if this is a callable or a pre-built schema dict
        if callable(fn_or_schema):
            schema = callable_to_schema(fn_or_schema)
        elif isinstance(fn_or_schema, dict):
            schema = fn_or_schema
        else:
            schema = {"name": tool_name, "description": "", "parameters": {}}

        desc = schema.get("description", "")
        # Support both MCP schemas (inputSchema) and OpenAI-style (parameters)
        params = schema.get("inputSchema", schema.get("parameters", {}))
        props = params.get("properties", {})
        req = set(params.get("required", []))

        param_lines: list[str] = []
        for pname, pschema in props.items():
            py_type = schema_to_python_type(pschema)
            req_flag = " (required)" if pname in req else " (optional)"
            param_lines.append(f"    - {pname}: {py_type}{req_flag}")

        # Build example call with param names
        if props:
            example_args = ", ".join(f"'{p}': ..." for p in list(props.keys())[:3])
            usage = f"await tools['{tool_name}']({{{{ {example_args} }}}})"
        else:
            usage = f"await tools['{tool_name}']({{{{}}}})"

        # Check for output schema (MCP tools may define this)
        output_schema = schema.get("outputSchema", {})
        output_props = output_schema.get("properties", {})

        output_lines: list[str] = []
        if output_props:
            for oname, oschema in output_props.items():
                py_type = schema_to_python_type(oschema)
                output_lines.append(f"    - {oname}: {py_type}")

        block = f"Tool: {tool_name}\n"
        if desc:
            block += f"  Description: {desc}\n"
        block += "  Input:\n"
        if param_lines:
            block += "\n".join(param_lines) + "\n"
        else:
            block += "    (none)\n"
        if output_lines:
            block += "  Output:\n"
            block += "\n".join(output_lines) + "\n"
        else:
            block += "  Output: dict (discover keys at runtime with result.keys())\n"
        block += f"  Usage: result = {usage}\n"

        sections.append(block)

    return "\n".join(sections)
