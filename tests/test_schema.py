"""Tests for src.schema -- callable_to_schema, schema_to_python_type, etc."""

import pytest

from python_codemode.schema import (
    callable_to_schema,
    format_tools_for_prompt,
    generate_dataclass,
    schema_to_python_type,
)


# ---------------------------------------------------------------------------
# callable_to_schema
# ---------------------------------------------------------------------------


class TestCallableToSchema:
    def test_basic_function(self):
        def greet(name: str, times: int = 1) -> str:
            """Say hello."""
            ...

        schema = callable_to_schema(greet)
        assert schema["name"] == "greet"
        assert schema["description"] == "Say hello."
        params = schema["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert params["properties"]["name"]["type"] == "string"
        assert params["properties"]["times"]["type"] == "integer"
        assert "name" in params["required"]
        assert "times" not in params["required"]

    def test_no_type_hints(self):
        def mystery(x, y):
            ...

        schema = callable_to_schema(mystery)
        props = schema["parameters"]["properties"]
        # Should default to string
        assert props["x"]["type"] == "string"
        assert props["y"]["type"] == "string"
        assert set(schema["parameters"]["required"]) == {"x", "y"}

    def test_complex_types(self):
        def process(items: list, data: dict) -> bool:
            """Process data."""
            ...

        schema = callable_to_schema(process)
        props = schema["parameters"]["properties"]
        assert props["items"]["type"] == "array"
        assert props["data"]["type"] == "object"

    def test_async_function(self):
        async def fetch(url: str) -> dict:
            """Fetch a URL."""
            ...

        schema = callable_to_schema(fetch)
        assert schema["name"] == "fetch"
        assert "url" in schema["parameters"]["properties"]
        assert "url" in schema["parameters"]["required"]

    def test_no_docstring(self):
        def bare(x: int):
            pass

        schema = callable_to_schema(bare)
        assert schema["description"] == ""

    def test_bool_param(self):
        def toggle(flag: bool) -> None:
            """Toggle something."""
            ...

        schema = callable_to_schema(toggle)
        assert schema["parameters"]["properties"]["flag"]["type"] == "boolean"

    def test_float_param(self):
        def compute(value: float) -> float:
            """Compute something."""
            ...

        schema = callable_to_schema(compute)
        assert schema["parameters"]["properties"]["value"]["type"] == "number"


# ---------------------------------------------------------------------------
# schema_to_python_type
# ---------------------------------------------------------------------------


class TestSchemaToPythonType:
    @pytest.mark.parametrize(
        "schema, expected",
        [
            ({"type": "string"}, "str"),
            ({"type": "integer"}, "int"),
            ({"type": "number"}, "float"),
            ({"type": "boolean"}, "bool"),
            ({"type": "array"}, "list"),
            ({"type": "object"}, "dict"),
            ({"type": "null"}, "None"),
        ],
    )
    def test_basic_types(self, schema, expected):
        assert schema_to_python_type(schema) == expected

    def test_array_with_items(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert schema_to_python_type(schema) == "list[str]"

    def test_object_with_additional_properties(self):
        schema = {"type": "object", "additionalProperties": {"type": "integer"}}
        assert schema_to_python_type(schema) == "dict[str, int]"

    def test_nested_array(self):
        schema = {
            "type": "array",
            "items": {"type": "array", "items": {"type": "integer"}},
        }
        assert schema_to_python_type(schema) == "list[list[int]]"

    def test_unknown_defaults_to_str(self):
        assert schema_to_python_type({}) == "str"


# ---------------------------------------------------------------------------
# generate_dataclass
# ---------------------------------------------------------------------------


class TestGenerateDataclass:
    def test_basic(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        code = generate_dataclass("Person", schema)
        assert "@dataclass" in code
        assert "class Person:" in code
        assert "name: str" in code
        assert "age: int" in code

    def test_optional_fields(self):
        schema = {
            "properties": {
                "name": {"type": "string"},
                "nickname": {"type": "string"},
            },
            "required": ["name"],
        }
        code = generate_dataclass("User", schema)
        assert "name: str" in code
        assert "nickname: Optional[str] = None" in code

    def test_empty_properties(self):
        schema = {"properties": {}}
        code = generate_dataclass("Empty", schema)
        assert "pass" in code

    def test_generated_code_compiles(self):
        schema = {
            "properties": {
                "title": {"type": "string"},
                "count": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        }
        code = generate_dataclass("Item", schema)
        # The generated code should be valid Python
        compile(code, "<test>", "exec")


# ---------------------------------------------------------------------------
# format_tools_for_prompt
# ---------------------------------------------------------------------------


class TestFormatToolsForPrompt:
    def test_basic_formatting(self):
        async def search(query: str) -> list:
            """Search the web."""
            ...

        async def create_event(title: str, date: str) -> dict:
            """Create a calendar event."""
            ...

        text = format_tools_for_prompt(
            {"search": search, "create_event": create_event}
        )
        assert "search" in text
        assert "create_event" in text
        assert "query" in text
        assert "title" in text
        assert "date" in text
        assert "tools[" in text

    def test_empty_tools(self):
        text = format_tools_for_prompt({})
        assert text == ""

    def test_required_optional_flags(self):
        def fn(a: int, b: str = "x"):
            """Do something."""
            ...

        text = format_tools_for_prompt({"fn": fn})
        assert "(required)" in text
        assert "(optional)" in text
