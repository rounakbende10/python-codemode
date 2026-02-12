"""LLM-powered Python code generation for sandbox execution."""

from __future__ import annotations

import ast
import re
import sys
import textwrap
from typing import Any, Optional

from .schema import format_tools_for_prompt


# Imports that must never appear in generated code
FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "sys", "shutil", "socket",
    "signal", "ctypes", "importlib",
})

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a code generating machine. Output ONLY Python code, nothing else.

    You are running in a SANDBOXED environment. You have NO access to:
    - Filesystem (no pathlib, no open, no glob)
    - Subprocess (no subprocess, no os.system, no asyncio.create_subprocess)
    - Network (no urllib, no requests, no http, no socket)
    - Environment variables (no os.environ)

    You can ONLY interact with the outside world through the `tools` dict.
    The tools are your ONLY way to get data. Use them.

    RULES:
    1. Write an `async def main()` function that takes no arguments.
    2. Call tools: `await tools['tool_name'](param1=value, param2=value)`
    3. The `main` function MUST return a dict with the results.
    4. Available modules: json, datetime, re, math, collections, itertools,
       functools, asyncio. No imports needed — they are pre-loaded.
    5. Use `asyncio.gather()` for parallel calls when independent.
    6. Chain tool calls — use results from one tool as input to another.
    7. Before calling a tool that needs an ID, first call a list/search tool
       to discover available IDs.

    ACCESSING TOOL RESPONSES:
    Tool responses are dicts. Discover values by type at runtime:
    - Find a list:   next((v for v in r.values() if isinstance(v, list)), [])
    - Find a string: next((v for v in r.values() if isinstance(v, str) and 'T' in v), '')
    - Find an ID:    item.get('id') or next((v for k, v in item.items() if 'id' in k.lower()), None)

    IDEMPOTENCY:
    Write code that is safe to re-run. Before creating a resource, check if
    it already exists. If a create call fails with "already exists", search
    for the existing resource and continue with it.
""")


def _generate_tool_stubs(tool_schemas: dict) -> str:
    """Generate Python-style type stubs from tool schemas.

    Mirrors Cloudflare's ``generateTypes()`` which creates TypeScript
    interfaces from tool schemas. We create Python docstring-style
    declarations so the LLM sees exactly what's available.
    """
    lines = ["# Available tools — call these using tools['name'](params)", ""]

    for name, schema in tool_schemas.items():
        desc = schema.get("description", "")
        params = schema.get("inputSchema", schema.get("parameters", {}))
        props = params.get("properties", {})
        required = set(params.get("required", []))

        # Build parameter signature
        param_parts = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "str")
            type_map = {"string": "str", "integer": "int", "number": "float",
                        "boolean": "bool", "array": "list", "object": "dict"}
            py_type = type_map.get(ptype, "Any")
            if pname in required:
                param_parts.append(f"{pname}: {py_type}")
            else:
                param_parts.append(f"{pname}: {py_type} = None")

        sig = ", ".join(param_parts) if param_parts else ""

        lines.append(f"# tools['{name}']({sig}) -> dict")
        if desc:
            lines.append(f"#   {desc[:120]}")
        lines.append("")

    return "\n".join(lines)


def _build_user_prompt(task: str, tool_schemas: dict) -> str:
    """Build the user-facing prompt with tool stubs and task."""
    stubs = _generate_tool_stubs(tool_schemas)
    tools_text = format_tools_for_prompt(tool_schemas)
    return (
        f"{stubs}\n\n"
        f"Detailed tool reference:\n{tools_text}\n\n"
        f"Task: {task}\n\n"
        f"Write an `async def main()` function that accomplishes this task "
        f"using ONLY the tools above. Return a dict with the results."
    )


class CodeGenerator:
    """Generate Python code from natural-language task descriptions using an LLM.

    Args:
        model: The model identifier (e.g. ``"gpt-4o-mini"``).
        api_key: Optional API key.  If ``None`` the underlying client should
            fall back to environment variables.
    """

    def __init__(self, model: str = "gpt-5.2-codex", api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key

    # Model to try when the primary model fails
    FALLBACK_MODEL = "gpt-4o"

    async def generate(self, task: str, tool_schemas: dict) -> str:
        """Generate Python source code for *task* using the available tools.

        This calls the LLM with a carefully crafted prompt, then extracts the
        Python code from the response.  If the primary model fails, it retries
        with a known-working fallback model (``gpt-4o``) before resorting to
        template-based generation.

        Args:
            task: Natural-language description of what the code should do.
            tool_schemas: Mapping of tool-name -> schema dict.

        Returns:
            The generated Python source code as a string.

        Raises:
            RuntimeError: If the LLM client is not available.
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            print(
                "[CodeGenerator] WARNING: openai package not installed, "
                "using template fallback.",
                file=sys.stderr,
            )
            return self._generate_fallback(task, tool_schemas)

        user_prompt = _build_user_prompt(task, tool_schemas)
        full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

        self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # --- Try primary model (structured output → clean code) ---
        try:
            code, usage = await self._call_model(self.model, full_prompt)
            self.last_usage = usage
            return extract_code(code) if "```" in code else code
        except Exception as exc:
            print(
                f"[CodeGenerator] WARNING: Primary model '{self.model}' "
                f"failed: {exc}",
                file=sys.stderr,
            )

        # --- Try fallback model ---
        if self.model != self.FALLBACK_MODEL:
            try:
                print(
                    f"[CodeGenerator] Retrying with fallback model "
                    f"'{self.FALLBACK_MODEL}'...",
                    file=sys.stderr,
                )
                code, usage = await self._call_model(self.FALLBACK_MODEL, full_prompt)
                self.last_usage = usage
                return extract_code(code) if "```" in code else code
            except Exception as exc:
                print(
                    f"[CodeGenerator] WARNING: Fallback model "
                    f"'{self.FALLBACK_MODEL}' also failed: {exc}",
                    file=sys.stderr,
                )

        # --- Last resort: template-based generation ---
        print(
            "[CodeGenerator] WARNING: All LLM calls failed, using template "
            "fallback.",
            file=sys.stderr,
        )
        return self._generate_fallback(task, tool_schemas)

    # JSON schema for structured code output (like Cloudflare's generateObject)
    _CODE_SCHEMA = {
        "type": "json_schema",
        "name": "generated_code",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code with async def main() that calls tools and returns a dict",
                }
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    }

    async def _call_model(self, model: str, prompt: str) -> tuple[str, dict]:
        """Call the OpenAI Responses API.

        Returns (code_string, usage_dict) where usage_dict has
        input_tokens, output_tokens, total_tokens.

        Tries structured output first (JSON schema with ``code`` field,
        like Cloudflare's ``generateObject``). If that fails, falls back
        to plain text generation with markdown extraction.
        """
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        def _extract_usage(response) -> dict:
            usage = getattr(response, "usage", None)
            if usage:
                return {
                    "input_tokens": getattr(usage, "input_tokens", 0),
                    "output_tokens": getattr(usage, "output_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0),
                }
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # Try structured output first
        try:
            response = await client.responses.create(
                model=model,
                input=prompt,
                text={"format": self._CODE_SCHEMA},
            )
            usage = _extract_usage(response)
            raw = response.output_text or ""
            import json as _json
            parsed = _json.loads(raw)
            code = parsed.get("code", "")
            if code and "async def main" in code:
                return code, usage
        except Exception as exc:
            print(
                f"[CodeGenerator] Structured output failed, using plain text: {exc}",
                file=sys.stderr,
            )

        # Fallback: plain text generation
        response = await client.responses.create(
            model=model,
            input=prompt,
        )
        usage = _extract_usage(response)
        return response.output_text or "", usage

    @staticmethod
    def _sanitize_var(name: str) -> str:
        """Turn a tool name into a valid Python identifier."""
        return name.replace("-", "_").replace(".", "_").replace(" ", "_")

    def _generate_fallback(self, task: str, tool_schemas: dict) -> str:
        """Simple fallback when all LLM calls fail.

        Calls the first few tools with minimal args and returns raw results.
        No use-case-specific logic — just call tools and return.
        """
        safe_task = task.split("\n")[0].replace("\\", "\\\\").replace("'", "\\'")
        tool_names = list(tool_schemas.keys())[:5]

        lines = ["async def main():"]
        lines.append("    results = {}")

        for name in tool_names:
            var = self._sanitize_var(name)
            schema = tool_schemas.get(name, {})
            params = schema.get("inputSchema", schema.get("parameters", {}))
            required = params.get("required", [])
            props = params.get("properties", {})

            # Build args: fill required params with task string or empty
            if required and props:
                first_param = required[0]
                lines.append(f"    results['{var}'] = await tools['{name}']({first_param}='{safe_task}')")
            else:
                lines.append(f"    results['{var}'] = await tools['{name}']()")

        lines.append("    return results")
        return "\n".join(lines)

    # Static/class method versions for compatibility
    @staticmethod
    def validate_code(code: str) -> tuple[bool, str]:
        """Validate generated code. Works as both static and instance method."""
        return validate_code(code)

    @staticmethod
    def extract_code(response: str) -> str:
        """Extract code from an LLM response. Works as both static and instance method."""
        return extract_code(response)

    # Instance method shortcuts (aliases)
    def validate(self, code: str) -> tuple[bool, str]:
        """Validate generated code. Shortcut for module-level ``validate_code``."""
        return validate_code(code)

    def extract(self, response: str) -> str:
        """Extract code from an LLM response. Shortcut for ``extract_code``."""
        return extract_code(response)


# ---------------------------------------------------------------------------
# Standalone helpers (also usable without instantiating CodeGenerator)
# ---------------------------------------------------------------------------

def validate_code(code: str) -> tuple[bool, str]:
    """Check generated code for syntactic correctness and forbidden imports.

    Returns:
        A tuple ``(is_valid, message)``.  ``message`` is ``"OK"`` when valid,
        or an error description otherwise.
    """
    # 1. Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    # 2. Check for forbidden imports
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_MODULES:
                    return False, f"Forbidden import: {top}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in FORBIDDEN_MODULES:
                    return False, f"Forbidden import: {top}"

    # 3. Ensure there is an async def main
    has_async_main = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "main":
            has_async_main = True
            break

    if not has_async_main:
        return False, "Missing async def main() function"

    return True, "OK"


def extract_code(response: str) -> str:
    """Extract Python code from an LLM response that may contain markdown fences.

    Looks for ```python ... ``` blocks first, then bare ``` ... ``` blocks.
    If none found, returns the response stripped of leading/trailing whitespace.
    """
    # Try ```python ... ```
    pattern_py = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
    match = pattern_py.search(response)
    if match:
        return match.group(1).strip()

    # Try bare ``` ... ```
    pattern_bare = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
    match = pattern_bare.search(response)
    if match:
        return match.group(1).strip()

    # Fallback: return as-is
    return response.strip()
