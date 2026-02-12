"""Tests for src.generator -- CodeGenerator, validate_code, extract_code."""

import pytest

from python_codemode.generator import CodeGenerator, extract_code, validate_code


# ---------------------------------------------------------------------------
# extract_code
# ---------------------------------------------------------------------------


class TestExtractCode:
    def test_python_fence(self):
        response = 'Here is the code:\n```python\nprint("hello")\n```\nDone.'
        assert extract_code(response) == 'print("hello")'

    def test_bare_fence(self):
        response = 'Code:\n```\nx = 1\n```'
        assert extract_code(response) == "x = 1"

    def test_no_fence(self):
        response = "x = 1"
        assert extract_code(response) == "x = 1"

    def test_multiple_fences_picks_first_python(self):
        response = (
            "```python\nfirst = 1\n```\n"
            "and also\n"
            "```python\nsecond = 2\n```"
        )
        assert extract_code(response) == "first = 1"

    def test_multiline(self):
        response = '```python\nasync def main():\n    return {"ok": True}\n```'
        code = extract_code(response)
        assert "async def main():" in code
        assert "return" in code

    def test_whitespace_stripping(self):
        response = "   \n  x = 42\n  "
        assert extract_code(response) == "x = 42"


# ---------------------------------------------------------------------------
# validate_code
# ---------------------------------------------------------------------------


class TestValidateCode:
    def test_valid_code(self):
        code = 'async def main():\n    return {"result": 42}'
        ok, msg = validate_code(code)
        assert ok is True
        assert msg.lower() == "ok"

    def test_syntax_error(self):
        code = "def main(\n"
        ok, msg = validate_code(code)
        assert ok is False
        assert "Syntax error" in msg

    def test_forbidden_import_os(self):
        code = "import os\nasync def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "os" in msg

    def test_forbidden_import_subprocess(self):
        code = "import subprocess\nasync def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "subprocess" in msg

    def test_forbidden_import_sys(self):
        code = "import sys\nasync def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "sys" in msg

    def test_forbidden_import_shutil(self):
        code = "import shutil\nasync def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "shutil" in msg

    def test_forbidden_import_socket(self):
        code = "import socket\nasync def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "socket" in msg

    def test_forbidden_from_import(self):
        code = "from os.path import join\nasync def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "os" in msg

    def test_allowed_import(self):
        code = "import json\nasync def main():\n    return json.dumps({})"
        ok, msg = validate_code(code)
        assert ok is True

    def test_missing_async_main(self):
        code = "def main():\n    return {}"
        ok, msg = validate_code(code)
        assert ok is False
        assert "async def main()" in msg

    def test_regular_function_not_enough(self):
        code = "x = 1"
        ok, msg = validate_code(code)
        assert ok is False
        assert "main" in msg.lower()

    def test_async_main_with_tools(self):
        code = (
            "async def main():\n"
            "    result = await tools['search_web']('python')\n"
            "    return {'results': result}\n"
        )
        ok, msg = validate_code(code)
        assert ok is True


# ---------------------------------------------------------------------------
# CodeGenerator class (without hitting a real LLM)
# ---------------------------------------------------------------------------


class TestCodeGenerator:
    def test_init_defaults(self):
        gen = CodeGenerator()
        assert gen.model == "gpt-5.2-codex"
        assert gen.api_key is None

    def test_init_custom(self):
        gen = CodeGenerator(model="gpt-4", api_key="sk-test")
        assert gen.model == "gpt-4"
        assert gen.api_key == "sk-test"

    def test_validate_shortcut(self):
        gen = CodeGenerator()
        ok, msg = gen.validate('async def main():\n    return {}')
        assert ok is True

    def test_extract_shortcut(self):
        gen = CodeGenerator()
        code = gen.extract('```python\nx = 1\n```')
        assert code == "x = 1"

    async def test_generate_fallback(self):
        """generate() should fall back to template-based generation when LLM unavailable."""
        gen = CodeGenerator()
        # Without a valid API key, generate() falls back to template generation
        code = await gen.generate("say hello", {"search_web": {"type": "object"}})
        assert "async def main" in code
        assert "search_web" in code
