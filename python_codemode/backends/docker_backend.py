"""Docker-based sandbox backend for Python code execution.

Runs Python code inside Docker containers with network isolation,
memory limits, and tool communication via JSON-RPC over stdin/stdout.
"""
import asyncio
import json
import tempfile
import textwrap
import time
import os
from typing import Any

from .base import ExecutionResult, SandboxBackend, parse_execution_output


# Default Docker image for code execution
DEFAULT_IMAGE = "python:3.11-slim"

# Default memory limit (128MB)
DEFAULT_MEMORY_LIMIT = "128m"


class DockerBackend(SandboxBackend):
    """Sandbox backend using Docker containers.

    Provides strong isolation by running code in ephemeral Docker containers
    with:
    - No network access (network_mode='none')
    - Memory limits (default 128MB)
    - Auto-removal of containers after execution
    - Tool communication via JSON-RPC on stdin/stdout
    """

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        network_mode: str = "none",
    ):
        """Initialize the Docker backend.

        Args:
            image: Docker image to use for execution.
            memory_limit: Memory limit for the container (e.g., '128m').
            network_mode: Docker network mode ('none' for no network).
        """
        self._image = image
        self._memory_limit = memory_limit
        self._network_mode = network_mode
        self._client = None

    def _get_client(self):
        """Lazily initialize Docker client."""
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    async def execute(self, code: str, tools: dict, timeout: int = 30) -> ExecutionResult:
        """Execute code in a Docker container.

        Creates a temporary file with wrapped code, mounts it into a
        python:3.11-slim container, and runs it with network isolation
        and memory limits. Tools are proxied via JSON-RPC over stdin/stdout.

        Args:
            code: Python code to execute.
            tools: Dict mapping tool names to callable functions.
            timeout: Maximum execution time in seconds.

        Returns:
            ExecutionResult with output, errors, and metrics.
        """
        start_time = time.monotonic()
        tool_calls = []

        # Build the wrapper script that runs inside the container
        tool_names_json = json.dumps(list(tools.keys()))
        wrapper_code = textwrap.dedent(f"""\
        import sys
        import json
        import io

        # Tool proxy: communicates with host via stdin/stdout JSON-RPC
        class ToolProxy:
            def __init__(self, name):
                self._name = name

            def __call__(self, *args, **kwargs):
                request = json.dumps({{
                    "jsonrpc": "2.0",
                    "method": self._name,
                    "params": {{"args": list(args), "kwargs": kwargs}},
                    "id": 1,
                }})
                # Write tool call request to stdout
                print("TOOL_REQUEST:" + request, flush=True)
                # Read response from stdin
                response_line = sys.stdin.readline().strip()
                if response_line:
                    response = json.loads(response_line)
                    if "error" in response:
                        raise RuntimeError(response["error"])
                    return response.get("result")
                return None

        tools = {{}}
        tool_names = json.loads('{tool_names_json}')
        for name in tool_names:
            tools[name] = ToolProxy(name)

        # Capture printed output
        _captured = []
        _orig_print = print

        def _capturing_print(*args, **kwargs):
            buf = io.StringIO()
            kwargs['file'] = buf
            _orig_print(*args, **kwargs)
            val = buf.getvalue()
            # Don't capture tool request lines
            if not val.startswith("TOOL_REQUEST:"):
                _captured.append(val)

        import builtins
        builtins.print = _capturing_print

        try:
            exec(compile(open('/sandbox/code.py').read(), '<sandbox>', 'exec'), {{
                '__builtins__': builtins,
                'tools': tools,
                'print': _capturing_print,
            }})
            output = ''.join(_captured).rstrip('\\n')
            result = {{"success": True, "output": output, "error": None}}
        except Exception as e:
            result = {{"success": False, "output": None, "error": f"{{type(e).__name__}}: {{str(e)}}"}}

        # Write final result
        _orig_print("SANDBOX_RESULT:" + json.dumps(result), flush=True)
        """)

        tmp_dir = None
        container = None
        try:
            # Create temporary directory with code files
            tmp_dir = tempfile.mkdtemp(prefix="codemode_docker_")
            code_file = os.path.join(tmp_dir, "code.py")
            wrapper_file = os.path.join(tmp_dir, "wrapper.py")

            with open(code_file, "w") as f:
                f.write(code)

            with open(wrapper_file, "w") as f:
                f.write(wrapper_code)

            # Run container using docker CLI via asyncio subprocess
            # This avoids blocking the event loop with the docker SDK
            cmd = [
                "docker", "run",
                "--rm",
                "-i",
                f"--memory={self._memory_limit}",
                f"--network={self._network_mode}",
                "--cpus=1",
                "--pids-limit=64",
                "--read-only",
                "--tmpfs=/tmp:size=32m",
                f"-v={tmp_dir}:/sandbox:ro",
                self._image,
                "python", "/sandbox/wrapper.py",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Read output and handle tool calls
            result_data = None

            async def _process_output():
                nonlocal result_data
                while True:
                    line = await proc.stdout.readline()
                    if not line:
                        break
                    line_text = line.decode().strip()

                    if line_text.startswith("TOOL_REQUEST:"):
                        # Handle tool call
                        request_json = line_text[len("TOOL_REQUEST:"):]
                        request = json.loads(request_json)
                        tool_name = request["method"]
                        params = request.get("params", {})
                        args = params.get("args", [])
                        kwargs = params.get("kwargs", {})

                        tool_calls.append({
                            "tool": tool_name,
                            "args": args,
                            "kwargs": kwargs,
                        })

                        try:
                            func = tools.get(tool_name)
                            if func is None:
                                response = {"error": f"Unknown tool: {tool_name}"}
                            elif asyncio.iscoroutinefunction(func):
                                result = await func(*args, **kwargs)
                                response = {"result": result}
                            else:
                                result = func(*args, **kwargs)
                                response = {"result": result}
                        except Exception as e:
                            response = {"error": str(e)}

                        proc.stdin.write((json.dumps(response) + "\n").encode())
                        await proc.stdin.drain()

                    elif line_text.startswith("SANDBOX_RESULT:"):
                        result_json = line_text[len("SANDBOX_RESULT:"):]
                        result_data = json.loads(result_json)

            try:
                await asyncio.wait_for(_process_output(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                duration = time.monotonic() - start_time
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds",
                    duration=duration,
                    tool_calls=tool_calls,
                )

            await proc.wait()
            duration = time.monotonic() - start_time

            if result_data:
                return ExecutionResult(
                    success=result_data["success"],
                    output=parse_execution_output(result_data["output"]) if result_data.get("output") else None,
                    error=result_data.get("error"),
                    duration=duration,
                    tool_calls=tool_calls,
                )

            # If no structured result, check stderr
            stderr_data = await proc.stderr.read()
            stderr_text = stderr_data.decode().strip()

            if proc.returncode != 0:
                return ExecutionResult(
                    success=False,
                    error=stderr_text or f"Container exited with code {proc.returncode}",
                    duration=duration,
                    tool_calls=tool_calls,
                )

            return ExecutionResult(
                success=True,
                output=None,
                duration=duration,
                tool_calls=tool_calls,
            )

        except FileNotFoundError:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                error="Docker is not installed or not in PATH",
                duration=duration,
            )
        except Exception as e:
            duration = time.monotonic() - start_time
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                duration=duration,
                tool_calls=tool_calls,
            )
        finally:
            # Clean up temp directory
            if tmp_dir:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def is_available(self) -> bool:
        """Check if Docker is available.

        Attempts to connect to the Docker daemon via the docker SDK.
        """
        try:
            import docker
            client = docker.from_env()
            client.ping()
            return True
        except Exception:
            return False

    def get_name(self) -> str:
        """Return the name of this backend."""
        return "docker"
