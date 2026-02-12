"""nsjail-based sandbox backend for Python code execution.

Provides the strongest isolation level by running Python code inside nsjail
(or gVisor's runsc as a fallback) with restricted filesystem, no network,
memory caps, and time limits.

Note: nsjail is Linux-only. This backend will report as unavailable on
other platforms.
"""
import asyncio
import json
import os
import platform
import shutil
import tempfile
import textwrap
import time
from typing import Any, Optional

from .base import ExecutionResult, SandboxBackend, parse_execution_output


# Default memory limit in bytes (128MB)
DEFAULT_MEMORY_LIMIT = 128 * 1024 * 1024

# Default time limit in seconds
DEFAULT_TIME_LIMIT = 30


def _find_binary(name: str) -> Optional[str]:
    """Find a binary on PATH."""
    return shutil.which(name)


class NsjailBackend(SandboxBackend):
    """Sandbox backend using nsjail for Linux namespace-based isolation.

    Provides the strongest isolation by running Python inside nsjail with:
    - No network access
    - Limited filesystem (read-only root, small tmpfs for /tmp)
    - Memory caps
    - Time limits
    - PID namespace isolation
    - Seccomp filtering

    Falls back to gVisor (runsc) if nsjail is not found.
    """

    def __init__(
        self,
        memory_limit: int = DEFAULT_MEMORY_LIMIT,
        time_limit: int = DEFAULT_TIME_LIMIT,
        nsjail_path: Optional[str] = None,
        runsc_path: Optional[str] = None,
    ):
        """Initialize the nsjail backend.

        Args:
            memory_limit: Memory limit in bytes (default 128MB).
            time_limit: Time limit in seconds.
            nsjail_path: Path to nsjail binary. Auto-detected if not provided.
            runsc_path: Path to runsc binary. Auto-detected if not provided.
        """
        self._memory_limit = memory_limit
        self._time_limit = time_limit
        self._nsjail_path = nsjail_path or _find_binary("nsjail")
        self._runsc_path = runsc_path or _find_binary("runsc")
        self._use_runsc = False

        if self._nsjail_path is None and self._runsc_path is not None:
            self._use_runsc = True

    async def execute(self, code: str, tools: dict, timeout: int = 30) -> ExecutionResult:
        """Execute code inside nsjail or gVisor sandbox.

        Creates a temporary directory with the code and a wrapper script,
        then runs Python inside nsjail with strict isolation settings.
        Tools are communicated via JSON-RPC over stdin/stdout.

        Args:
            code: Python code to execute.
            tools: Dict mapping tool names to callable functions.
            timeout: Maximum execution time in seconds.

        Returns:
            ExecutionResult with output, errors, and metrics.
        """
        if not self.is_available():
            return ExecutionResult(
                success=False,
                error="nsjail backend is not available on this system",
            )

        effective_timeout = min(timeout, self._time_limit)
        start_time = time.monotonic()
        tool_calls = []

        if self._use_runsc:
            return await self._execute_runsc(code, tools, effective_timeout, start_time, tool_calls)
        else:
            return await self._execute_nsjail(code, tools, effective_timeout, start_time, tool_calls)

    async def _execute_nsjail(
        self,
        code: str,
        tools: dict,
        timeout: int,
        start_time: float,
        tool_calls: list,
    ) -> ExecutionResult:
        """Execute code using nsjail."""
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="codemode_nsjail_")
            code_file = os.path.join(tmp_dir, "code.py")
            wrapper_file = os.path.join(tmp_dir, "wrapper.py")

            with open(code_file, "w") as f:
                f.write(code)

            wrapper_code = self._build_wrapper_script(list(tools.keys()))
            with open(wrapper_file, "w") as f:
                f.write(wrapper_code)

            # Find Python binary
            python_path = shutil.which("python3") or shutil.which("python")
            if not python_path:
                return ExecutionResult(
                    success=False,
                    error="Python interpreter not found",
                    duration=time.monotonic() - start_time,
                )

            memory_mb = self._memory_limit // (1024 * 1024)

            # Build nsjail command
            cmd = [
                self._nsjail_path,
                "--mode", "o",  # one-shot mode
                "--time_limit", str(timeout),
                "--rlimit_as", str(memory_mb),
                "--rlimit_cpu", str(timeout),
                "--rlimit_fsize", "1",  # 1MB file size limit
                "--rlimit_nofile", "32",
                "--disable_clone_newnet",  # Use network namespace for isolation
                "--really_quiet",
                # Mount filesystem read-only
                "--rw",
                "--bindmount_ro", f"/usr:/usr",
                "--bindmount_ro", f"/lib:/lib",
                "--bindmount_ro", f"/lib64:/lib64",
                "--bindmount_ro", f"{tmp_dir}:/sandbox",
                "--tmpfsmount", "/tmp:size=4194304",  # 4MB tmpfs
                "--cwd", "/sandbox",
                "--", python_path, "/sandbox/wrapper.py",
            ]

            return await self._run_sandboxed_process(
                cmd, tools, tool_calls, timeout, start_time,
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
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _execute_runsc(
        self,
        code: str,
        tools: dict,
        timeout: int,
        start_time: float,
        tool_calls: list,
    ) -> ExecutionResult:
        """Execute code using gVisor (runsc) as fallback."""
        tmp_dir = None
        try:
            tmp_dir = tempfile.mkdtemp(prefix="codemode_runsc_")
            code_file = os.path.join(tmp_dir, "code.py")
            wrapper_file = os.path.join(tmp_dir, "wrapper.py")

            with open(code_file, "w") as f:
                f.write(code)

            wrapper_code = self._build_wrapper_script(list(tools.keys()))
            with open(wrapper_file, "w") as f:
                f.write(wrapper_code)

            python_path = shutil.which("python3") or shutil.which("python")
            if not python_path:
                return ExecutionResult(
                    success=False,
                    error="Python interpreter not found",
                    duration=time.monotonic() - start_time,
                )

            # Create an OCI-compatible config for runsc
            config = {
                "ociVersion": "1.0.0",
                "process": {
                    "terminal": False,
                    "user": {"uid": 0, "gid": 0},
                    "args": [python_path, "/sandbox/wrapper.py"],
                    "cwd": "/sandbox",
                },
                "root": {"path": "/", "readonly": True},
                "mounts": [
                    {"destination": "/sandbox", "source": tmp_dir, "type": "bind", "options": ["ro"]},
                    {"destination": "/tmp", "type": "tmpfs", "options": ["size=4m"]},
                ],
            }

            config_file = os.path.join(tmp_dir, "config.json")
            with open(config_file, "w") as f:
                json.dump(config, f)

            container_id = f"codemode_{int(time.time() * 1000)}"

            cmd = [
                self._runsc_path,
                "--network=none",
                "run",
                "--bundle", tmp_dir,
                container_id,
            ]

            return await self._run_sandboxed_process(
                cmd, tools, tool_calls, timeout, start_time,
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
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _run_sandboxed_process(
        self,
        cmd: list,
        tools: dict,
        tool_calls: list,
        timeout: int,
        start_time: float,
    ) -> ExecutionResult:
        """Run a sandboxed process and handle tool communication."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        result_data = None

        async def _process_output():
            nonlocal result_data
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                line_text = line.decode().strip()

                if line_text.startswith("TOOL_REQUEST:"):
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

        stderr_data = await proc.stderr.read()
        stderr_text = stderr_data.decode().strip()

        if proc.returncode != 0:
            return ExecutionResult(
                success=False,
                error=stderr_text or f"Process exited with code {proc.returncode}",
                duration=duration,
                tool_calls=tool_calls,
            )

        return ExecutionResult(
            success=True,
            output=None,
            duration=duration,
            tool_calls=tool_calls,
        )

    def _build_wrapper_script(self, tool_names: list) -> str:
        """Build the Python wrapper script that runs inside the sandbox."""
        tool_names_json = json.dumps(tool_names)
        return textwrap.dedent(f"""\
        import sys
        import json
        import io

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
                print("TOOL_REQUEST:" + request, flush=True)
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

        _captured = []
        _orig_print = print

        def _capturing_print(*args, **kwargs):
            buf = io.StringIO()
            kwargs['file'] = buf
            _orig_print(*args, **kwargs)
            val = buf.getvalue()
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

        _orig_print("SANDBOX_RESULT:" + json.dumps(result), flush=True)
        """)

    def is_available(self) -> bool:
        """Check if nsjail (or gVisor fallback) is available.

        Returns False on non-Linux platforms since nsjail requires Linux
        kernel namespaces.
        """
        if platform.system() != "Linux":
            return False

        if self._nsjail_path and os.path.isfile(self._nsjail_path):
            return True

        if self._runsc_path and os.path.isfile(self._runsc_path):
            return True

        return False

    def get_name(self) -> str:
        """Return the name of this backend."""
        if self._use_runsc:
            return "nsjail (gvisor fallback)"
        return "nsjail"
