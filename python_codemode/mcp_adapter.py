"""MCP (Model Context Protocol) server adapter for python-codemode.

Connects to running MCP servers and wraps their tools as async callables
that can be passed directly to codemode().

Usage with already-running SSE servers:

    from python_codemode.mcp_adapter import MCPToolLoader

    loader = MCPToolLoader()

    # Connect to MCP servers already running on localhost
    loader.add_sse_server("calendar", "http://localhost:3001/sse")
    loader.add_sse_server("serper",   "http://localhost:3002/sse")
    loader.add_sse_server("github",   "http://localhost:3003/sse")

    tools = await loader.load_tools()
    cm = codemode(tools=tools, backend='pyodide')
    result = await cm.run_code("...")

    await loader.close()

Usage with stdio servers (launches them as subprocesses):

    loader.add_stdio_server("calendar", "npx", ["-y", "google-calendar-mcp"])
"""

import asyncio
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport: str  # "stdio" or "sse"
    command: Optional[str] = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    url: Optional[str] = None  # For SSE transport


class MCPConnection:
    """Manages a connection to a single MCP server via stdio or SSE."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: Optional[asyncio.subprocess.Process] = None
        self._request_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._reader_task: Optional[asyncio.Task] = None
        self._tools: dict[str, dict] = {}
        # SSE-specific
        self._session: Optional[Any] = None  # aiohttp.ClientSession
        self._message_endpoint: Optional[str] = None

    async def connect(self):
        """Connect to the MCP server and initialize."""
        if self.config.transport == "stdio":
            await self._connect_stdio()
        elif self.config.transport == "sse":
            await self._connect_sse()
        else:
            raise ValueError(f"Unknown transport: {self.config.transport}")
        await self._initialize()

    async def _connect_stdio(self):
        """Connect via stdio subprocess."""
        env = {**os.environ, **self.config.env}
        self._process = await asyncio.create_subprocess_exec(
            self.config.command, *self.config.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        self._reader_task = asyncio.create_task(self._read_stdio_responses())

    async def _connect_sse(self):
        """Connect to an already-running MCP server via SSE.

        The MCP SSE protocol:
        1. Client opens GET to the /sse endpoint
        2. Server sends an 'endpoint' event with the URL to POST messages to
        3. Client sends JSON-RPC requests via POST to that endpoint
        4. Server sends responses back as 'message' SSE events
        """
        if not HAS_AIOHTTP:
            raise ImportError(
                "aiohttp is required for SSE connections. "
                "Install it with: pip install aiohttp"
            )

        self._session = aiohttp.ClientSession()
        sse_url = self.config.url

        # Open the SSE stream and wait for the endpoint event
        endpoint_future = asyncio.get_event_loop().create_future()

        async def sse_listener():
            try:
                async with self._session.get(sse_url) as resp:
                    if resp.status != 200:
                        endpoint_future.set_exception(
                            ConnectionError(f"SSE connection failed: HTTP {resp.status}")
                        )
                        return

                    event_type = None
                    data_lines = []

                    async for line_bytes in resp.content:
                        line = line_bytes.decode("utf-8").rstrip("\n\r")

                        if line.startswith("event:"):
                            event_type = line[len("event:"):].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[len("data:"):].strip())
                        elif line == "":
                            # End of event
                            if event_type and data_lines:
                                data = "\n".join(data_lines)
                                self._handle_sse_event(event_type, data, endpoint_future)
                            event_type = None
                            data_lines = []
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if not endpoint_future.done():
                    endpoint_future.set_exception(e)

        self._reader_task = asyncio.create_task(sse_listener())

        # Wait for the endpoint URL (with timeout)
        try:
            self._message_endpoint = await asyncio.wait_for(endpoint_future, timeout=10)
        except asyncio.TimeoutError:
            raise ConnectionError(
                f"Timed out waiting for endpoint event from {sse_url}. "
                f"Is the MCP server running and accepting SSE connections?"
            )

        # Resolve relative endpoint URLs
        if self._message_endpoint.startswith("/"):
            from urllib.parse import urlparse
            parsed = urlparse(sse_url)
            self._message_endpoint = f"{parsed.scheme}://{parsed.netloc}{self._message_endpoint}"

    def _handle_sse_event(self, event_type: str, data: str, endpoint_future: asyncio.Future):
        """Process an SSE event from the server."""
        if event_type == "endpoint":
            # Server tells us where to POST requests
            if not endpoint_future.done():
                endpoint_future.set_result(data)
        elif event_type == "message":
            # Server sends a JSON-RPC response
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                return
            if "id" in msg and msg["id"] in self._pending:
                future = self._pending.pop(msg["id"])
                if "error" in msg:
                    future.set_exception(
                        RuntimeError(f"MCP error: {msg['error'].get('message', str(msg['error']))}")
                    )
                else:
                    future.set_result(msg.get("result", {}))

    async def _initialize(self):
        """Send MCP initialize handshake."""
        result = await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "python-codemode", "version": "0.1.0"},
        })
        await self._send_notification("notifications/initialized", {})
        return result

    async def list_tools(self) -> list[dict]:
        """Get available tools from the MCP server."""
        result = await self._send_request("tools/list", {})
        tools = result.get("tools", [])
        for tool in tools:
            self._tools[tool["name"]] = tool
        return tools

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Call a tool on the MCP server.

        Raises RuntimeError if the MCP server returns an error response,
        so the sandbox catches it and the retry loop can feed it back to the LLM.
        """
        result = await self._send_request("tools/call", {
            "name": name,
            "arguments": arguments,
        })

        # Check for MCP-level error flag
        if result.get("isError"):
            content = result.get("content", [])
            error_text = ""
            for c in content:
                if c.get("type") == "text":
                    error_text += c["text"]
            raise RuntimeError(f"Tool '{name}' error: {error_text or result}")

        content = result.get("content", [])
        if not content:
            return result

        # Check if any content item is an error
        for c in content:
            if c.get("type") == "text" and c.get("text", "").startswith("MCP error"):
                raise RuntimeError(f"Tool '{name}' error: {c['text']}")

        if len(content) == 1 and content[0].get("type") == "text":
            text = content[0]["text"]
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                # Check if it looks like an error string
                if "error" in text.lower() and "invalid" in text.lower():
                    raise RuntimeError(f"Tool '{name}' error: {text}")
                return text
        return [
            json.loads(c["text"]) if c.get("type") == "text" else c
            for c in content
        ]

    async def _send_request(self, method: str, params: dict, _retry: bool = True) -> dict:
        """Send a JSON-RPC request and wait for the response.

        For SSE transport, auto-reconnects on 503 (stale session) errors.
        """
        self._request_id += 1
        request_id = self._request_id
        message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params,
        }

        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        if self.config.transport == "stdio":
            line = json.dumps(message) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        elif self.config.transport == "sse":
            async with self._session.post(
                self._message_endpoint,
                json=message,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if resp.status == 503 and _retry:
                    # SSE session expired — reconnect and retry once
                    self._pending.pop(request_id, None)
                    print(
                        f"[MCP] SSE session expired for {self.config.name}, reconnecting...",
                        file=sys.stderr,
                    )
                    await self._reconnect_sse()
                    return await self._send_request(method, params, _retry=False)
                elif resp.status not in (200, 202, 204):
                    self._pending.pop(request_id, None)
                    text = await resp.text()
                    raise RuntimeError(f"MCP POST failed ({resp.status}): {text}")

        try:
            return await asyncio.wait_for(future, timeout=30)
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise TimeoutError(f"MCP request '{method}' timed out after 30s")

    async def _reconnect_sse(self):
        """Reconnect SSE transport after session expiry."""
        # Cancel old reader
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        # Re-establish SSE connection
        await self._connect_sse()
        await self._initialize()

    async def _send_notification(self, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        if self.config.transport == "stdio":
            line = json.dumps(message) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
        elif self.config.transport == "sse":
            async with self._session.post(
                self._message_endpoint,
                json=message,
                headers={"Content-Type": "application/json"},
            ) as resp:
                pass  # Notifications don't expect responses

    async def _read_stdio_responses(self):
        """Read JSON-RPC responses from stdio stdout."""
        try:
            while True:
                line = await self._process.stdout.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line.decode().strip())
                except json.JSONDecodeError:
                    continue
                if "id" in msg and msg["id"] in self._pending:
                    future = self._pending.pop(msg["id"])
                    if "error" in msg:
                        future.set_exception(
                            RuntimeError(f"MCP error: {msg['error'].get('message', str(msg['error']))}")
                        )
                    else:
                        future.set_result(msg.get("result", {}))
        except asyncio.CancelledError:
            pass

    async def close(self):
        """Shut down the MCP server connection."""
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        if self._process:
            self._process.stdin.close()
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except (asyncio.TimeoutError, ProcessLookupError):
                self._process.kill()
        if self._session:
            await self._session.close()


class MCPToolLoader:
    """Load tools from multiple MCP servers and expose them as async callables.

    Example:
        loader = MCPToolLoader()
        loader.add_stdio_server("calendar", "npx", ["-y", "google-calendar-mcp"])
        loader.add_stdio_server("github", "npx", ["-y", "github-mcp"],
                                env={"GITHUB_TOKEN": os.environ["GITHUB_TOKEN"]})

        tools = await loader.load_tools()
        cm = codemode(tools=tools, backend='pyodide')
    """

    def __init__(self):
        self._configs: list[MCPServerConfig] = []
        self._connections: list[MCPConnection] = []
        self._tools: dict[str, Callable] = {}

    def add_stdio_server(
        self,
        name: str,
        command: str,
        args: list[str] = None,
        env: dict[str, str] = None,
    ):
        """Register a stdio-based MCP server.

        Args:
            name: A label for this server (used to prefix tool names if conflicts).
            command: The command to launch the server (e.g., "npx", "node", "python").
            args: Command-line arguments.
            env: Additional environment variables (merged with current env).
        """
        self._configs.append(MCPServerConfig(
            name=name,
            transport="stdio",
            command=command,
            args=args or [],
            env=env or {},
        ))

    def add_sse_server(self, name: str, url: str):
        """Register an SSE-based MCP server.

        Args:
            name: A label for this server.
            url: The SSE endpoint URL.
        """
        self._configs.append(MCPServerConfig(
            name=name,
            transport="sse",
            url=url,
        ))

    async def load_tools(self, prefix_with_server: bool = False) -> dict[str, Callable]:
        """Connect to all servers and load their tools.

        Args:
            prefix_with_server: If True, prefix tool names with server name
                                to avoid conflicts (e.g., "calendar_create_event").

        Returns:
            Dict mapping tool names to async callables.
        """
        for config in self._configs:
            conn = MCPConnection(config)
            try:
                await conn.connect()
                tools = await conn.list_tools()
                self._connections.append(conn)

                for tool_def in tools:
                    tool_name = tool_def["name"]
                    if prefix_with_server:
                        tool_name = f"{config.name}_{tool_name}"

                    # Build the async callable wrapper
                    self._tools[tool_name] = self._make_tool_callable(
                        conn, tool_def
                    )
            except Exception as e:
                print(f"Warning: Failed to connect to MCP server '{config.name}': {e}",
                      file=sys.stderr)
                await conn.close()

        return dict(self._tools)

    @staticmethod
    def _normalize_arg(key: str, value: Any, prop_schema: dict) -> Any:
        """Normalize an argument value based on its schema.

        Handles common LLM mistakes like appending timezone names to datetimes.
        """
        if not isinstance(value, str):
            return value

        # If the schema describes a date/time format, or the key suggests it,
        # strip timezone names/offsets that LLMs commonly append
        fmt = prop_schema.get("format", "")
        is_time_field = (
            "date-time" in fmt
            or "date" in fmt
            or key.lower() in ("timemin", "timemax", "start", "end", "date", "starttime", "endtime")
        )
        if is_time_field and "T" in value:
            import re as _re
            # Strip trailing timezone name like "America/New_York"
            value = _re.sub(r'[A-Z][a-z]+/[A-Z][a-z_]+$', '', value)
            # Strip trailing offset like "-05:00" or "+00:00" AFTER the seconds
            # Match: 2026-02-11T00:00:00-05:00 → 2026-02-11T00:00:00
            match = _re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)', value)
            if match:
                value = match.group(1)
            # Strip trailing 'Z'
            value = value.rstrip('Z')

        return value

    def _make_tool_callable(self, conn: MCPConnection, tool_def: dict) -> Callable:
        """Create an async callable that invokes an MCP tool.

        The wrapper accepts any calling convention the LLM might generate:
          - tools['name']({'param': value})       # single dict arg
          - tools['name'](param=value)            # keyword args
          - tools['name']('positional', 'args')   # positional args mapped to param names
          - tools['name']({'p': 1}, extra=2)      # mixed
        All are normalized to a single dict before calling the MCP server.
        """
        mcp_name = tool_def["name"]
        schema = tool_def.get("inputSchema", {})
        properties = schema.get("properties", {})
        param_names = list(properties.keys())
        description = tool_def.get("description", "")
        normalize = self._normalize_arg

        async def tool_fn(*args, **kwargs) -> Any:
            arguments = {}

            # Case 1: single dict argument — tools['name']({'q': 'test'})
            if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                arguments = args[0]
            # Case 2: positional args — tools['name']('test') mapped to param names
            elif args:
                for i, arg in enumerate(args):
                    if isinstance(arg, dict) and i == 0:
                        # First arg is a dict, treat as params dict
                        arguments.update(arg)
                    elif i < len(param_names):
                        arguments[param_names[i]] = arg
                arguments.update(kwargs)
            # Case 3: keyword args only — tools['name'](q='test')
            else:
                arguments = kwargs

            # Normalize values based on schema (fix common LLM formatting mistakes)
            for key in list(arguments.keys()):
                prop_schema = properties.get(key, {})
                arguments[key] = normalize(key, arguments[key], prop_schema)

            return await conn.call_tool(mcp_name, arguments)

        # Set metadata for schema extraction
        tool_fn.__name__ = mcp_name
        tool_fn.__qualname__ = mcp_name
        tool_fn.__doc__ = description

        # Attach schema for the code generator
        tool_fn._mcp_schema = tool_def

        return tool_fn

    def get_schemas(self) -> dict[str, dict]:
        """Return the MCP tool schemas for all loaded tools."""
        schemas = {}
        for name, fn in self._tools.items():
            if hasattr(fn, '_mcp_schema'):
                schemas[name] = fn._mcp_schema
            else:
                schemas[name] = {"name": name, "description": ""}
        return schemas

    async def close(self):
        """Close all MCP server connections."""
        for conn in self._connections:
            await conn.close()
        self._connections.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
