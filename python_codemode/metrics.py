"""Metrics collection for codemode execution runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCallRecord:
    """A single recorded tool invocation."""
    name: str
    duration: float
    success: bool


@dataclass
class ExecutionRecord:
    """A single recorded sandbox execution."""
    backend: str
    duration: float
    success: bool


@dataclass
class LLMCallRecord:
    """A single recorded LLM call with token usage."""
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    duration: float
    attempt: int


class MetricsCollector:
    """Collect and summarise execution metrics.

    Tracks LLM token usage per call, execution time per backend,
    retry counts, and individual tool call statistics.
    """

    def __init__(self) -> None:
        self._executions: list[ExecutionRecord] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._llm_calls: list[LLMCallRecord] = []
        self._retry_count: int = 0

    # ---- Recording methods ----

    def record_execution(self, backend: str, duration: float, success: bool) -> None:
        """Record one sandbox execution attempt."""
        self._executions.append(
            ExecutionRecord(backend=backend, duration=duration, success=success)
        )

    def record_tool_call(self, name: str, duration: float, success: bool) -> None:
        """Record one tool invocation."""
        self._tool_calls.append(
            ToolCallRecord(name=name, duration=duration, success=success)
        )

    def record_llm_call(
        self, model: str, input_tokens: int, output_tokens: int,
        total_tokens: int, duration: float, attempt: int,
    ) -> None:
        """Record one LLM call with token usage."""
        self._llm_calls.append(LLMCallRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            duration=duration,
            attempt=attempt,
        ))

    def record_tokens(self, count: int) -> None:
        """Accumulate token usage (legacy — prefer record_llm_call)."""
        pass  # Kept for backward compat, llm_calls tracks this now

    def record_retry(self) -> None:
        """Increment the retry counter."""
        self._retry_count += 1

    # ---- Query methods ----

    @property
    def execution_count(self) -> int:
        return len(self._executions)

    @property
    def tool_call_count(self) -> int:
        return len(self._tool_calls)

    @property
    def token_count(self) -> int:
        return sum(c.total_tokens for c in self._llm_calls)

    @property
    def retry_count(self) -> int:
        return self._retry_count

    # ---- Summaries ----

    def summary(self) -> dict:
        """Return a summary dict of all collected metrics."""
        # Execution stats
        total_exec = len(self._executions)
        success_exec = sum(1 for e in self._executions if e.success)
        fail_exec = total_exec - success_exec
        total_exec_time = sum(e.duration for e in self._executions)
        avg_exec_time = total_exec_time / total_exec if total_exec else 0.0

        # Tool call stats
        total_tc = len(self._tool_calls)
        success_tc = sum(1 for t in self._tool_calls if t.success)
        fail_tc = total_tc - success_tc
        total_tc_time = sum(t.duration for t in self._tool_calls)
        avg_tc_time = total_tc_time / total_tc if total_tc else 0.0

        # LLM token stats
        total_input = sum(c.input_tokens for c in self._llm_calls)
        total_output = sum(c.output_tokens for c in self._llm_calls)
        total_tokens = sum(c.total_tokens for c in self._llm_calls)
        total_llm_time = sum(c.duration for c in self._llm_calls)

        # Per-tool breakdown
        tool_breakdown: dict[str, dict] = {}
        for tc in self._tool_calls:
            entry = tool_breakdown.setdefault(
                tc.name, {"calls": 0, "successes": 0, "failures": 0, "total_time": 0.0}
            )
            entry["calls"] += 1
            if tc.success:
                entry["successes"] += 1
            else:
                entry["failures"] += 1
            entry["total_time"] += tc.duration

        # Per-LLM-call breakdown
        llm_calls = [
            {
                "attempt": c.attempt,
                "model": c.model,
                "input_tokens": c.input_tokens,
                "output_tokens": c.output_tokens,
                "total_tokens": c.total_tokens,
                "duration": round(c.duration, 4),
            }
            for c in self._llm_calls
        ]

        return {
            "total_executions": total_exec,
            "successful_executions": success_exec,
            "failed_executions": fail_exec,
            "total_execution_time": round(total_exec_time, 4),
            "avg_execution_time": round(avg_exec_time, 4),
            "total_tool_calls": total_tc,
            "successful_tool_calls": success_tc,
            "failed_tool_calls": fail_tc,
            "total_tool_time": round(total_tc_time, 4),
            "avg_tool_time": round(avg_tc_time, 4),
            "llm_calls": llm_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_llm_time": round(total_llm_time, 4),
            "token_count": total_tokens,
            "retry_count": self._retry_count,
            "retries": self._retry_count,
            "tool_breakdown": tool_breakdown,
        }

    def format_table(self) -> str:
        """Return a human-readable table of collected metrics."""
        s = self.summary()
        lines = [
            "",
            "╔══════════════════════════════════════════════════════════╗",
            "║               CODEMODE METRICS                         ║",
            "╠══════════════════════════════════════════════════════════╣",
        ]

        # LLM Calls
        lines.append("║ LLM Calls")
        for call in s.get("llm_calls", []):
            lines.append(
                f"║   Attempt {call['attempt']} | {call['model']}"
                f" | in={call['input_tokens']} out={call['output_tokens']}"
                f" total={call['total_tokens']} | {call['duration']:.2f}s"
            )
        lines.append(
            f"║   Cumulative: in={s['total_input_tokens']}"
            f" out={s['total_output_tokens']}"
            f" total={s['total_tokens']}"
            f" | {s['total_llm_time']:.2f}s"
        )

        # Executions
        lines.append("╠──────────────────────────────────────────────────────────")
        lines.append("║ Sandbox Executions")
        lines.append(
            f"║   Total: {s['total_executions']}"
            f" | OK: {s['successful_executions']}"
            f" | Failed: {s['failed_executions']}"
            f" | Retries: {s['retry_count']}"
        )
        if s['total_executions']:
            lines.append(f"║   Avg time: {s['avg_execution_time']:.4f}s")

        # Tool Calls
        lines.append("╠──────────────────────────────────────────────────────────")
        lines.append("║ Tool Calls")
        lines.append(
            f"║   Total: {s['total_tool_calls']}"
            f" | OK: {s['successful_tool_calls']}"
            f" | Failed: {s['failed_tool_calls']}"
        )
        if s['total_tool_calls']:
            lines.append(f"║   Avg time: {s['avg_tool_time']:.4f}s")

        breakdown = s.get("tool_breakdown", {})
        if breakdown:
            lines.append("║")
            lines.append(f"║   {'Name':<25} {'Calls':>5} {'OK':>4} {'Fail':>4} {'Time':>8}")
            lines.append("║   " + "─" * 48)
            for name, info in breakdown.items():
                lines.append(
                    f"║   {name:<25} {info['calls']:>5} {info['successes']:>4}"
                    f" {info['failures']:>4} {info['total_time']:>7.3f}s"
                )

        lines.append("╚══════════════════════════════════════════════════════════╝")
        return "\n".join(lines)

    # ---- Reset ----

    def reset(self) -> None:
        """Clear all collected metrics."""
        self._executions.clear()
        self._tool_calls.clear()
        self._llm_calls.clear()
        self._retry_count = 0
