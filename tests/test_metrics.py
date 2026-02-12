"""Tests for src.metrics -- MetricsCollector."""

import pytest

from python_codemode.metrics import MetricsCollector


class TestMetricsCollector:
    def test_initial_state(self):
        m = MetricsCollector()
        assert m.execution_count == 0
        assert m.tool_call_count == 0
        assert m.token_count == 0
        assert m.retry_count == 0

    def test_record_execution(self):
        m = MetricsCollector()
        m.record_execution("docker", 1.5, True)
        m.record_execution("docker", 0.3, False)
        assert m.execution_count == 2

    def test_record_tool_call(self):
        m = MetricsCollector()
        m.record_tool_call("search_web", 0.2, True)
        m.record_tool_call("create_event", 0.1, True)
        m.record_tool_call("search_web", 0.3, False)
        assert m.tool_call_count == 3

    def test_record_tokens(self):
        m = MetricsCollector()
        m.record_llm_call("gpt-5.2-codex", 80, 20, 100, 1.0, 1)
        m.record_llm_call("gpt-5.2-codex", 30, 20, 50, 0.5, 2)
        assert m.token_count == 150

    def test_record_retry(self):
        m = MetricsCollector()
        m.record_retry()
        m.record_retry()
        assert m.retry_count == 2

    def test_reset(self):
        m = MetricsCollector()
        m.record_execution("docker", 1.0, True)
        m.record_tool_call("search", 0.5, True)
        m.record_tokens(200)
        m.record_retry()
        m.reset()
        assert m.execution_count == 0
        assert m.tool_call_count == 0
        assert m.token_count == 0
        assert m.retry_count == 0

    def test_summary_empty(self):
        m = MetricsCollector()
        s = m.summary()
        assert s["total_executions"] == 0
        assert s["successful_executions"] == 0
        assert s["failed_executions"] == 0
        assert s["total_execution_time"] == 0.0
        assert s["avg_execution_time"] == 0.0
        assert s["total_tool_calls"] == 0
        assert s["token_count"] == 0
        assert s["retry_count"] == 0
        assert s["tool_breakdown"] == {}

    def test_summary_with_data(self):
        m = MetricsCollector()
        m.record_execution("pyodide", 1.0, True)
        m.record_execution("pyodide", 2.0, False)
        m.record_tool_call("search_web", 0.5, True)
        m.record_tool_call("search_web", 0.3, True)
        m.record_tool_call("create_event", 0.2, False)
        m.record_llm_call("gpt-5.2-codex", 400, 100, 500, 2.0, 1)
        m.record_retry()

        s = m.summary()
        assert s["total_executions"] == 2
        assert s["successful_executions"] == 1
        assert s["failed_executions"] == 1
        assert s["total_execution_time"] == 3.0
        assert s["avg_execution_time"] == 1.5
        assert s["total_tool_calls"] == 3
        assert s["successful_tool_calls"] == 2
        assert s["failed_tool_calls"] == 1
        assert s["token_count"] == 500
        assert s["total_input_tokens"] == 400
        assert s["total_output_tokens"] == 100
        assert s["retry_count"] == 1
        assert len(s["llm_calls"]) == 1
        assert s["llm_calls"][0]["model"] == "gpt-5.2-codex"

        # Per-tool breakdown
        bd = s["tool_breakdown"]
        assert "search_web" in bd
        assert bd["search_web"]["calls"] == 2
        assert bd["search_web"]["successes"] == 2
        assert bd["search_web"]["failures"] == 0
        assert "create_event" in bd
        assert bd["create_event"]["calls"] == 1
        assert bd["create_event"]["failures"] == 1

    def test_format_table(self):
        m = MetricsCollector()
        m.record_execution("docker", 0.8, True)
        m.record_tool_call("search_web", 0.2, True)
        m.record_llm_call("gpt-5.2-codex", 80, 20, 100, 1.0, 1)
        table = m.format_table()
        assert "CODEMODE METRICS" in table
        assert "Executions" in table
        assert "Tool Calls" in table
        assert "Cumulative" in table
        assert "search_web" in table
        assert "gpt-5.2-codex" in table

    def test_format_table_empty(self):
        m = MetricsCollector()
        table = m.format_table()
        assert "CODEMODE METRICS" in table
        assert "0" in table

    def test_summary_avg_tool_time(self):
        m = MetricsCollector()
        m.record_tool_call("a", 1.0, True)
        m.record_tool_call("b", 3.0, True)
        s = m.summary()
        assert s["avg_tool_time"] == 2.0
