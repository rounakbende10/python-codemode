"""Shared test fixtures for python-codemode."""
import pytest


@pytest.fixture
def mock_tools():
    """Provide mock tools for testing."""
    async def search_web(query: str) -> list:
        return [{"title": f"Result for {query}", "url": "https://example.com", "snippet": "A result"}]

    async def create_event(title: str, date: str) -> dict:
        return {"id": "evt_123", "title": title, "date": date, "status": "created"}

    async def create_issue(repo: str, title: str, body: str = "") -> dict:
        return {"number": 1, "title": title, "body": body, "url": f"https://github.com/{repo}/issues/1"}

    return {
        "search_web": search_web,
        "create_event": create_event,
        "create_issue": create_issue,
    }


class MockCodeMode:
    """Mock CodeMode instance for testing integrations."""

    def __init__(self, tools=None):
        self.tools = tools or {}

    async def run(self, task: str, verbose: bool = False):
        """Mock run that returns a structured result."""
        return {
            "success": True,
            "output": {"task": task, "result": f"Executed: {task}"},
        }


@pytest.fixture
def mock_codemode():
    """Provide a mock CodeMode instance for integration testing."""
    tools = {
        "search_web": lambda q: None,
        "create_event": lambda t, d: None,
        "create_issue": lambda r, t: None,
    }
    return MockCodeMode(tools=tools)


@pytest.fixture
def mock_codemode_no_tools():
    """Provide a mock CodeMode instance with no tools."""
    return MockCodeMode()
