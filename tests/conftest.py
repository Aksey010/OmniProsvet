import sys
from pathlib import Path

# Add src to path for all tests
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory path."""
    return tmp_path


@pytest.fixture
def mock_nim_client():
    """Mock NIM client for testing."""
    from unittest.mock import MagicMock
    from src.nim_client import NimClient

    mock = MagicMock(spec=NimClient)
    mock.send_prompt.return_value = "Mocked response"
    return mock