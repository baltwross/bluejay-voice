"""
Reading state persistence tests.

These are lightweight unit tests that ensure the agent does not crash if the
configured reading state path is unexpectedly a directory.
"""

from pathlib import Path

import pytest


def _get_reading_state_cls():
    """
    Import ReadingState from `backend/agent.py`.

    We import lazily (inside tests) so the file can still be collected even in
    environments missing optional LiveKit plugins, and will be reported as
    skipped instead of "no tests collected".
    """
    try:
        from agent import ReadingState  # type: ignore
        return ReadingState
    except ImportError as e:
        pytest.skip(f"Agent dependencies unavailable in this environment: {e}")


def test_reading_state_load_and_save_support_directory_path(tmp_path: Path) -> None:
    """
    If `.reading_state.json` is a directory (accidentally created), we should:
    - not crash on load()
    - save state inside it as `state.json`
    - load that state successfully
    """
    state_dir = tmp_path / ".reading_state.json"
    state_dir.mkdir()

    # Empty directory should behave like "no saved state" (and never raise)
    ReadingState = _get_reading_state_cls()
    assert ReadingState.load(file_path=state_dir) is None

    state = ReadingState(
        is_reading=False,
        is_paused=True,
        document_id="doc-123",
        document_title="Test Document",
        current_chunk=2,
        total_chunks=10,
    )
    assert state.save(file_path=state_dir) is True

    # Should write the state file inside the directory
    state_file = state_dir / "state.json"
    assert state_file.exists()
    assert state_file.is_file()

    loaded = ReadingState.load(file_path=state_dir)
    assert loaded is not None
    assert loaded.document_id == "doc-123"
    assert loaded.document_title == "Test Document"
    assert loaded.current_chunk == 2
    assert loaded.total_chunks == 10
    # Loading always yields a paused state so the agent can offer resume.
    assert loaded.is_reading is False
    assert loaded.is_paused is True


def test_reading_state_load_cleans_up_corrupt_json(tmp_path: Path) -> None:
    """Corrupt JSON should not crash; it should be cleaned up."""
    ReadingState = _get_reading_state_cls()
    state_file = tmp_path / ".reading_state.json"
    state_file.write_text("{not valid json", encoding="utf-8")

    assert ReadingState.load(file_path=state_file) is None
    assert not state_file.exists()


