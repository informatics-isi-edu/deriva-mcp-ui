"""Phase 0 smoke test -- package imports cleanly."""

import deriva_mcp_ui


def test_version() -> None:
    assert deriva_mcp_ui.__version__ == "0.1.0"
