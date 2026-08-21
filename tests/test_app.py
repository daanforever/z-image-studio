from __future__ import annotations

from pathlib import Path

import pytest

from app import main, parse_args
from zimage.ui.theme import CUSTOM_CSS


def test_parse_args_defaults():
    args = parse_args([])
    assert args.host == "0.0.0.0"
    assert args.share is False


def test_parse_args_overrides():
    args = parse_args(["--host", "127.0.0.1", "--port", "8000", "--share"])
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.share is True


def test_main_launches_with_appearance(monkeypatch):
    captured = {}

    def fake_launch(*_args, **kwargs):
        captured.update(kwargs)
        return None, "", ""

    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    monkeypatch.setattr("gradio.blocks.Blocks.launch", fake_launch)
    monkeypatch.setattr("app.ensure_console_logging", lambda: None)
    monkeypatch.setattr("app.log.info", lambda *_args, **_kwargs: None)

    main(["--host", "127.0.0.1", "--port", "8000"])

    assert captured["server_name"] == "127.0.0.1"
    assert captured["server_port"] == 8000
    assert captured["theme"] is not None
    assert captured["css"] == CUSTOM_CSS
    assert "color-scheme" in captured["head"]


def test_module_entrypoint_logs_oserror(monkeypatch, capsys):
    import runpy

    monkeypatch.setattr("zimage.ui.layout.format_status", lambda: "ready")
    monkeypatch.setattr(
        "gradio.blocks.Blocks.launch",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("address already in use")),
    )
    monkeypatch.setattr("sys.argv", ["app.py", "--host", "127.0.0.1", "--port", "9"])

    app_path = Path(__file__).resolve().parents[1] / "app.py"
    with pytest.raises(OSError, match="address already in use"):
        runpy.run_path(str(app_path), run_name="__main__")
    assert "Failed to start server" in capsys.readouterr().out
