from __future__ import annotations

from pathlib import Path

import pytest

from app import main, parse_args
from zimage.ui.theme import CUSTOM_CSS, CUSTOM_JS


def test_parse_args_defaults():
    args = parse_args([])
    assert args.host == "127.0.0.1"
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
    assert captured["share"] is False
    assert captured["theme"] is not None
    assert captured["css"] == CUSTOM_CSS
    assert captured["js"] == CUSTOM_JS
    assert "color-scheme" in captured["head"]


def test_main_forwards_share_to_build_ui(monkeypatch):
    captured = {}

    class FakeDemo:
        def queue(self, max_size=4):
            return self

        def launch(self, **kwargs):
            captured["launch"] = kwargs
            return None, "", ""

    def fake_build_ui(*, share=False):
        captured["share"] = share
        return FakeDemo()

    monkeypatch.setattr("app.build_ui", fake_build_ui)
    monkeypatch.setattr("app.ensure_console_logging", lambda: None)
    monkeypatch.setattr("app.log.info", lambda *_args, **_kwargs: None)

    main(["--host", "127.0.0.1", "--port", "8000"])
    assert captured["share"] is False
    assert captured["launch"]["share"] is False

    main(["--host", "127.0.0.1", "--port", "8000", "--share"])
    assert captured["share"] is True
    assert captured["launch"]["share"] is True


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
    captured = capsys.readouterr()
    assert "Failed to start server" in captured.out + captured.err
