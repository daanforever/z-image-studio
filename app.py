"""Entry point for Z-Image-Turbo Gradio studio."""

from __future__ import annotations

import argparse

from zimage.config import DEFAULT_PORT, load_dotenv
from zimage.ui.layout import build_ui
from zimage.ui.log import ensure_console_logging, log, log_error
from zimage.ui.theme import appearance_kwargs

load_dotenv()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z-Image-Turbo Gradio studio")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    ensure_console_logging()
    args = parse_args(argv)
    demo = build_ui()
    log.info("Starting server at http://%s:%s", args.host, args.port)
    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        **appearance_kwargs(),
    )


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        log_error(f"Failed to start server: {exc}")
        raise
