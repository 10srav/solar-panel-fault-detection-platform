#!/usr/bin/env python
"""Start the FastAPI server."""

from __future__ import annotations

import argparse
import uvicorn

from src.config import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the API server")

    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    return parser.parse_args()


def main() -> None:
    """Start the server."""
    args = parse_args()
    config = load_config(args.config)

    host = args.host or config.api.host
    port = args.port or config.api.port
    workers = args.workers or config.api.workers

    print(f"Starting server on {host}:{port}")
    print(f"Workers: {workers}")

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        workers=1 if args.reload else workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
