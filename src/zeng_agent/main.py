from __future__ import annotations

import argparse

import uvicorn

from .api import create_app
from .bootstrap import build_service
from .config import AgentConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="zeng-agent service")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    return parser


def load_config(path: str | None) -> AgentConfig:
    if not path:
        return AgentConfig()
    return AgentConfig.from_yaml(path)


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    service = build_service(config=config, waypoint_names=config.waypoint_names)
    uvicorn.run(create_app(service), host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
