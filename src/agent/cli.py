"""Command-line entry point.

stdout carries only the JSON result; every diagnostic goes to stderr, so
piping the result into a parser is safe even on a failing run.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import research


def main() -> None:
    parser = argparse.ArgumentParser(description="Research a question and cite the sources.")
    parser.add_argument("--topic", type=str, required=False)
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("question", nargs="?", help="Research question (positional for Docker)")
    args = parser.parse_args()

    topic = args.topic or args.question
    if not topic:
        parser.error("Please provide a research topic/question.")

    result = research(topic, args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # Non-zero on a fallback answer, so a shell caller can branch on it.
    sys.exit(1 if result["degraded"] else 0)


if __name__ == "__main__":
    main()
