"""JSONL transcript replay. External model command is optional.

A transcript event updates the model; a boundary event attempts delivery.
With --async-model, updates never block on model inference and boundaries never
wait for an unfinished candidate. This is an event replay, not microphone I/O.
"""
import argparse
import json
import subprocess
import sys
from .core import Session
from .async_session import AsyncSession


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-command", nargs="+", help="external decision model executable")
    parser.add_argument("--async-model", action="store_true", help="non-blocking inference")
    args = parser.parse_args()

    def complete(system: str, user: str) -> str:
        if not args.model_command:
            return '{"action":"NONE"}'
        result = subprocess.run(
            args.model_command,
            input=json.dumps({"system": system, "user": user}, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=5,
            check=True,
        )
        return result.stdout

    session = AsyncSession(complete) if args.async_model else Session(complete)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            event = json.loads(line)
            at = event["at_ms"]
            if event["type"] == "transcript":
                session.update(event["text"], at)
            elif event["type"] == "boundary":
                decision = session.emit(at, speaking=event.get("speaking", False))
                print(json.dumps({
                    "at_ms": at,
                    "action": decision.action,
                    "text": decision.text,
                    "reason": decision.reason,
                }, ensure_ascii=False), flush=True)
            else:
                raise ValueError("unknown event type")
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


if __name__ == "__main__":
    main()
