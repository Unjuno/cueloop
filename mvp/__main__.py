"""JSONL replay: stdin events; stdout decisions. Optional external model command.

Events: {"type":"transcript","text":"...","at_ms":0} or
{"type":"boundary","at_ms":1000,"speaking":false}.
Model command reads {"system":...,"user":...} on stdin and returns JSON on stdout.
"""
import argparse
import json
import subprocess
import sys
from .core import Session

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-command", nargs="+", help="external decision model executable")
    args = parser.parse_args()

    def complete(system: str, user: str) -> str:
        if not args.model_command:
            return '{"action":"NONE"}'
        result = subprocess.run(args.model_command, input=json.dumps({"system":system,"user":user}),
                                text=True, capture_output=True, timeout=5, check=True)
        return result.stdout

    session = Session(complete)
    for line in sys.stdin:
        if not line.strip():
            continue
        event = json.loads(line)
        at = event["at_ms"]
        if event["type"] == "transcript":
            session.update(event["text"], at)
        elif event["type"] == "boundary":
            decision = session.emit(at, speaking=event.get("speaking", False))
            print(json.dumps({"at_ms":at,"action":decision.action,"text":decision.text,
                              "reason":decision.reason}, ensure_ascii=False), flush=True)
        else:
            raise ValueError("unknown event type")

if __name__ == "__main__":
    main()
