#!/usr/bin/env python3
from __future__ import annotations

import json
import sys


def main() -> int:
    request = json.loads(sys.stdin.read())
    user = str(request.get("user", ""))

    if any(token in user for token in ("反例がない", "同じところ", "分からない", "stuck")):
        result = {
            "state": "STUCK",
            "confidence": 0.90,
            "evidence": ["current turn explicitly reports a blocked or repeated reasoning trajectory"],
            "proposed_action": "CUE",
            "cue": "反例は？",
        }
    elif user.rstrip().endswith(("けど", "ので", "because", "but")):
        result = {
            "state": "PAUSED",
            "confidence": 0.75,
            "evidence": ["current turn ends as an unfinished proposition"],
            "proposed_action": "CONTINUE",
            "cue": None,
        }
    else:
        result = {
            "state": "FLOWING",
            "confidence": 0.82,
            "evidence": ["current turn contains an advancing reasoning statement"],
            "proposed_action": "SILENCE",
            "cue": None,
        }

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
