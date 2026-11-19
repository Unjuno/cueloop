#!/usr/bin/env python
"""Audit live paired sessions for data integrity.

For every design-choice pair, read the .report files and classify:

  EMPTY      — both arms produced zero Japanese clause text (mic issue)
  SILENT     — exactly one arm produced no clause text; the other arm did
  LIGHT      — nonzero but <3 utterances with text on either arm
  OK         — both arms had Japanese clause text
  NON-LIVE   — at least one arm's report file is missing or unreadable

Prints one line per pair plus a summary. Use --json for machine-readable
output. Does NOT read raw audio (it never exists).
"""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _count_clauses(path):
    if not Path(path).exists():
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.lstrip().startswith("{"):
                try:
                    d = json.loads(line)
                    if d.get("clause"):
                        count += 1
                except ValueError:
                    continue
    return count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("roots", nargs="*", default=["logs/self-design-choice-paired-*"])
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true", help="skip per-pair lines")
    args = p.parse_args()

    pairs = []
    for arg in args.roots:
        pairs.extend(glob.glob(arg))
    pairs = sorted(set(pairs))

    table = []
    summary = Counter()
    for pair in pairs:
        cue_report = _load(Path(pair) / "cue.json")
        sil_report = _load(Path(pair) / "silence.json")
        cue_clauses = _count_clauses(Path(pair) / "cue.jsonl")
        sil_clauses = _count_clauses(Path(pair) / "silence.jsonl")
        if cue_report is None or sil_report is None:
            verdict = "NON-LIVE" if cue_report is None else "NON-LIVE-SILENT-ONLY"
        elif cue_clauses == 0 and sil_clauses == 0:
            verdict = "EMPTY"
        elif cue_clauses == 0 or sil_clauses == 0:
            verdict = "SILENT"
        elif cue_report.get("utterances", 0) + sil_report.get("utterances", 0) < 3:
            verdict = "LIGHT"
        else:
            verdict = "OK"
        summary[verdict] += 1
        table.append(
            {
                "pair": Path(pair).name,
                "verdict": verdict,
                "cue_utterances": cue_report.get("utterances") if cue_report else None,
                "cue_clauses": cue_clauses,
                "cue_delivered": cue_report.get("delivered") if cue_report else None,
                "silence_utterances": sil_report.get("utterances")
                if sil_report
                else None,
                "silence_clauses": sil_clauses,
                "silence_delivered": sil_report.get("delivered")
                if sil_report
                else None,
            }
        )

    if args.json:
        print(
            json.dumps(
                {"pairs": table, "summary": dict(summary)}, indent=2, ensure_ascii=False
            )
        )
        return

    for row in table:
        if args.quiet:
            break
        print(
            f"{row['verdict']:9s} {row['pair']:<48s} "
            f"cue_utt={row['cue_utterances']!s:>4} cue_clauses={row['cue_clauses']:>3} "
            f"cue_del={row['cue_delivered']!s:>3} sil_utt={row['silence_utterances']!s:>4} "
            f"sil_clauses={row['silence_clauses']:>3} sil_del={row['silence_delivered']!s:>3}"
        )
    print()
    print(f"scanned={len(pairs)}")
    for k, v in sorted(summary.items()):
        print(f"  {k:<8s} {v}")


if __name__ == "__main__":
    main()
