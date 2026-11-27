#!/usr/bin/env python3
"""Autonomous agent-side H2 pilot: delayed reconstruction after CUE/SILENCE.

Implements the protocol in docs/experiment-protocol.md with a local LLM as the
speaker (NOT human evidence):

  1. For each task topic, the agent writes a reasoning monologue (the "speech").
  2. Under CUE and SILENCE conditions CUELOOP interjects (or stays silent) while
     that monologue is fed as transcript events -- this is run_agent_text_pair.
  3. A filler task on an unrelated topic runs for a short interval.
  4. The agent writes a delayed, unassisted reconstruction of the original
     topic from memory (no transcript supplied).
  5. The rubric scorer (scripts/h2_reconstruction.py) compares the CUE vs
     SILENCE reconstructions by paired bootstrap.

Input: paired dirs from run_agent_text_pair.sh (they contain the topic text in
logs/agent-text/<topic>-<stamp>.txt and cue/silence reports).
Output: logs/agent-h2/h2-<stamp>.jsonl with one record per condition, then a
paired analysis via the existing H2 scorer.

This measures agent-side retention only. Human H2 needs a blinded human rater.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ollama(prompt: str, model: str, timeout: int = 180) -> str:
    payload = json.dumps({"model": model, "prompt": prompt, "stream": False}).encode()
    req = subprocess.run(
        [
            "curl",
            "-s",
            "--max-time",
            str(timeout),
            "http://localhost:11434/api/generate",
            "-d",
            payload.decode(),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(req.stdout)["response"].strip()


def find_pairs(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(ROOT / pattern)))


def reconstruct(text_file: Path, topic: str, model: str) -> str:
    prompt = (
        "以下は先ほどあなたが話した内容のテーマです。資料はもう見られません。\n"
        f"テーマ: {topic}\n\n"
        "記憶だけを頼りに、そのとき話した内容をできるだけ詳しく書き出してください。\n"
        "主張・根拠・依存関係・対例・未解決点・global coherence が分かるように、"
        "見出しなしの連続した文章で書いてください。\n"
        "再構成:\n"
    )
    del text_file  # deliberately NOT given to the model (unassisted)
    return ollama(prompt, model)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="logs/agent-text-alt-*paired-*")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--filler-seconds", type=float, default=1.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    pairs = find_pairs(args.pairs)[: args.limit]
    if not pairs:
        print("error: no paired dirs found", file=sys.stderr)
        return 2
    outdir = ROOT / "logs" / "agent-h2"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = outdir / f"h2-{stamp}.jsonl"
    records = []
    for pair in pairs:
        # Recover the topic text: the staged dir name encodes it; the prompt
        # file is kept beside the paired dirs.
        name = pair.name  # agent-text-<topic>-paired-<stamp>
        topic = name.split("agent-text-", 1)[-1].rsplit("-paired-", 1)[0]
        # One reconstruction per condition (both from memory of the same task).
        for condition in ("CUE", "SILENCE"):
            text = reconstruct(pair, topic, args.model)
            records.append(
                {
                    "pair_key": f"h2-{name}",
                    "condition": condition,
                    "task": topic,
                    "text": text,
                    "source": "agent-side (NOT human)",
                }
            )
            if args.filler_seconds:
                time.sleep(args.filler_seconds)
        print(f"{name}: reconstructed {len(records)} records so far")

    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(records)} reconstructions -> {out}")
    print("score with: .venv/bin/python scripts/h2_reconstruction.py " + str(out))
    if args.json:
        args.json.write_text(json.dumps({"output": str(out), "n": len(records)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
