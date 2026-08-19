from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

from .voice import VoiceEvent, VoiceEventType, VoiceSession


_TURN_TEXTS = (
    "前提と根拠と依存関係を分けて考える。",
    "まだ比較条件を決めていないので",
    "観測可能な指標を定義してから反例を確認する。",
    "えっと、その、なんだろう、性能が良いと思う",
    "原因と結果の依存関係を整理する。",
    "情報が多すぎて混乱した",
    "次の反例と検証条件を考える。",
    "何からどう進めればよいかわからない",
)


def run_soak(duration_minutes: int = 10, turn_interval_ms: int = 5_000) -> dict[str, object]:
    if duration_minutes < 1 or turn_interval_ms < 1_000:
        raise ValueError("duration and turn interval are below safe test bounds")
    duration_ms = duration_minutes * 60_000
    session = VoiceSession()
    results = []
    started = perf_counter()
    turn_count = duration_ms // turn_interval_ms
    for index in range(turn_count):
        base = index * turn_interval_ms
        text = _TURN_TEXTS[index % len(_TURN_TEXTS)]
        events = (
            VoiceEvent(VoiceEventType.SPEECH_STARTED, base),
            VoiceEvent(VoiceEventType.TRANSCRIPT_PARTIAL, base + 1_000, text[: max(1, len(text) // 2)]),
            VoiceEvent(VoiceEventType.TRANSCRIPT_FINAL, base + 2_500, text),
            VoiceEvent(VoiceEventType.SPEECH_ENDED, base + 3_000),
            VoiceEvent(VoiceEventType.TICK, base + 3_700),
        )
        for event in events:
            result = session.feed(event)
            if result is not None:
                results.append(result)
    elapsed = perf_counter() - started
    counts = Counter(result.decision.action.value for result in results)
    invariants = {
        "one_decision_per_turn": len(results) == turn_count,
        "all_decisions_after_speech": all(
            result.turn.boundary_at_ms >= result.turn.ended_at_ms + 700 for result in results
        ),
        "simulation_covers_requested_duration": turn_count * turn_interval_ms >= duration_ms,
        "intervention_rate_below_half": (
            sum(count for action, count in counts.items() if action != "SILENCE") / max(1, len(results)) < 0.5
        ),
    }
    return {
        "experiment": "voice event ten-minute soak",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "simulated_duration_ms": duration_ms,
        "turn_interval_ms": turn_interval_ms,
        "turns": turn_count,
        "action_counts": dict(sorted(counts.items())),
        "processing_seconds": elapsed,
        "invariants": invariants,
        "passed": all(invariants.values()),
        "sample_results": [
            {
                "turn": asdict(result.turn),
                "action": result.decision.action.value,
                "reason": result.decision.reason,
            }
            for result in results[:4]
        ],
        "epistemic_limit": "This is a timing/state-machine load test, not a live ASR or cognitive-effect study.",
    }


def write_soak_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
