from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Sequence

from .core import Action, Analysis, Analyzer, ControllerConfig, Decision, HeuristicAnalyzer, InterventionController
from .commands import parse_command_json
from .knowledge import GroundedRetrievalAnalyzer, KnowledgeIndex
from .llm import CommandCompletion, HybridAnalyzer, LLMAnalyzer
from .personalization import AdaptiveController, PersonalizationProfile
from .voice import (
    CommandSpeaker,
    MidSpeechInjector,
    NullSpeaker,
    ProfileSpeaker,
    Speaker,
    TurnBoundaryDetector,
    VoiceProfile,
    VoiceSession,
    command_event_lines,
    run_live_event_stream,
    run_event_stream,
)


class ObservationOnlyController:
    """Records expiring state observations without claiming cue delivery."""

    def __init__(self, profile: PersonalizationProfile) -> None:
        self.profile = profile

    def decide(self, analysis: Analysis) -> Decision:
        self.profile.observe_state(analysis)
        return Decision(Action.SILENCE, None, "live monitoring has no delivery channel", analysis)


def _speaker_from_profiles_json(
    profiles_json: str | None,
    tts_command: Sequence[str] | None,
    timeout_seconds: float,
) -> Speaker:
    if profiles_json:
        raw = json.loads(profiles_json)
        if not isinstance(raw, list) or not raw:
            raise ValueError("--voice-profiles-json must be a non-empty JSON array")
        profiles = [_profile_from_dict(item) for item in raw]
        intents = {profile.intent for profile in profiles}
        if "neutral" not in intents:
            raise ValueError("voice profiles must include a neutral profile")
        return ProfileSpeaker(profiles, timeout_seconds)
    command = list(tts_command or [])
    if not command:
        raise ValueError("voice profiles require a TTS command or explicit profiles")
    return ProfileSpeaker(
        [VoiceProfile(name="neutral", command=tuple(command), intent="neutral")],
        timeout_seconds,
    )


def _profile_from_dict(item: object) -> VoiceProfile:
    if not isinstance(item, dict):
        raise ValueError("each voice profile must be a JSON object")
    invalid = set(item) - {"name", "command", "voice", "rate", "volume", "intent"}
    if invalid:
        raise ValueError(f"unknown voice profile fields: {sorted(invalid)}")
    if not isinstance(item.get("name"), str) or not item["name"]:
        raise ValueError("voice profile requires a name")
    if not isinstance(item.get("intent"), str) or not item["intent"]:
        raise ValueError("voice profile requires an intent")
    command = item.get("command")
    if command is None:
        raise ValueError("voice profile requires a command")
    if isinstance(command, str):
        command = [command]
    if not isinstance(command, list) or not command or not all(isinstance(c, str) for c in command):
        raise ValueError("voice profile command must be a non-empty string or list")
    rate = item.get("rate")
    if rate is not None and (not isinstance(rate, int) or rate < 0):
        raise ValueError("voice profile rate must be a non-negative integer")
    volume = item.get("volume")
    if volume is not None and (not isinstance(volume, (int, float)) or not 0.0 <= float(volume) <= 1.0):
        raise ValueError("voice profile volume must be between 0 and 1")
    voice = item.get("voice")
    if voice is not None and not isinstance(voice, str):
        raise ValueError("voice profile voice must be a string")
    return VoiceProfile(
        name=item["name"],
        command=tuple(command),
        voice=voice,
        rate=rate,
        volume=float(volume) if volume is not None else None,
        intent=item["intent"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CUELOOP from streaming VAD/ASR JSONL events.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", type=Path, help="Replay a JSONL voice-event file; use - for stdin.")
    source.add_argument("--event-command", nargs="+", help="Command that streams JSONL voice events.")
    source.add_argument("--event-command-json", help="JSON string array for commands containing option-like args.")
    parser.add_argument("--tts-command", nargs="+", help="TTS command; the utterance is appended as one argument.")
    parser.add_argument("--tts-command-json", help="JSON string array for TTS command arguments.")
    parser.add_argument(
        "--tts-voice",
        help="Voice name passed to the TTS command (e.g. Kyoko). When the command "
        "is macOS `say`, the default English voice mispronounces Japanese cues, "
        "so set a ja_JP voice such as Kyoko.",
    )
    parser.add_argument(
        "--voice-profiles-json",
        help="JSON string array of voice profiles, each an object like "
        '{"name":"question","voice":"Kyoko","rate":180,"volume":0.6,"intent":"question"}. '
        "Profiles let each delivery intent use its own expression. A neutral profile is required.",
    )
    parser.add_argument("--tts-timeout", type=float, default=8.0, help="Seconds allowed for one TTS call.")
    parser.add_argument("--policy", type=Path, help="Controller policy JSON produced by an experiment.")
    parser.add_argument("--analyzer-command", nargs="+", help="External semantic analyzer command.")
    parser.add_argument("--analyzer-command-json", help="JSON string array for analyzer command arguments.")
    parser.add_argument("--analyzer-timeout", type=float, default=8.0)
    parser.add_argument("--sources", nargs="*", type=Path, help="Trusted local sources for RETRIEVE.")
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("data/personalization.json"),
        help="Explicit-feedback profile; adaptation can only suppress or fade cues.",
    )
    parser.add_argument("--log", type=Path, default=Path("logs/voice-session.jsonl"))
    parser.add_argument("--endpoint-silence-ms", type=int, default=700)
    parser.add_argument(
        "--max-decision-latency-ms",
        type=int,
        default=1_500,
        help="Suppress a candidate if analysis finishes after this delivery window.",
    )
    parser.add_argument("--cooldown-ms", type=int, default=5_000)
    parser.add_argument(
        "--mid-speech",
        action="store_true",
        help="While the user is still speaking, insert very short context-aware cues "
        "derived from the user's own vocabulary every few seconds.",
    )
    parser.add_argument("--mid-speech-interval-ms", type=int, default=2_500)
    parser.add_argument("--mid-speech-cooldown-ms", type=int, default=2_000)
    parser.add_argument("--mid-speech-confidence", type=float, default=0.6)
    parser.add_argument(
        "--mid-speech-rotate",
        action="store_true",
        help="Opt-in legacy mode: keep inserting a rotating pool of canonical cues "
        "even when no context-aware operation matches. Context-aware derivation is the default.",
    )
    parser.add_argument(
        "--event-timeout-seconds", type=float, default=1_900,
        help="Terminate a streaming adapter that exceeds the session wall-time limit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ControllerConfig.from_json_file(args.policy) if args.policy else ControllerConfig()
        profile = PersonalizationProfile.load(args.profile)
        event_command = parse_command_json(args.event_command_json, args.event_command, label="event")
        tts_command = parse_command_json(args.tts_command_json, args.tts_command, label="tts")
        if tts_command and args.tts_voice:
            tts_command = [*tts_command, "-v", args.tts_voice]
        analyzer_command = parse_command_json(
            args.analyzer_command_json, args.analyzer_command, label="analyzer"
        )
        speaker = (
            _speaker_from_profiles_json(args.voice_profiles_json, tts_command, args.tts_timeout)
            if args.voice_profiles_json or (tts_command is not None and args.tts_voice)
            else (CommandSpeaker(tts_command, args.tts_timeout) if tts_command else None)
        )
        speaker = speaker or NullSpeaker()
        analyzer: Analyzer = (
            HybridAnalyzer(LLMAnalyzer(CommandCompletion(analyzer_command, args.analyzer_timeout)))
            if analyzer_command
            else HeuristicAnalyzer()
        )
        knowledge_index = KnowledgeIndex.from_paths(args.sources) if args.sources else None
        if knowledge_index is not None:
            analyzer = GroundedRetrievalAnalyzer(analyzer, knowledge_index)
        trusted_refs = knowledge_index.source_refs if knowledge_index is not None else frozenset()
        controller = (
            ObservationOnlyController(profile)
            if event_command and not tts_command
            else AdaptiveController(InterventionController(config, trusted_refs), profile)
        )
        mid_speech: MidSpeechInjector | None = None
        if args.mid_speech:
            if not tts_command:
                raise ValueError("--mid-speech requires a TTS delivery channel (--tts-command)")
            mid_speech = MidSpeechInjector(
                analyzer=HeuristicAnalyzer(),
                speaker=speaker,
                interval_ms=args.mid_speech_interval_ms,
                cooldown_ms=args.mid_speech_cooldown_ms,
                minimum_confidence=args.mid_speech_confidence,
                always_rotate=args.mid_speech_rotate,
                log_path=args.log,
            )
        session = VoiceSession(
            analyzer=analyzer,
            controller=controller,
            detector=TurnBoundaryDetector(args.endpoint_silence_ms),
            speaker=speaker,
            max_decision_latency_ms=args.max_decision_latency_ms,
            cooldown_ms=args.cooldown_ms,
            mid_speech=mid_speech,
        )
        if event_command:
            context = nullcontext(command_event_lines(event_command, args.event_timeout_seconds))
        elif str(args.events) == "-":
            context = nullcontext(sys.stdin)
        else:
            context = args.events.open("r", encoding="utf-8")
        with context as lines:
            results = (
                run_live_event_stream(lines, session, args.log)
                if event_command
                else run_event_stream(lines, session, args.log)
            )
        profile.save(args.profile)
        for result in results:
            cue = f" {result.decision.cue}" if result.decision.cue else ""
            print(f"[{result.decision.action.value}]{cue} :: {result.turn.text}")
        if session.mid_speech is not None:
            for cue in session.mid_speech.injected_cues:
                print(f"[MID-SPEECH] {cue}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
