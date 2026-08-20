from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import StrEnum
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from time import monotonic, perf_counter
from typing import Callable, Iterable, Iterator, Protocol, Sequence

from .core import (
    Action,
    Analysis,
    Analyzer,
    Decision,
    HeuristicAnalyzer,
    InterventionController,
    SessionContext,
    State,
    safe_analysis,
)
from .knowledge_state import KnowledgeStore
from .structure import SessionStructure
from .unified_analyzer import (
    MidSpeechUnifiedAnalyzer,
    UnifiedAnalyzerConfig,
    detect_spontaneous_reuse,
    record_cue_delivery,
)


class VoiceEventType(StrEnum):
    SPEECH_STARTED = "speech_started"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    SPEECH_ENDED = "speech_ended"
    TICK = "tick"
    INTERVENTIONS_PAUSED = "interventions_paused"
    INTERVENTIONS_RESUMED = "interventions_resumed"
    PARTICIPANT_STOP = "participant_stop"
    ADAPTER_WARNING = "adapter_warning"
    SESSION_ENDED = "session_ended"


@dataclass(frozen=True)
class VoiceEvent:
    type: VoiceEventType
    at_ms: int
    text: str | None = None

    def __post_init__(self) -> None:
        if self.at_ms < 0:
            raise ValueError("event at_ms must be non-negative")
        needs_text = self.type in {
            VoiceEventType.TRANSCRIPT_PARTIAL,
            VoiceEventType.TRANSCRIPT_FINAL,
            VoiceEventType.ADAPTER_WARNING,
        }
        if needs_text and not (self.text and self.text.strip()):
            raise ValueError(f"{self.type.value} requires non-empty text")

    @classmethod
    def from_json(cls, raw: str) -> "VoiceEvent":
        payload = json.loads(raw)
        if not isinstance(payload, dict) or not set(payload) <= {"type", "at_ms", "text"}:
            raise ValueError("voice event must contain only type, at_ms, and optional text")
        return cls(
            type=VoiceEventType(payload["type"]),
            at_ms=int(payload["at_ms"]),
            text=payload.get("text"),
        )


@dataclass(frozen=True)
class CompletedTurn:
    text: str
    started_at_ms: int
    ended_at_ms: int
    boundary_at_ms: int


@dataclass
class TurnBoundaryDetector:
    endpoint_silence_ms: int = 700
    transcript_grace_ms: int = 3_000

    def __post_init__(self) -> None:
        if self.endpoint_silence_ms < 0:
            raise ValueError("endpoint_silence_ms must be non-negative")
        if self.transcript_grace_ms < self.endpoint_silence_ms:
            raise ValueError("transcript_grace_ms must cover endpoint_silence_ms")
        self._speaking = False
        self._started_at_ms: int | None = None
        self._ended_at_ms: int | None = None
        self._final_fragments: list[str] = []
        self._partial: str | None = None
        self._last_at_ms = -1

    @property
    def speaking(self) -> bool:
        return self._speaking

    def feed(self, event: VoiceEvent) -> CompletedTurn | None:
        if event.at_ms < self._last_at_ms:
            raise ValueError("voice events must be ordered by at_ms")
        self._last_at_ms = event.at_ms

        if event.type is VoiceEventType.SPEECH_STARTED:
            previous: CompletedTurn | None = None
            if self._ended_at_ms is not None:
                previous = self._complete_if_ready(event.at_ms)
            self._speaking = True
            self._ended_at_ms = None
            if self._started_at_ms is None:
                self._started_at_ms = event.at_ms
            return previous
        if event.type is VoiceEventType.TRANSCRIPT_PARTIAL:
            self._partial = event.text.strip() if event.text else None
            return None
        if event.type is VoiceEventType.TRANSCRIPT_FINAL:
            self._final_fragments.append(event.text.strip())
            self._partial = None
            if self._started_at_ms is None:
                self._started_at_ms = event.at_ms
            return None
        if event.type is VoiceEventType.SPEECH_ENDED:
            self._speaking = False
            self._ended_at_ms = event.at_ms
            return self._complete_if_ready(event.at_ms)
        if event.type in {VoiceEventType.TICK, VoiceEventType.SESSION_ENDED}:
            return self._complete_if_ready(event.at_ms)
        return None

    def _complete_if_ready(self, at_ms: int, force: bool = False) -> CompletedTurn | None:
        if self._speaking or self._started_at_ms is None or self._ended_at_ms is None:
            return None
        if not force and at_ms - self._ended_at_ms < self.endpoint_silence_ms:
            return None
        text = " ".join(self._final_fragments).strip()
        if not text and self._partial:
            text = self._partial.strip()
        if not text:
            # ASR hypotheses can arrive after VAD reports speech_ended. Keep the
            # turn open for a bounded grace period instead of dropping it at the
            # first endpoint tick.
            if at_ms - self._ended_at_ms < self.transcript_grace_ms:
                return None
            self._reset()
            return None
        turn = CompletedTurn(text, self._started_at_ms, self._ended_at_ms, at_ms)
        self._reset()
        return turn

    def _reset(self) -> None:
        self._speaking = False
        self._started_at_ms = None
        self._ended_at_ms = None
        self._final_fragments.clear()
        self._partial = None


class Speaker(Protocol):
    def speak(self, text: str) -> None: ...


@dataclass(frozen=True)
class VoiceProfile:
    """Expression parameters for one delivery intent.

    The provider is neutral: the `voice`, `rate`, and `volume` fields are
    appended to the TTS command only when the command is a macOS `say`
    invocation (voice/rate/volume recognized as `-v`, `-r`, `--volume=`).
    For any other command, only the base args are used and the utterance is
    appended unchanged, so the profile never breaks a custom provider.
    """

    name: str
    command: tuple[str, ...]
    voice: str | None = None
    rate: int | None = None
    volume: float | None = None
    intent: str = "neutral"

    def to_command(self, utterance: str) -> list[str]:
        parts = list(self.command)
        if "say" in parts and len(parts) == 1:
            if self.voice:
                parts += ["-v", self.voice]
            if self.rate:
                parts += ["-r", str(self.rate)]
            if self.volume is not None:
                parts += [f"--volume={self.volume:.2f}"]
        parts.append(utterance)
        return parts


class SpeechCancelled(RuntimeError):
    """Raised when user speech cancels an in-progress synthesized cue."""


def _voice_intent_for_analysis(analysis: Analysis) -> str:
    action = analysis.proposed_action
    if action is Action.BACKCHANNEL:
        return "backchannel"
    if action is Action.PERTURB:
        return "perturb"
    if action is Action.CONSTRAIN:
        return "constrain"
    if action is Action.VERIFY:
        return "verify"
    if action is Action.MIRROR:
        return "mirror"
    return "question"


class NullSpeaker:
    def speak(self, text: str) -> None:
        del text


class CommandSpeaker:
    """Invokes a TTS command with the utterance as its final argument."""

    def __init__(
        self,
        command: Sequence[str] | VoiceProfile,
        timeout_seconds: float = 8.0,
    ) -> None:
        if isinstance(command, VoiceProfile):
            self._profile = command
        else:
            if not command:
                raise ValueError("speaker command must not be empty")
            self._profile = VoiceProfile(
                name="neutral", command=tuple(command), intent="neutral"
            )
        self._timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._cancelled = False

    def to_command(self, text: str) -> list[str]:
        return self._profile.to_command(text)

    def speak(self, text: str) -> None:
        process = subprocess.Popen(
            self.to_command(text),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        with self._lock:
            self._process = process
            self._cancelled = False
        try:
            process.communicate(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise RuntimeError("speaker command timed out") from exc
        finally:
            with self._lock:
                cancelled = self._cancelled
                self._process = None
        if cancelled:
            raise SpeechCancelled("speaker command cancelled by resumed user speech")
        if process.returncode != 0:
            raise RuntimeError(f"speaker command failed ({process.returncode})")

    def cancel(self) -> None:
        with self._lock:
            process = self._process
            if process is not None and process.poll() is None:
                self._cancelled = True
                process.terminate()


class ProfileSpeaker:
    """Serves a named delivery intent by selecting a matching VoiceProfile.

    The intent is part of the cue context (for example `question`,
    `backchannel`, `mirror`, `constrain`, `verify`). Profiles are declarative
    and may differ in voice, rate, and volume so that each intent has the
    least-disruptive expression. If no profile matches, the neutral profile
    (or the first registered one) is used. The decision to deliver is always
    made by the controller upstream; this speaker only changes *how* a cue
    sounds.
    """

    def __init__(self, profiles: Iterable[VoiceProfile], timeout_seconds: float = 8.0) -> None:
        self._profiles = list(profiles)
        if not self._profiles:
            raise ValueError("at least one voice profile is required")
        self._by_intent: dict[str, VoiceProfile] = {
            profile.intent: profile for profile in self._profiles
        }
        if "neutral" not in self._by_intent:
            raise ValueError("voice profiles must include a neutral profile")
        self._neutral = self._by_intent["neutral"]
        self._workers: dict[str, CommandSpeaker] = {}

    def _speaker_for(self, intent: str) -> CommandSpeaker:
        profile = self._by_intent.get(intent, self._neutral)
        speaker = self._workers.get(profile.name)
        if speaker is None:
            speaker = CommandSpeaker(profile)
            self._workers[profile.name] = speaker
        return speaker

    def speak(self, text: str, *, intent: str = "neutral") -> None:
        self._speaker_for(intent).speak(text)

    def to_command_for_intent(self, intent: str, text: str) -> list[str]:
        return self._speaker_for(intent).to_command(text)

    def cancel(self) -> None:
        for speaker in self._workers.values():
            speaker.cancel()

    def registered_intents(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_intent))


@dataclass(frozen=True)
class VoiceTurnResult:
    turn: CompletedTurn
    decision: Decision
    decision_latency_ms: int
    state_observed_at_ms: int
    state_expires_at_ms: int
    candidate_id: str
    candidate_created_at_ms: int
    candidate_expires_at_ms: int
    candidate_cooldown_ms: int
    timeline_origin: str
    delivery_status: str


class MidSpeechInjector:
    """Delivers short cognitive cues while the user is still speaking.

    The core idea of CUELOOP is not to wait for silence and then respond: it
    is to insert a minimal cue every few seconds during ongoing speech so that
    the user's own reasoning strategy is nudged and the pretrained model's
    knowledge is distilled into the user's mind. This injector runs on the
    evolving partial transcript and emits very short single-word cues that are

    - context-sensitive: derived from the user's own vocabulary, structural gaps
      vs. source knowledge, and knowledge state tracking
    - very short: one cognitive operation expressed in a few characters
    - non-repetitive and non-evaluative
    - suppressed during OVERLOAD and never delivered after the turn has ended
    - faded when the user shows evidence of internalizing the concept

    The analysis combines heuristic state detection, structural diff awareness,
    and knowledge state tracking for consistent intervention decisions.
    """

    INTERVENTION_ACTIONS: frozenset[Action] = frozenset(
        {Action.QUESTION, Action.MIRROR, Action.PERTURB, Action.CONSTRAIN}
    )

    # Explicit opt-in rotation mode (--mid-speech-rotate) keeps a stable,
    # predictable rhythm of canonical cues. DISABLED by default: the unified
    # analyzer path is the primary mechanism.
    ROTATION_CUES: tuple[str, ...] = (
        "何に依存？",
        "具体的には？",
        "逆なら？",
        "条件は？",
        "結論は？",
        "根拠は？",
    )

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        speaker: Speaker | None = None,
        interval_ms: int = 2_500,
        cooldown_ms: int = 2_000,
        minimum_confidence: float = 0.6,
        max_cue_characters: int = 8,
        min_text_characters: int = 4,
        always_rotate: bool = False,
        log_path: Path | None = None,
        source_structure: SessionStructure | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        if interval_ms < 500:
            raise ValueError("mid-speech interval_ms must be at least 500")
        if cooldown_ms < 0:
            raise ValueError("mid-speech cooldown_ms must be non-negative")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("mid-speech minimum_confidence must be between 0 and 1")
        if max_cue_characters < 1:
            raise ValueError("mid-speech max_cue_characters must be positive")
        if min_text_characters < 1:
            raise ValueError("mid-speech min_text_characters must be positive")
        if max_cue_characters > (10 if always_rotate else 8):
            raise ValueError("mid-speech max_cue_characters is too large for live cues")

        unified_config = UnifiedAnalyzerConfig(
            minimum_confidence=minimum_confidence,
            source_structure=source_structure,
            structural_minimum_confidence=0.3,
            knowledge_store=knowledge_store,
            max_cue_characters=16,
            max_mid_speech_cue_characters=max_cue_characters,
        )
        self.unified_analyzer = MidSpeechUnifiedAnalyzer(unified_config, analyzer)
        self.speaker = speaker or NullSpeaker()
        self.interval_ms = interval_ms
        self.cooldown_ms = cooldown_ms
        self.minimum_confidence = minimum_confidence
        self.max_cue_characters = max_cue_characters
        self.min_text_characters = min_text_characters
        self.always_rotate = always_rotate
        self.log_path = log_path
        self.knowledge_store = knowledge_store
        self.timeline_origin = datetime.now(timezone.utc).isoformat()
        self.delivered_count = 0
        self.injected_cues: list[str] = []
        self._last_injection_at_ms: int | None = None
        self._recent_partial: str = ""
        self._accumulated_partial: str = ""
        self._speaking = False
        self._rotation_index = 0
        self._last_cue: str | None = None
        self._used_cues_this_session: set[str] = set()

    def feed(self, event: VoiceEvent, context: SessionContext | None = None) -> str | None:
        """Return the spoken cue when one was delivered during speech.

        ``context`` carries the completed prior turns so the unified analyzer can
        compute structural differences against source knowledge while the user
        is still speaking. Without it, the structural branch is inert and only
        vocabulary echoes would be delivered.
        """
        if event.type is VoiceEventType.SPEECH_STARTED:
            self._speaking = True
            self._recent_partial = ""
            self._accumulated_partial = ""
            self._last_injection_at_ms = event.at_ms
            self._last_cue = None
            self._used_cues_this_session = set()
            # Reset unified analyzer session state
            if hasattr(self.unified_analyzer, "reset_session"):
                self.unified_analyzer.reset_session()
            return None
        if event.type is VoiceEventType.SPEECH_ENDED or event.type is VoiceEventType.SESSION_ENDED:
            self._speaking = False
            self._recent_partial = ""
            self._accumulated_partial = ""
            return None
        if not self._speaking:
            return None
        if event.type is VoiceEventType.TRANSCRIPT_PARTIAL:
            self._recent_partial = event.text.strip()
        elif event.type is VoiceEventType.TRANSCRIPT_FINAL:
            self._recent_partial = event.text.strip()
        else:
            return None
        if self._recent_partial:
            self._accumulated_partial = self._recent_partial
        now = event.at_ms
        if self._last_injection_at_ms is not None and now - self._last_injection_at_ms < self.interval_ms:
            return None
        text = self._accumulated_partial
        if len(text) < self.min_text_characters:
            return None
        if self.always_rotate:
            cue = self._next_rotation_cue()
            analysis = self.unified_analyzer.analyze(text, context)
            if analysis.state is State.OVERLOAD:
                self._last_injection_at_ms = now
                return None
            return self._deliver(cue, now, text, analysis)
        # Use unified analyzer for context-aware, structurally-informed cue selection
        analysis = self.unified_analyzer.analyze(text, context)
        if analysis.state is State.OVERLOAD:
            self._last_injection_at_ms = now
            return None
        # Mid-speech is a method-steering channel, not an answering channel:
        # an explicit knowledge question is honored at the turn boundary, never
        # mid-utterance. Suppressing RETRIEVE here keeps a source fragment
        # (which also exceeds the 8-char live-cue limit) out of live speech.
        if analysis.proposed_action is Action.RETRIEVE:
            self._last_injection_at_ms = now
            return None
        # Detect spontaneous reuse of previously cued concepts
        if self.knowledge_store is not None and self._recent_partial:
            detect_spontaneous_reuse(self.knowledge_store, self._recent_partial)
        # Unified analyzer returns None cue if no intervention warranted
        if analysis.cue is None:
            return None
        # Avoid repeating the exact same cue
        cue = analysis.cue
        if cue == self._last_cue or cue in self._used_cues_this_session:
            return None
        return self._deliver(cue, now, text, analysis)

    def _next_rotation_cue(self) -> str:
        cue = self.ROTATION_CUES[self._rotation_index % len(self.ROTATION_CUES)]
        self._rotation_index += 1
        return cue

    def _deliver(self, cue: str, now: int, text: str, analysis: Analysis) -> str:
        self._last_injection_at_ms = now
        self._last_cue = cue
        self._used_cues_this_session.add(cue)
        # Record cue delivery for knowledge state tracking
        if self.knowledge_store is not None:
            record_cue_delivery(
                self.knowledge_store,
                cue,
                source_session_id="",
                confidence=analysis.confidence,
            )
        intent = _voice_intent_for_analysis(analysis)
        if isinstance(self.speaker, ProfileSpeaker):
            self.speaker.speak(cue, intent=intent)
        else:
            self.speaker.speak(cue)
        self.delivered_count += 1
        self.injected_cues.append(cue)
        if self.log_path is not None:
            self._append_log(cue, now, text, analysis)
        return cue

    def _append_log(
        self, cue: str, at_ms: int, source: str, analysis: Analysis
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "kind": "mid_speech_cue",
            "timestamp": (datetime.fromisoformat(self.timeline_origin) + timedelta(milliseconds=at_ms)).isoformat(),
            "at_ms": at_ms,
            "source": source,
            "cue": cue,
            "state": analysis.state.value,
            "confidence": analysis.confidence,
            "action": analysis.proposed_action.value,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class VoiceSession:
    """Consumes streaming ASR/VAD events and delivers only post-boundary actions."""

    def __init__(
        self,
        analyzer: Analyzer | None = None,
        controller: InterventionController | None = None,
        detector: TurnBoundaryDetector | None = None,
        speaker: Speaker | None = None,
        context: SessionContext | None = None,
        max_decision_latency_ms: int = 1_500,
        cooldown_ms: int = 5_000,
        mid_speech: MidSpeechInjector | None = None,
    ) -> None:
        if max_decision_latency_ms < 1:
            raise ValueError("max_decision_latency_ms must be positive")
        if cooldown_ms < 0:
            raise ValueError("cooldown_ms must be non-negative")
        selected_detector = detector or TurnBoundaryDetector()
        if max_decision_latency_ms <= selected_detector.endpoint_silence_ms:
            raise ValueError("max_decision_latency_ms must exceed endpoint silence")
        self.analyzer = analyzer or HeuristicAnalyzer()
        self.controller = controller or InterventionController()
        self.detector = selected_detector
        self.speaker = speaker or NullSpeaker()
        self.context = context or SessionContext()
        self.mid_speech = mid_speech
        self.max_decision_latency_ms = max_decision_latency_ms
        self.cooldown_ms = cooldown_ms
        self.timeline_origin = datetime.now(timezone.utc).isoformat()
        self.interventions_paused = False
        self.interventions_ever_paused = False
        self.participant_requested_stop = False
        self.adapter_warning_count = 0
        self.analyzer_failure_count = 0
        self.resume_latencies_ms: list[int] = []
        self._awaiting_resume_after_ms: int | None = None
        self._last_delivery_at_ms: int | None = None

    def feed(
        self,
        event: VoiceEvent,
        *,
        deliver: bool = True,
        cancellation_check: Callable[[], str | None] | None = None,
    ) -> VoiceTurnResult | None:
        if event.type is VoiceEventType.SPEECH_STARTED and self._awaiting_resume_after_ms is not None:
            self.resume_latencies_ms.append(max(0, event.at_ms - self._awaiting_resume_after_ms))
            self._awaiting_resume_after_ms = None
        if event.type is VoiceEventType.ADAPTER_WARNING:
            self.adapter_warning_count += 1
            return None
        if event.type is VoiceEventType.INTERVENTIONS_PAUSED:
            self.interventions_paused = True
            self.interventions_ever_paused = True
            return None
        if event.type is VoiceEventType.INTERVENTIONS_RESUMED:
            self.interventions_paused = False
            return None
        if event.type is VoiceEventType.PARTICIPANT_STOP:
            self.interventions_paused = True
            self.interventions_ever_paused = True
            self.participant_requested_stop = True
            return None
        if self.mid_speech is not None and not self.interventions_paused:
            if event.type in {
                VoiceEventType.TRANSCRIPT_PARTIAL,
                VoiceEventType.TRANSCRIPT_FINAL,
                VoiceEventType.SPEECH_STARTED,
                VoiceEventType.SPEECH_ENDED,
                VoiceEventType.SESSION_ENDED,
            }:
                try:
                    self.mid_speech.feed(event, context=self.context)
                except SpeechCancelled:
                    self._awaiting_resume_after_ms = event.at_ms
        turn = self.detector.feed(event)
        if turn is None:
            return None
        analysis_started = perf_counter()
        available_analysis_ms = self.max_decision_latency_ms - max(
            0, event.at_ms - turn.ended_at_ms
        )
        analysis, analysis_timed_out = self._analyze_with_timeout(
            turn.text, available_analysis_ms
        )
        if analysis_timed_out or (analysis.confidence == 0 and any(
            item.description.startswith("analyzer") for item in analysis.evidence
        )):
            self.analyzer_failure_count += 1
        decision_latency_ms = max(
            0,
            event.at_ms - turn.ended_at_ms + round((perf_counter() - analysis_started) * 1_000),
        )
        cancellation_reason = cancellation_check() if cancellation_check is not None else None
        if event.type is VoiceEventType.SPEECH_STARTED:
            cancellation_reason = "user resumed speaking before delivery"
        if self.interventions_paused:
            decision = Decision(Action.SILENCE, None, "interventions paused by user control", analysis)
        elif cancellation_reason:
            decision = Decision(Action.SILENCE, None, cancellation_reason, analysis)
        elif analysis_timed_out or decision_latency_ms > self.max_decision_latency_ms:
            decision = Decision(Action.SILENCE, None, "candidate expired before delivery", analysis)
        elif (
            self._last_delivery_at_ms is not None
            and turn.boundary_at_ms - self._last_delivery_at_ms < self.cooldown_ms
        ):
            decision = Decision(Action.SILENCE, None, "voice intervention cooldown active", analysis)
        else:
            decision = self.controller.decide(analysis)
        self.context.add(turn.text)
        candidate_id = "candidate-" + hashlib.sha256(
            f"{turn.started_at_ms}:{turn.boundary_at_ms}:{turn.text}".encode("utf-8")
        ).hexdigest()[:12]
        if decision.reason == "candidate expired before delivery":
            delivery_status = "EXPIRED"
        elif decision.action is Action.SILENCE and analysis.proposed_action is not Action.SILENCE:
            delivery_status = "CANCELLED"
        else:
            delivery_status = "DELIVERED"
        result = VoiceTurnResult(
            turn,
            decision,
            decision_latency_ms,
            event.at_ms,
            event.at_ms + 10_000,
            candidate_id,
            turn.boundary_at_ms,
            turn.ended_at_ms + self.max_decision_latency_ms,
            self.cooldown_ms,
            self.timeline_origin,
            delivery_status,
        )
        if deliver:
            self.deliver(result)
        return result

    def _analyze_with_timeout(self, text: str, available_ms: int) -> tuple[Analysis, bool]:
        if available_ms <= 0:
            return safe_analysis("analyzer delivery deadline exhausted"), True
        result: queue.Queue[Analysis | BaseException] = queue.Queue(maxsize=1)
        context = SessionContext(self.context.max_turns, self.context.max_characters)
        for turn in self.context.snapshot():
            context.add(turn)

        def analyze() -> None:
            try:
                result.put(self.analyzer.analyze(text, context))
            except BaseException as exc:
                result.put(exc)

        worker = threading.Thread(target=analyze, daemon=True)
        worker.start()
        try:
            outcome = result.get(timeout=available_ms / 1_000)
        except queue.Empty:
            cancel = getattr(self.analyzer, "cancel", None)
            if callable(cancel):
                cancel()
            return safe_analysis("analyzer delivery deadline exceeded"), True
        if isinstance(outcome, BaseException):
            return safe_analysis(f"analyzer failure: {type(outcome).__name__}"), False
        return outcome, False

    def deliver(self, result: VoiceTurnResult) -> bool:
        if result.delivery_status != "DELIVERED":
            return False
        try:
            if result.decision.action is Action.BACKCHANNEL:
                self.speaker.speak("うん。")
            elif result.decision.cue:
                self.speaker.speak(result.decision.cue)
        except SpeechCancelled:
            return False
        if result.decision.action is not Action.SILENCE:
            self._awaiting_resume_after_ms = result.candidate_created_at_ms
            self._last_delivery_at_ms = result.candidate_created_at_ms
        return True

    def cancel_active_delivery(self) -> None:
        cancel = getattr(self.speaker, "cancel", None)
        if callable(cancel):
            cancel()

    def cancel_result(self, result: VoiceTurnResult, reason: str) -> VoiceTurnResult:
        """Cancel a pending live candidate when newer user activity invalidates it."""
        if result.decision.action is Action.SILENCE:
            return result
        cancel_delivery = getattr(self.controller, "cancel_delivery", None)
        if callable(cancel_delivery):
            cancel_delivery(result.decision)
        return VoiceTurnResult(
            result.turn,
            Decision(Action.SILENCE, None, reason, result.decision.analysis),
            result.decision_latency_ms,
            result.state_observed_at_ms,
            result.state_expires_at_ms,
            result.candidate_id,
            result.candidate_created_at_ms,
            result.candidate_expires_at_ms,
            result.candidate_cooldown_ms,
            result.timeline_origin,
            "CANCELLED",
        )


def read_events(lines: Iterable[str]) -> Iterator[VoiceEvent]:
    last_at_ms = -1
    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = VoiceEvent.from_json(stripped)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid voice event on line {line_number}: {exc}") from exc
        if event.at_ms < last_at_ms:
            raise ValueError(f"voice event on line {line_number} is out of order")
        last_at_ms = event.at_ms
        yield event


def append_voice_result(path: Path, result: VoiceTurnResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "kind": "voice_turn",
        "turn": asdict(result.turn),
        "decision_latency_ms": result.decision_latency_ms,
        "state_observed_at_ms": result.state_observed_at_ms,
        "state_expires_at_ms": result.state_expires_at_ms,
        "candidate": {
            "id": result.candidate_id,
            "action": result.decision.analysis.proposed_action.value,
            "text": result.decision.analysis.cue,
            "intended_effect": _intended_effect(result.decision.analysis.proposed_action),
            "state": {
                "state": result.decision.analysis.state.value,
                "confidence": result.decision.analysis.confidence,
                "evidence": [asdict(item) for item in result.decision.analysis.evidence],
                "observed_at": _event_timestamp(result, result.state_observed_at_ms),
                "expires_at": _event_timestamp(result, result.state_expires_at_ms),
                "notes": None,
            },
            "confidence": result.decision.analysis.confidence,
            "created_at": _event_timestamp(result, result.candidate_created_at_ms),
            "expires_at": _event_timestamp(result, result.candidate_expires_at_ms),
            "delivery_constraints": {
                "allow_while_speaking": False,
                "requires_pause": True,
                "minimum_pause_ms": result.turn.boundary_at_ms - result.turn.ended_at_ms,
                "cooldown_ms": result.candidate_cooldown_ms,
                "max_delivery_latency_ms": (
                    result.candidate_expires_at_ms - result.candidate_created_at_ms
                ),
            },
            "source_refs": list(result.decision.analysis.source_refs),
        },
        "candidate_lifecycle": {
            "status": result.delivery_status,
            "reason": result.decision.reason,
            "timestamp": _event_timestamp(
                result,
                result.turn.ended_at_ms + result.decision_latency_ms
            ),
        },
        "decision": {
            "action": result.decision.action.value,
            "cue": result.decision.cue,
            "reason": result.decision.reason,
            "state": result.decision.analysis.state.value,
            "confidence": result.decision.analysis.confidence,
            "evidence": [asdict(item) for item in result.decision.analysis.evidence],
            "source_refs": list(result.decision.analysis.source_refs),
        },
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _event_timestamp(result: VoiceTurnResult, at_ms: int) -> str:
    return (datetime.fromisoformat(result.timeline_origin) + timedelta(milliseconds=at_ms)).isoformat()


def _intended_effect(action: Action) -> str:
    return {
        Action.SILENCE: "preserve self-generation",
        Action.BACKCHANNEL: "preserve the user's turn",
        Action.QUESTION: "elicit one missing relation",
        Action.MIRROR: "externalize current structure",
        Action.PERTURB: "open an alternative reasoning path",
        Action.CONSTRAIN: "reduce the search space",
        Action.RETRIEVE: "supply one grounded knowledge fragment",
        Action.VERIFY: "defer a claim for verification",
    }[action]


def run_event_stream(
    lines: Iterable[str], session: VoiceSession, log_path: Path | None = None
) -> list[VoiceTurnResult]:
    results: list[VoiceTurnResult] = []
    for event in read_events(lines):
        result = session.feed(event)
        if result is not None:
            results.append(result)
            if log_path is not None:
                append_voice_result(log_path, result)
    return results


def run_live_event_stream(
    lines: Iterable[str], session: VoiceSession, log_path: Path | None = None
) -> list[VoiceTurnResult]:
    """Consume live events while analysis runs, cancelling on resumed speech.

    A single-worker analyzer preserves controller/context order. The producer
    continues reading VAD events so a speech_started event can invalidate an
    intervention before it is spoken.
    """

    events: queue.Queue[VoiceEvent | BaseException | None] = queue.Queue()

    def produce() -> None:
        try:
            for event in read_events(lines):
                if event.type is VoiceEventType.SPEECH_STARTED:
                    session.cancel_active_delivery()
                events.put(event)
        except BaseException as exc:
            events.put(exc)
        finally:
            events.put(None)

    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    results: list[VoiceTurnResult] = []
    pending_events: deque[VoiceEvent] = deque()
    input_finished = False
    while True:
        if pending_events:
            item: VoiceEvent | BaseException | None = pending_events.popleft()
        elif input_finished:
            break
        else:
            item = events.get()
        if item is None:
            input_finished = True
            continue
        if isinstance(item, BaseException):
            raise item
        result_holder: queue.Queue[VoiceTurnResult | BaseException | None] = queue.Queue(maxsize=1)
        cancellation_reason: list[str | None] = [None]

        def analyze_event() -> None:
            try:
                result_holder.put(
                    session.feed(
                        item,
                        deliver=False,
                        cancellation_check=lambda: cancellation_reason[0],
                    )
                )
            except BaseException as exc:
                result_holder.put(exc)

        worker = threading.Thread(target=analyze_event, daemon=True)
        worker.start()
        while worker.is_alive():
            try:
                queued = events.get(timeout=0.005)
            except queue.Empty:
                continue
            if queued is None:
                input_finished = True
                continue
            if isinstance(queued, BaseException):
                raise queued
            pending_events.append(queued)
            if queued.type is VoiceEventType.SPEECH_STARTED:
                cancellation_reason[0] = "user resumed speaking before delivery"
        worker.join()
        result_or_error = result_holder.get()
        if isinstance(result_or_error, BaseException):
            raise result_or_error
        result = result_or_error
        resumed = any(event.type is VoiceEventType.SPEECH_STARTED for event in pending_events)
        if result is not None:
            if resumed and result.decision.action is not Action.SILENCE:
                result = session.cancel_result(result, "user resumed speaking before delivery")
            else:
                delivered = session.deliver(result)
                if result.decision.action is not Action.SILENCE and not delivered:
                    result = session.cancel_result(result, "user resumed speaking during delivery")
            results.append(result)
            if log_path is not None:
                append_voice_result(log_path, result)
    return results


def command_event_lines(command: Sequence[str], timeout_seconds: float | None = None) -> Iterator[str]:
    if not command:
        raise ValueError("event command must not be empty")
    # A temporary file prevents verbose adapters from blocking on a full stderr
    # pipe while their JSONL stdout is consumed incrementally.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_file, text=True)
        assert process.stdout is not None
        output: queue.Queue[str | None] = queue.Queue()

        def read_stdout() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output.put(line)
            finally:
                output.put(None)

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()
        deadline = monotonic() + timeout_seconds if timeout_seconds is not None else None
        try:
            while True:
                remaining = None if deadline is None else deadline - monotonic()
                if remaining is not None and remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout_seconds)
                try:
                    line = output.get(timeout=remaining)
                except queue.Empty as exc:
                    raise subprocess.TimeoutExpired(command, timeout_seconds) from exc
                if line is None:
                    break
                yield line
            returncode = process.wait(timeout=None if deadline is None else max(0.001, deadline - monotonic()))
        except subprocess.TimeoutExpired as exc:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            raise RuntimeError("event command timed out") from exc
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
            raise
        finally:
            process.stdout.close()
            reader.join(timeout=0.2)
        if returncode != 0:
            stderr_file.seek(0)
            stderr = stderr_file.read().strip()[:240]
            raise RuntimeError(f"event command failed ({returncode}): {stderr}")
