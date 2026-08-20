from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from cueloop.knowledge_state import (
    KnowledgeAcquisition,
    KnowledgeState,
    KnowledgeStore,
    extract_keywords,
    speech_contains_keywords,
)


class ExtractKeywordsTests(unittest.TestCase):
    def test_extracts_nouns_from_cue(self) -> None:
        kw = extract_keywords("何に依存？")
        self.assertIsInstance(kw, tuple)
        self.assertTrue(len(kw) >= 1)
        self.assertTrue(all(isinstance(k, str) for k in kw))
        # 依存 should be extracted (何 is filtered as single-char)
        self.assertIn("依存", kw)

    def test_removes_particles(self) -> None:
        kw = extract_keywords("根拠は？")
        # 根拠 should be extracted, は should be removed
        self.assertIn("根拠", kw)

    def test_max_keywords_respected(self) -> None:
        kw = extract_keywords("比較軸は何か依存する前提条件", max_keywords=2)
        self.assertLessEqual(len(kw), 2)

    def test_empty_text_returns_empty(self) -> None:
        kw = extract_keywords("")
        self.assertEqual(kw, ())


class SpeechContainsKeywordsTests(unittest.TestCase):
    def test_all_keywords_must_appear(self) -> None:
        self.assertTrue(speech_contains_keywords("依存する前提がある", ["依存", "前提"]))
        self.assertFalse(speech_contains_keywords("依存する", ["依存", "前提"]))

    def test_case_insensitive(self) -> None:
        self.assertTrue(speech_contains_keywords("Root cause is clear", ["root", "cause"]))

    def test_empty_keywords_returns_false(self) -> None:
        self.assertFalse(speech_contains_keywords("anything", []))


class KnowledgeAcquisitionTests(unittest.TestCase):
    def test_valid_creation(self) -> None:
        now = datetime.now(timezone.utc)
        ac = KnowledgeAcquisition(
            concept_key="abc123",
            concept_text="何に依存？",
            state=KnowledgeState.CUE_ASSISTED,
            confidence=0.7,
            keywords=("依存",),
            source_session_id="s1",
            observed_at=now.isoformat(),
            expires_at=(now + timedelta(days=30)).isoformat(),
            last_state_change_at=now.isoformat(),
        )
        self.assertEqual(ac.state, KnowledgeState.CUE_ASSISTED)
        self.assertFalse(ac.is_expired(now + timedelta(days=10)))
        self.assertTrue(ac.is_expired(now + timedelta(days=40)))

    def test_rejects_empty_concept_text(self) -> None:
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, "concept_text"):
            KnowledgeAcquisition(
                concept_key="abc",
                concept_text="  ",
                state=KnowledgeState.CUE_ASSISTED,
                confidence=0.7,
                keywords=("x",),
                source_session_id="s1",
                observed_at=now.isoformat(),
                expires_at=(now + timedelta(days=30)).isoformat(),
                last_state_change_at=now.isoformat(),
            )

    def test_rejects_out_of_range_confidence(self) -> None:
        now = datetime.now(timezone.utc)
        with self.assertRaisesRegex(ValueError, "confidence"):
            KnowledgeAcquisition(
                concept_key="abc",
                concept_text="test",
                state=KnowledgeState.CUE_ASSISTED,
                confidence=1.5,
                keywords=("x",),
                source_session_id="s1",
                observed_at=now.isoformat(),
                expires_at=(now + timedelta(days=30)).isoformat(),
                last_state_change_at=now.isoformat(),
            )

    def test_round_trip(self) -> None:
        now = datetime.now(timezone.utc)
        ac = KnowledgeAcquisition(
            concept_key="abc123",
            concept_text="何に依存？",
            state=KnowledgeState.SPONTANEOUSLY_REUSED,
            confidence=0.8,
            keywords=("依存",),
            source_session_id="s1",
            observed_at=now.isoformat(),
            expires_at=(now + timedelta(days=30)).isoformat(),
            last_state_change_at=now.isoformat(),
            evidence=("cue delivered",),
        )
        restored = KnowledgeAcquisition.from_dict(ac.to_dict())
        self.assertEqual(restored.concept_key, ac.concept_key)
        self.assertEqual(restored.state, KnowledgeState.SPONTANEOUSLY_REUSED)
        self.assertEqual(restored.evidence, ("cue delivered",))


class KnowledgeStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)
        self.store = KnowledgeStore(retention_days=90)

    def test_record_cue_delivery(self) -> None:
        ac = self.store.record_cue_delivery(
            "何に依存？",
            source_session_id="s1",
            now=self.now,
        )
        self.assertEqual(ac.state, KnowledgeState.CUE_ASSISTED)
        self.assertEqual(len(self.store.acquisitions), 1)

    def test_duplicate_cue_is_not_re_recorded(self) -> None:
        self.store.record_cue_delivery("何に依存？", source_session_id="s1", now=self.now)
        self.store.record_cue_delivery("何に依存？", source_session_id="s1", now=self.now)
        self.assertEqual(len(self.store.acquisitions), 1)

    def test_different_cues_are_separate(self) -> None:
        self.store.record_cue_delivery("何に依存？", source_session_id="s1", now=self.now)
        self.store.record_cue_delivery("根拠は？", source_session_id="s1", now=self.now)
        self.assertEqual(len(self.store.acquisitions), 2)

    def test_spontaneous_reuse_detection(self) -> None:
        self.store.record_cue_delivery("依存する前提", source_session_id="s1", now=self.now)
        upgraded = self.store.detect_spontaneous_reuse(
            "依存関係と前提条件を整理する",
            now=self.now,
        )
        self.assertEqual(len(upgraded), 1)
        self.assertEqual(upgraded[0].state, KnowledgeState.SPONTANEOUSLY_REUSED)
        self.assertIn("spontaneous reuse", upgraded[0].evidence[-1])

    def test_spontaneous_reuse_requires_all_keywords(self) -> None:
        self.store.record_cue_delivery("依存する前提", source_session_id="s1", now=self.now)
        # Only contains one of two keywords
        upgraded = self.store.detect_spontaneous_reuse("依存がある", now=self.now)
        self.assertEqual(len(upgraded), 0)

    def test_no_reuse_for_already_reused(self) -> None:
        self.store.record_cue_delivery("依存する前提", source_session_id="s1", now=self.now)
        self.store.detect_spontaneous_reuse("依存関係と前提条件", now=self.now)
        # Second detection should not upgrade again
        upgraded = self.store.detect_spontaneous_reuse("依存関係と前提条件", now=self.now)
        self.assertEqual(len(upgraded), 0)

    def test_fading_level_returns_zero_for_unknown_cue(self) -> None:
        self.assertEqual(self.store.get_fading_level("未知のキュー"), 0)

    def test_fading_level_returns_one_after_spontaneous_reuse(self) -> None:
        self.store.record_cue_delivery("依存する前提", source_session_id="s1", now=self.now)
        self.store.detect_spontaneous_reuse("依存関係と前提条件", now=self.now)
        self.assertEqual(self.store.get_fading_level("依存する前提"), 1)

    def test_expired_acquisition_returns_fading_level_two(self) -> None:
        old = self.now - timedelta(days=200)
        self.store = KnowledgeStore(retention_days=30)
        self.store.record_cue_delivery("依存する前提", source_session_id="s1", now=old)
        self.assertEqual(self.store.get_fading_level("依存する前提"), 2)

    def test_prune_expired(self) -> None:
        old = self.now - timedelta(days=200)
        self.store.record_cue_delivery("依存は？", source_session_id="s1", now=old)
        self.assertEqual(len(self.store.acquisitions), 1)
        pruned = self.store.prune_expired(self.now)
        self.assertEqual(pruned, 1)
        self.assertEqual(len(self.store.acquisitions), 0)

    def test_serialization_round_trip(self) -> None:
        self.store.record_cue_delivery("何に依存？", source_session_id="s1", now=self.now)
        data = self.store.to_list()
        restored = KnowledgeStore.from_list(data, retention_days=90)
        self.assertEqual(len(restored.acquisitions), 1)
        self.assertEqual(restored.acquisitions[0].concept_text, "何に依存？")


if __name__ == "__main__":
    unittest.main()
