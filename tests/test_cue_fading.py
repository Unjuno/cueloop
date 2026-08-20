from __future__ import annotations

import unittest
from datetime import datetime, timezone

from cueloop.cue_fading import CueFadingPolicy, keyword_cue
from cueloop.knowledge_state import KnowledgeStore


class CueFadingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime.now(timezone.utc)

    def test_no_fading_without_store(self) -> None:
        policy = CueFadingPolicy(None)
        result = policy.apply("何に依存？")
        self.assertTrue(result.should_deliver)
        self.assertEqual(result.fading_level, 0)
        self.assertEqual(result.faded_cue, "何に依存？")

    def test_no_fading_for_unknown_cue(self) -> None:
        store = KnowledgeStore(retention_days=90)
        policy = CueFadingPolicy(store)
        result = policy.apply("何に依存？")
        self.assertTrue(result.should_deliver)
        self.assertEqual(result.fading_level, 0)

    def test_keyword_fading_after_spontaneous_reuse(self) -> None:
        store = KnowledgeStore(retention_days=90)
        store.record_cue_delivery("依存する前提", source_session_id="s1", now=self.now)
        store.detect_spontaneous_reuse("依存関係と前提条件", now=self.now)
        policy = CueFadingPolicy(store)
        result = policy.apply("依存する前提")
        self.assertTrue(result.should_deliver)
        self.assertEqual(result.fading_level, 1)
        self.assertNotEqual(result.faded_cue, "依存する前提")
        self.assertIn("keyword", result.reason)

    def test_suppress_after_explicit_retention(self) -> None:
        from cueloop.knowledge_state import KnowledgeAcquisition, KnowledgeState

        store = KnowledgeStore(retention_days=90)
        ac = store.record_cue_delivery("依存する前提", source_session_id="s1", now=self.now)
        # Force state to EXPLICITLY_RETRAINED
        updated = KnowledgeAcquisition(
            concept_key=ac.concept_key,
            concept_text=ac.concept_text,
            state=KnowledgeState.EXPLICITLY_RETRAINED,
            confidence=ac.confidence,
            keywords=ac.keywords,
            source_session_id=ac.source_session_id,
            observed_at=ac.observed_at,
            expires_at=ac.expires_at,
            last_state_change_at=self.now.isoformat(),
        )
        idx = store.acquisitions.index(ac)
        store.acquisitions[idx] = updated
        store._index[updated.concept_key] = updated

        policy = CueFadingPolicy(store)
        result = policy.apply("依存する前提")
        self.assertFalse(result.should_deliver)
        self.assertEqual(result.fading_level, 2)

    def test_empty_cue_returns_none(self) -> None:
        policy = CueFadingPolicy(KnowledgeStore())
        result = policy.apply("")
        self.assertFalse(result.should_deliver)


class KeywordCueTests(unittest.TestCase):
    def test_extracts_single_keyword(self) -> None:
        result = keyword_cue("何に依存？")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_max_characters_applied(self) -> None:
        result = keyword_cue("何に依存？", max_characters=2)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result), 2)

    def test_returns_none_for_empty(self) -> None:
        result = keyword_cue("")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
