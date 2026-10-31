import unittest
from mvp.core import Decision, Gate, Session, parse

class TestDecisionSync(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse('{"action":"CUE","text":"比較軸"}').text, "比較軸")
        self.assertEqual(parse('not json').action, "NONE")
        self.assertEqual(parse('{"action":"CUE","text":"abcdefghijklmnopq"}').action, "NONE")

    def test_fresh_cue(self):
        s = Session(lambda system, user: '{"action":"CUE","text":"比較軸"}')
        s.update("性能が高い", 100)
        self.assertEqual(s.emit(500).text, "比較軸")
        self.assertEqual(s.emit(600).action, "NONE")

    def test_silence_duplicate_cooldown(self):
        g = Gate()
        c = Decision("CUE","比較軸")
        self.assertEqual(g.choose(c,candidate_revision=1,current_revision=1,created_ms=0,now_ms=100,transcript="比較軸").action,"NONE")
        self.assertEqual(g.choose(c,candidate_revision=1,current_revision=1,created_ms=0,now_ms=100,transcript="別").action,"CUE")
        self.assertEqual(g.choose(Decision("CUE","反例"),candidate_revision=2,current_revision=2,created_ms=200,now_ms=300,transcript="別").action,"NONE")

    def test_stale_expired_speaking(self):
        g = Gate()
        c = Decision("CUE","反例")
        for params in [dict(candidate_revision=1,current_revision=2,created_ms=0,now_ms=100),
                       dict(candidate_revision=1,current_revision=1,created_ms=0,now_ms=2000),
                       dict(candidate_revision=1,current_revision=1,created_ms=0,now_ms=100,speaking=True)]:
            self.assertEqual(g.choose(c,transcript="仮説",**params).action,"NONE")

    def test_fail_closed(self):
        def broken(*args): raise RuntimeError("offline")
        s = Session(broken)
        s.update("テスト", 0)
        self.assertEqual(s.emit(100).action, "NONE")

if __name__ == "__main__":
    unittest.main()
