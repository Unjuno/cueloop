import threading
import time
import unittest

from mvp.async_session import AsyncSession
from mvp.core import Gate


class TestAsyncDecision(unittest.TestCase):
    def test_nonblocking_boundary_and_later_delivery(self):
        release = threading.Event()
        def slow(system, user):
            release.wait(2)
            return '{"action":"CUE","text":"比較軸"}'
        session = AsyncSession(slow)
        try:
            session.update("性能の違い", 0)
            start = time.perf_counter()
            self.assertEqual(session.emit(100).action, "NONE")
            self.assertLess(time.perf_counter() - start, 0.1)
            release.set()
            for _ in range(100):
                result = session.emit(500)
                if result.action == "CUE":
                    break
                time.sleep(0.005)
            self.assertEqual(result.text, "比較軸")
            self.assertEqual(session.emit(501).action, "NONE")
        finally:
            release.set()
            session.close()

    def test_old_result_cannot_be_delivered(self):
        old_release = threading.Event()
        new_release = threading.Event()
        def complete(system, user):
            if user == "古い文脈":
                old_release.wait(2)
                return '{"action":"CUE","text":"古い"}'
            new_release.wait(2)
            return '{"action":"CUE","text":"新しい軸"}'
        session = AsyncSession(complete)
        try:
            session.update("古い文脈", 0)
            session.update("新しい文脈", 100)
            old_release.set()
            new_release.set()
            for _ in range(100):
                result = session.emit(400)
                if result.action == "CUE":
                    break
                time.sleep(0.005)
            self.assertEqual(result.text, "新しい軸")
        finally:
            old_release.set()
            new_release.set()
            session.close()

    def test_failure_abstains(self):
        def fail(*args):
            raise RuntimeError("network down")
        session = AsyncSession(fail)
        try:
            session.update("考えている", 0)
            for _ in range(100):
                result = session.emit(500)
                if result.reason != "model still running":
                    break
                time.sleep(0.005)
            self.assertEqual(result.action, "NONE")
        finally:
            session.close()

    def test_speaking_does_not_deliver(self):
        session = AsyncSession(lambda *_: '{"action":"CUE","text":"別の軸"}')
        try:
            session.update("考えている", 0)
            for _ in range(100):
                result = session.emit(500, speaking=True)
                if result.reason != "model still running":
                    break
                time.sleep(0.005)
            self.assertEqual(result.action, "NONE")
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
