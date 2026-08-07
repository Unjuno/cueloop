from __future__ import annotations

from cuelloop.core import HeuristicAnalyzer, InterventionController


CASES = (
    ("flow", "この設計は前提と依存関係を分けて考える必要がある。反例も確認する。"),
    ("unfinished", "まだ比較軸を決めていないので"),
    ("stuck", "えっと、その、なんだろう、性能が良いと思う"),
    ("short", "短い"),
)


def main() -> int:
    analyzer = HeuristicAnalyzer()
    controller = InterventionController()

    for name, text in CASES:
        analysis = analyzer.analyze(text)
        decision = controller.decide(analysis)
        print(
            f"{name}\tstate={analysis.state.value}\tconfidence={analysis.confidence:.2f}"
            f"\taction={decision.action.value}\tcue={decision.cue or '-'}"
            f"\treason={decision.reason}"
        )

    print("cooldown-sequence")
    controller = InterventionController()
    for index in range(3):
        analysis = analyzer.analyze("えっと、その、なんだろう、性能が良い")
        decision = controller.decide(analysis)
        print(
            f"{index + 1}\taction={decision.action.value}\tcue={decision.cue or '-'}"
            f"\treason={decision.reason}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
