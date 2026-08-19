from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import Action
from .personalization import FeedbackRating, PersonalizationProfile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record explicit feedback for one delivered intervention.")
    parser.add_argument("rating", choices=[rating.value for rating in FeedbackRating])
    parser.add_argument("--action", choices=[action.value for action in Action if action is not Action.SILENCE], required=True)
    parser.add_argument("--cue")
    parser.add_argument("--profile", type=Path, default=Path("data/personalization.json"))
    args = parser.parse_args(argv)
    try:
        action = Action(args.action)
        if action in {
            Action.QUESTION, Action.MIRROR, Action.PERTURB, Action.CONSTRAIN, Action.RETRIEVE
        } and not args.cue:
            raise ValueError(f"{action.value} feedback requires --cue")
        profile = PersonalizationProfile.load(args.profile)
        profile.record_feedback(action, args.cue, FeedbackRating(args.rating))
        profile.save(args.profile)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"recorded={args.rating} action={args.action} profile={args.profile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
