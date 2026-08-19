import unittest

from cueloop.commands import parse_command_json
from cueloop.session_cli import build_parser as build_session_parser


class CommandParsingTests(unittest.TestCase):
    def test_json_command_preserves_option_like_child_arguments(self) -> None:
        args = build_session_parser().parse_args(
            [
                "run",
                "--event-command-json",
                '["adapter","--locale","ja-JP","--duration","600"]',
                "--goal",
                "test",
            ]
        )
        command = parse_command_json(args.event_command_json, args.event_command, label="event")
        self.assertEqual(command, ["adapter", "--locale", "ja-JP", "--duration", "600"])

    def test_invalid_command_json_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_command_json('{"cmd":"x"}', None, label="event")


if __name__ == "__main__":
    unittest.main()
