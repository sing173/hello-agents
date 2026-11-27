"""Security and behavior tests for the read-only TerminalTool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from components.terminal import TerminalPolicyError, TerminalTool


class TerminalToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.workspace = root / "workspace"
        self.workspace.mkdir()
        (self.workspace / "hello.txt").write_text(
            "hello alpha\nsecond line\nthird line\n",
            encoding="utf-8",
        )
        (self.workspace / "big.txt").write_text("abcdefghij", encoding="utf-8")
        subdirectory = self.workspace / "sub"
        subdirectory.mkdir()
        (subdirectory / "nested.txt").write_text("nested beta\n", encoding="utf-8")
        self.outside = root / "outside.txt"
        self.outside.write_text("outside secret\n", encoding="utf-8")
        self.terminal = TerminalTool(self.workspace)

    def test_allowlisted_read_commands_execute_inside_workspace(self) -> None:
        pwd = self.terminal.run("pwd")
        listing = self.terminal.run("ls -la")
        content = self.terminal.run("cat hello.txt")
        first_line = self.terminal.run("head -n 1 hello.txt")
        matching = self.terminal.run("grep -n alpha hello.txt")
        nested = self.terminal.run("cat sub/nested.txt")

        self.assertEqual(0, pwd.exit_code)
        self.assertEqual(str(self.workspace), pwd.stdout.strip())
        self.assertIn("hello.txt", listing.stdout)
        self.assertIn("hello alpha", content.stdout)
        self.assertEqual("hello alpha", first_line.stdout.strip())
        self.assertIn("1:hello alpha", matching.stdout)
        self.assertIn("nested beta", nested.stdout)

    def test_absolute_paths_are_rejected(self) -> None:
        for command in (
            "cat /etc/passwd",
            "ls /tmp",
            "head -n 1 /etc/hosts",
            "/bin/cat hello.txt",
        ):
            with self.subTest(command=command):
                with self.assertRaises(TerminalPolicyError):
                    self.terminal.run(command)

    def test_parent_traversal_is_rejected_even_when_it_returns_inside(self) -> None:
        for command in (
            "cat ../outside.txt",
            "head -n 1 ../../etc/passwd",
            "cat sub/../hello.txt",
        ):
            with self.subTest(command=command):
                with self.assertRaises(TerminalPolicyError):
                    self.terminal.run(command)

    def test_shell_metacharacters_and_expansion_are_rejected(self) -> None:
        commands = [
            "cat hello.txt; id",
            "cat hello.txt | wc -l",
            "cat hello.txt && id",
            "cat hello.txt > copied.txt",
            "cat hello.txt < /etc/passwd",
            "cat $(id)",
            "cat `id`",
            "cat hello.txt\nid",
            "cat hello.txt &",
            "ls *.txt",
            "cat \"hello.txt\"",
            "cat 'hello.txt'",
            "cat hello.txt # comment",
        ]
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(TerminalPolicyError):
                    self.terminal.run(command)

    def test_non_allowlisted_commands_and_write_options_are_rejected(self) -> None:
        commands = [
            "rm hello.txt",
            "python3 -c pass",
            "sh -c id",
            "find . -exec cat hello.txt",
            "find . -delete",
            "grep -R alpha .",
            "find . -fprint output.txt",
        ]
        for command in commands:
            with self.subTest(command=command):
                with self.assertRaises(TerminalPolicyError):
                    self.terminal.run(command)

    def test_symlink_escape_is_rejected(self) -> None:
        link = self.workspace / "outside-link"
        try:
            link.symlink_to(self.outside)
        except OSError:
            self.skipTest("symbolic links are unavailable")
        with self.assertRaises(TerminalPolicyError):
            self.terminal.run("cat outside-link")

    def test_subprocess_is_invoked_as_argv_with_shell_disabled(self) -> None:
        completed = SimpleNamespace(returncode=0, stdout="ok\n", stderr="")
        with mock.patch(
            "components.terminal.tool.subprocess.run",
            return_value=completed,
        ) as subprocess_run:
            result = self.terminal.run("ls -la")

        self.assertEqual(0, result.exit_code)
        self.assertEqual("ok\n", result.stdout)
        arguments, keywords = subprocess_run.call_args
        self.assertIsInstance(arguments[0], list)
        self.assertEqual("-la", arguments[0][-1])
        self.assertFalse(keywords["shell"])

    def test_output_is_bounded_and_marked_truncated(self) -> None:
        terminal = TerminalTool(self.workspace, max_output_chars=5)
        result = terminal.run("cat big.txt")

        self.assertTrue(result.truncated)
        self.assertEqual("abcde", result.stdout)
        self.assertLessEqual(len(result.output), 5)


if __name__ == "__main__":
    unittest.main()