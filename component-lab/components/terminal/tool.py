"""Read-only, workspace-confined terminal execution without a shell."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ALLOWED_COMMANDS = {
    "pwd": frozenset(),
    "ls": frozenset(
        {"-a", "-A", "-l", "-h", "-1", "-d", "-R", "--all", "--almost-all", "--human-readable"}
    ),
    "cat": frozenset({"-n"}),
    "head": frozenset({"-n", "-c", "-q", "-v"}),
    "tail": frozenset({"-n", "-c", "-q", "-v"}),
    "wc": frozenset({"-l", "-w", "-c", "-m"}),
    "grep": frozenset(
        {"-i", "-n", "-r", "-l", "-L", "-F", "-E", "-w", "-v", "--include", "--exclude"}
    ),
    "find": frozenset({"-name", "-iname", "-type", "-maxdepth", "-mindepth", "-print"}),
}

_FORBIDDEN_SHELL_CHARACTERS = frozenset(";|&><$`\\\"'(){}[]*?~!#\n\r\t\x00^%")


class TerminalPolicyError(ValueError):
    """Raised when a command is outside the read-only allowlist."""


@dataclass(frozen=True, slots=True)
class TerminalResult:
    """Captured output from one allowlisted command."""

    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False

    @property
    def output(self) -> str:
        """Return stdout followed by a labeled stderr section when present."""
        if not self.stderr:
            return self.stdout
        return f"{self.stdout}\n[stderr]\n{self.stderr}"


class TerminalTool:
    """Explore a workspace using only read-only allowlisted argv calls."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        timeout_seconds: float = 10.0,
        max_output_chars: int = 100_000,
        allowed_commands: Sequence[str] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        if not self.workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive")

        if allowed_commands is None:
            selected = ALLOWED_COMMANDS
        else:
            unknown = set(allowed_commands) - set(ALLOWED_COMMANDS)
            if unknown:
                raise ValueError(f"unsupported command(s): {', '.join(sorted(unknown))}")
            selected = {
                command: ALLOWED_COMMANDS[command] for command in allowed_commands
            }
        self.allowed_commands = {
            command: frozenset(options) for command, options in selected.items()
        }
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def run(self, command: str) -> TerminalResult:
        """Validate, execute without a shell, and capture bounded output."""
        argv = self._validate_command(command)
        executable = shutil.which(
            argv[0],
            path=os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        )
        if executable is None:
            raise TerminalPolicyError(f"command executable is unavailable: {argv[0]}")

        environment = {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": str(self.workspace),
        }
        try:
            completed = subprocess.run(
                [executable, *argv[1:]],
                cwd=self.workspace,
                env=environment,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return TerminalResult(
                command=tuple(argv),
                exit_code=None,
                stdout=error.stdout or "",
                stderr="command timed out",
                timed_out=True,
            )

        stdout, stderr, truncated = self._bound_output(
            completed.stdout or "",
            completed.stderr or "",
        )
        return TerminalResult(
            command=tuple(argv),
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=truncated,
        )

    def _validate_command(self, command: str) -> list[str]:
        if not isinstance(command, str) or not command.strip():
            raise TerminalPolicyError("command must be a non-empty string")
        forbidden = sorted(set(command) & _FORBIDDEN_SHELL_CHARACTERS)
        if forbidden:
            raise TerminalPolicyError(
                f"shell metacharacters are not allowed: {''.join(forbidden)!r}"
            )
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as error:
            raise TerminalPolicyError(f"invalid command syntax: {error}") from error
        if not argv:
            raise TerminalPolicyError("command must contain an executable")

        name = argv[0]
        if "/" in name or name not in self.allowed_commands:
            raise TerminalPolicyError(f"command is not allowlisted: {name}")
        if name == "pwd" and len(argv) != 1:
            raise TerminalPolicyError("pwd accepts no arguments")

        for argument in argv[1:]:
            if argument.startswith("-"):
                self._validate_option(name, argument)
            else:
                self._validate_path(argument)

        self._validate_arity(name, argv[1:])
        return argv

    def _validate_option(self, command: str, argument: str) -> None:
        options = self.allowed_commands[command]
        if argument.startswith("--"):
            if "=" in argument:
                option_name, _ = argument.split("=", 1)
                if option_name not in options or option_name not in {
                    "--include",
                    "--exclude",
                }:
                    raise TerminalPolicyError(f"option is not allowed: {option_name}")
                return
            if argument not in options:
                raise TerminalPolicyError(f"option is not allowed: {argument}")
            return
        if argument in options:
            return
        if len(argument) > 2 and all(f"-{character}" in options for character in argument[1:]):
            return
        raise TerminalPolicyError(f"option is not allowed: {argument}")

    def _validate_path(self, argument: str) -> None:
        candidate = Path(argument)
        if candidate.is_absolute():
            raise TerminalPolicyError(f"absolute paths are not allowed: {argument}")
        if ".." in candidate.parts:
            raise TerminalPolicyError(f"parent traversal is not allowed: {argument}")
        resolved = (self.workspace / candidate).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError as error:
            raise TerminalPolicyError(f"path escapes the workspace: {argument}") from error

    @staticmethod
    def _validate_arity(command: str, arguments: Sequence[str]) -> None:
        positional = [argument for argument in arguments if not argument.startswith("-")]
        if command in {"cat", "head", "tail", "wc"} and not positional:
            raise TerminalPolicyError(f"{command} requires a workspace path")
        if command == "grep" and len(positional) < 2:
            raise TerminalPolicyError("grep requires a pattern and a workspace path")

    def _bound_output(self, stdout: str, stderr: str) -> tuple[str, str, bool]:
        remaining = self.max_output_chars
        truncated = False
        if len(stdout) > remaining:
            stdout = stdout[:remaining]
            truncated = True
            remaining = 0
        else:
            remaining -= len(stdout)
        if len(stderr) > remaining:
            stderr = stderr[:remaining]
            truncated = True
        return (stdout, stderr, truncated)