"""Guard the lab against accidental third-party runtime imports."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

COMPONENTS = Path(__file__).resolve().parents[1] / "components"


class StandardLibraryIsolationTests(unittest.TestCase):
    def test_component_imports_are_stdlib_or_local(self) -> None:
        allowed = set(sys.stdlib_module_names) | {"components"}
        for path in sorted(COMPONENTS.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                roots: set[str] = set()
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".", 1)[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    roots = {node.module.split(".", 1)[0]}
                with self.subTest(path=str(path.relative_to(COMPONENTS)), roots=roots):
                    self.assertTrue(roots <= allowed, f"unexpected imports in {path}: {roots - allowed}")


if __name__ == "__main__":
    unittest.main()