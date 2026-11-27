"""Regression tests for importing components from the notebooks directory."""

from __future__ import annotations

import json
import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = ROOT / "notebooks"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


@unittest.skipUnless(VENV_PYTHON.is_file(), "component-lab/.venv Python is required")
class NotebookImportTests(unittest.TestCase):
    def run_python(self, code: str, *, without_site_packages: bool = False) -> subprocess.CompletedProcess[str]:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        command = [str(VENV_PYTHON)]
        if without_site_packages:
            command.append("-S")
        command.extend(["-c", code])
        return subprocess.run(
            command,
            cwd=NOTEBOOKS_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def test_import_components_with_venv_from_notebooks_cwd(self) -> None:
        result = self.run_python("import components; print(components.__file__)")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("components/__init__.py", result.stdout)

    def test_notebook_bootstrap_imports_without_site_packages(self) -> None:
        notebook = json.loads(
            (NOTEBOOKS_DIR / "offline_walkthrough.ipynb").read_text(encoding="utf-8")
        )
        first_code_cell = next(
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        source = "".join(first_code_cell["source"])
        result = self.run_python(
            source + "\nimport components\nprint(components.__file__)",
            without_site_packages=True,
        )
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("components/__init__.py", result.stdout)


if __name__ == "__main__":
    unittest.main()
