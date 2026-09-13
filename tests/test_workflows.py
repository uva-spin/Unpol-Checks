"""Lightweight regression tests for both public verification workflows."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_main_result(self):
        subprocess.run([sys.executable, "main-result/scripts/verify_main_result.py"], cwd=ROOT, check=True)

    def test_finite_y(self):
        subprocess.run([sys.executable, "finite-y/scripts/verify_finite_y.py"], cwd=ROOT, check=True)
