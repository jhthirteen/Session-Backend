"""Shared pytest config for top-level tests/.

Ensures the repo root is importable so tests can do
`from src.data_tooling import ...` regardless of CWD.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
