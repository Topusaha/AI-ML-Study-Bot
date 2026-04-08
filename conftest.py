"""
Pytest configuration — ensures the repo root is on sys.path so that
imports like `from ml.parser import MarkdownParser` work in all test files.
"""
import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).parent))
