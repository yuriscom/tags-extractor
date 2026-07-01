import sys
from pathlib import Path

# Ensure both the tests dir (for extraction_harness) and the repo root (for the
# project modules) are importable regardless of pytest's rootdir detection.
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
for path in (str(TESTS_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
