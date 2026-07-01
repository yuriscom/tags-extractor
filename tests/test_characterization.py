"""Characterization tests: pin the current extractor output byte-for-byte.

These lock in the existing behavior before the cleanup refactor. Every cleanup phase
must keep these green. Both the term order and the exact float scores are compared,
because ranking order is part of the output contract.

If a test fails during cleanup, the refactor changed behavior — investigate rather
than regenerating the goldens.
"""
import json

import pytest

import extraction_harness as h


@pytest.fixture(scope="session")
def nlp():
    return h.load_nlp()


@pytest.mark.parametrize("filename", h.INPUT_FILES)
def test_output_matches_golden(nlp, filename):
    golden = h.golden_path(filename)
    assert golden.exists(), (
        f"Missing golden file {golden}. Generate it first with:\n"
        f"    python tests/generate_golden.py"
    )

    result = h.extract_for_file(nlp, filename)
    expected = json.loads(golden.read_text(encoding="utf-8"))

    # Compare ordered items: both the ranking order and the exact scores must match.
    assert list(result.items()) == list(expected.items())
