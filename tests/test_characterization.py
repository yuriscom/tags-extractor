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
from text_keyword_extractor import ENRICHMENT_VERSION, to_enrichment_columns


@pytest.fixture(scope="session")
def nlp():
    return h.load_nlp()


@pytest.fixture(scope="session")
def features(nlp):
    """Parse each fixture once through the structured pipeline; reuse across tests."""
    return {filename: h.features_for_file(nlp, filename) for filename in h.INPUT_FILES}


def _assert_close(actual, expected, path=""):
    """Recursively compare JSON-like structures, using tolerance on numbers.

    Order matters: bool is checked before int/float (bool is an int subclass).
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), f"key mismatch at {path}"
        for key in expected:
            _assert_close(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), f"length mismatch at {path}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_close(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool):
        assert actual == expected, f"value mismatch at {path}: {actual!r} != {expected!r}"
    elif isinstance(expected, (int, float)):
        assert actual == pytest.approx(expected, rel=1e-9, abs=1e-9), f"value mismatch at {path}"
    else:
        assert actual == expected, f"value mismatch at {path}: {actual!r} != {expected!r}"


@pytest.mark.parametrize("filename", h.INPUT_FILES)
def test_output_matches_golden(nlp, filename):
    golden = h.golden_path(filename)
    assert golden.exists(), (
        f"Missing golden file {golden}. Generate it first with:\n"
        f"    python tests/generate_golden.py"
    )

    result = h.extract_for_file(nlp, filename)
    expected = json.loads(golden.read_text(encoding="utf-8"))

    # The ranking order (the output contract) must match exactly...
    assert list(result.keys()) == list(expected.keys())

    # ...and each score must match within a small tolerance. Scores are floats built
    # from many multiplications, so exact equality is too brittle; ranking + near-equal
    # values is the meaningful contract.
    for key in expected:
        assert result[key] == pytest.approx(expected[key], rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("filename", h.INPUT_FILES)
def test_keywords_match_golden(features, filename):
    golden_path = h.keywords_golden_path(filename)
    assert golden_path.exists(), (
        f"Missing golden file {golden_path}. Generate it with: python tests/generate_golden.py")
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    _assert_close(h.keywords_golden(features[filename]), expected)


@pytest.mark.parametrize("filename", h.INPUT_FILES)
def test_keywords_wrap_legacy_scores(nlp, features, filename):
    """keywords must carry exactly the same {term: score} map as extract()."""
    legacy = h.extract_for_file(nlp, filename)
    wrapped = {keyword.text: keyword.score for keyword in features[filename].keywords}
    assert list(wrapped.keys()) == list(legacy.keys())
    for key in legacy:
        assert wrapped[key] == pytest.approx(legacy[key], rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("filename", h.INPUT_FILES)
def test_entities_match_golden(features, filename):
    golden_path = h.entities_golden_path(filename)
    assert golden_path.exists(), (
        f"Missing golden file {golden_path}. Generate it with: python tests/generate_golden.py")
    expected = json.loads(golden_path.read_text(encoding="utf-8"))
    _assert_close(h.entities_golden(features[filename]), expected)


@pytest.mark.parametrize("filename", h.INPUT_FILES)
def test_enrichment_columns_are_consistent(features, filename):
    """The flattened columns must be faithful derivations of the structured result."""
    result = features[filename]
    cols = to_enrichment_columns(result)

    # Nested JSON round-trips back to the structured payloads.
    _assert_close(json.loads(cols["keywords_json"]), h.keywords_golden(result))
    _assert_close(json.loads(cols["entities_json"]), h.entities_golden(result))

    # Flat keyword arrays mirror the structured lists.
    assert json.loads(cols["keyword_texts_json"]) == [k.normalized for k in result.keywords]
    assert json.loads(cols["entity_texts_json"]) == [e.normalized for e in result.entities]
    assert json.loads(cols["entity_labels_json"]) == sorted({e.label for e in result.entities})
    assert json.loads(cols["entity_pairs_json"]) == [
        f"{e.label}:{e.normalized}" for e in result.entities]

    # Metadata: versioned, timestamped, no error on the happy path.
    assert cols["enrichment_version"] == ENRICHMENT_VERSION
    assert cols["enrichment_error"] is None
    assert cols["enrichment_processed_at"]  # non-empty ISO timestamp
