"""Generate (or regenerate) golden output files for the characterization tests.

Run this ONLY when the current extractor output is the intended, reviewed baseline:

    python tests/generate_golden.py

During the behavior-preserving cleanup, you should NOT need to regenerate goldens —
if the test fails, the refactor changed behavior. Regeneration is expected later,
when we deliberately fix the deferred scoring/ambiguity issues.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import extraction_harness as h  # noqa: E402


def main():
    h.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading spaCy model...")
    nlp = h.load_nlp()
    for filename in h.INPUT_FILES:
        result = h.extract_for_file(nlp, filename)
        out_path = h.golden_path(filename)
        out_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  wrote {out_path.relative_to(h.REPO_ROOT)}  ({len(result)} terms)")
    print("Done.")


if __name__ == "__main__":
    main()
