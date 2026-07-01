# Cleanup Plan

> **Scope:** This is a strict, behavior-preserving lift-and-shift. Every step must leave the
> `tags` / `candidate_terms` output byte-for-byte identical, verified against golden files
> (Phase 0). Anything that would change output is explicitly **out of scope** and tracked in
> [Deferred behavioral issues](#deferred-behavioral-issues-post-cleanup) for a follow-up pass.

## Phase 0 — Safety net (do this first)

Before touching anything, lock in current behavior so we can prove the refactor changes nothing.

1. Pick a few representative inputs (e.g. `data/content2.txt`, `content3.txt`, and 1–2 from a batch CSV if available).
2. Run the current code and save the exact JSON output as "golden" files under `tests/golden/`.
3. Add a tiny characterization test (`tests/test_characterization.py`) that runs `TagsExtractor(nlp).extract(...)` on those inputs and asserts the output equals the golden files.

This is the backstop — after each cleanup step I re-run it and confirm zero diff. Without it, refactoring code with mixed scoring units and order-dependence is risky.

## Phase 1 — Remove dead weight (no logic change)

1. **Delete `textRank.py`** — legacy test/prototype file, not in any path. *(Decision B)*
2. **Move `cli.py` and `server.py` into `archive/`** — both are broken against the current API
   (they call `TagsExtractor()` with no `nlp`). Fixing them is out of scope for now; archive
   rather than fix or delete. *(Decision A)*
3. Remove dead code inside `text_keyword_extractor.py`:
   - `get_keywords()` method + its commented call site (line 160).
   - `multiWinsMode` global + the commented `multiWinsMode` block (lines 236–239).
4. Minor hygiene: drop stray trailing semicolons, the commented-out `config`/`lemma_fix_map` blocks in `spacy_wrapper.py`, and the dead `download_dir` S3 helper in `spacy_module_loader.py` if you confirm it's truly unused (README says `SOURCE_BUCKET` is legacy).

## Phase 2 — Naming & readability (no logic change)

Rename confusing locals/typos for clarity (all internal, output-safe):
- `occurance` → `occurrence`
- `wordsAr` → `words` / `word_parts`
- `cnt` → `entity_weights`, `mapFullTerm` → `full_term_map`, etc.
- Consistent `snake_case`, add short docstrings on the non-obvious methods.

## Phase 3 — Decompose `get_tags` into named helpers (no logic change)

Split the ~90-line method into readable, single-responsibility pieces that compute *exactly* the same values:

- `_build_entity_weights(doc)` → returns `(entity_weights, occurrence, full_term_map)` — the first loop over `doc.ents`.
- `_resolve_ambiguity(full_terms, entity_weights)` → the winner-selection logic.
- `_merge_terms(node_weight, entity_weights, full_term_map, number)` → the second loop that produces the final scores.
- `get_tags()` becomes a thin orchestrator calling these.

Each extraction is a pure "move code into a method" step, verified against the golden output.

## Phase 4 — Centralize config (structure only, same defaults)

Keep this **minimal** *(Decision C)*: do the simplest possible extraction, no more.
Move the module-level globals (`mwEntityLabelWeight`, `fullTermWeightBonusCoefficient`,
`excludePos`, `multipleOccuranceMultiplier`) into a small `ExtractorConfig` dataclass with
**identical default values**, and have `TagsExtractor` accept it (defaulting to the current
values). Behavior is unchanged, but this is the seam that R6 (YAML config) plugs into later.
Do not add YAML loading, validation, or per-field logic here — that belongs to the feature phase.

## Testing (Decision D)

- Add `pytest` to `requirements.txt` and create a `tests/` folder.
- Phase 0's characterization test is the core deliverable; keep it running green after every phase.
- No need for broad unit coverage during cleanup — the golden-output test is the safety net.

---

## Resolved decisions

- **(A)** `cli.py` / `server.py` → **move to `archive/`** (out of scope to fix now).
- **(B)** `textRank.py` → **delete**.
- **(C)** Config extraction → **keep minimal**, simplest extraction only; improve later.
- **(D)** Tests → **yes**, add `pytest` + `tests/`.

---

## Deferred behavioral issues (post-cleanup)

These are the known correctness/readability problems in `get_tags` that the cleanup **intentionally
does not fix**, because fixing them changes output. Once the lift-and-shift is done and the golden
tests are green, we tackle these as a separate, deliberate pass (each will require updating the
golden files with reviewed, intended output). No solution is committed yet — captured here so they
aren't lost.

### #2 — Scoring unit is inconsistent across branches
The final `tags` dict receives values on different scales depending on the code path: the
ambiguity-no-winner branch adds raw `value` (TextRank only), the single-word branch adds
`value * weight`, and the multi-word branches add `value * singleWeight` / `value * multiWeight`.
As a result, two terms' scores aren't necessarily on the same scale.
- **Direction (TBD):** decide on one scoring formula and apply it uniformly across all branches.

### #3 — Reads from `tags` while still building `tags`
`extraWeight = tags[txt] if txt in tags else 0` then `multiWeight = (cnt[txt] + extraWeight) * weightBonus`
reads an already-accumulated *score* and folds it back into a *count*-scale computation, making the
result order-dependent and mixing units.
- **Direction (TBD):** separate the accumulation pass from the read pass; don't feed output back
  into intermediate weight math.

### #4 — Ambiguity resolution is hard to follow
The winner-selection (`pCnt.most_common()[0][1] != pCnt.most_common()[-1][1]`) is opaque and relies
on `most_common()` ordering for ties.
- **Direction (TBD):** rewrite as an explicit, documented rule (e.g. clear tie-break policy) once
  Phase 3 has isolated it into `_resolve_ambiguity`.

### #5 — Top-N cutoff is unreliable
`if i > number: break` counts position in the sorted `node_weight`, not the number of terms actually
emitted, and is off-by-one (`> number` yields `number + 1`). Because nodes get skipped/merged, the
returned count is unpredictable.
- **Direction (TBD):** cut off on the count of emitted terms after final sorting, not on the input
  index.

> Note: #4 becomes much easier to address after Phase 3 isolates the ambiguity logic into its own
> method, and #2/#3 become clearer once `_merge_terms` is a self-contained function. So the cleanup
> is a natural setup for this follow-up pass.