# Cleanup Plan

> **Scope:** This is a strict, behavior-preserving lift-and-shift. Every step must leave the
> `tags` / `keywords` output byte-for-byte identical, verified against golden files
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

### #2 — Scoring unit is inconsistent across branches — RESOLVED (leave as-is, documented)
The one branch that differs is the ambiguity-tie fallback in `_merge_terms`: it adds
`rank_score` alone (no entity-weight multiplier), whereas the other branches add
`rank_score * <entity-weight>`. This makes the bare shared token land on a smaller scale
than the full phrases.
- **Resolution:** kept as-is on purpose. When a shared token ties across entities, keeping it
  at its bare `rank_score` avoids dropping it, while the full phrases still receive their proper
  `rank_score * weight` via their own unique tokens (e.g. `Stanley -> Morgan Stanley`,
  `Freeman -> Morgan Freeman`). In practice both full terms end up with sensible, comparable
  scores and the low-scored bare token is harmless. Behavior is unchanged; the branch now carries
  a comment explaining the intent.

### #3 — Reads from `tags` while still building `tags` — RESOLVED
`multi_weight = (entity_weights[text] + tags[text]) * bonus` read the accumulated output score
back in and added it to a raw weight (mixing units).
- **Resolution:** the multi-word amplification is intentionally kept, but reformulated with a
  per-phrase **growth factor** (`phrase_growth`, starts at 1.0). The recurrence
  `weight + accumulated_score` is algebraically `weight × growth`, so now
  `multi_weight = entity_weights[text] * growth * bonus` — a weight times a dimensionless growth
  factor, with no score folded into a weight, and nothing read back from the `tags` output.
  Growth advances as `growth *= (1 + rank_score * bonus)` on each phrase contribution.
  This is numerically equivalent to the old formula (identical ranking; scores differ only by
  floating-point noise ~1e-13), so the multi-word amplification behavior is preserved. The
  characterization test now checks exact ranking order plus values within a small tolerance.
  (Side note: the earlier claim that this was "order-dependent" was inaccurate — the closed form
  `weight × (∏(1 + rank_i·bonus) − 1)` is symmetric, so token order doesn't change the result.)

### #4 — Ambiguity resolution is hard to follow — RESOLVED
The winner-selection was opaque (`most_common()` called three times inline).
- **Resolution:** the logic is now isolated in `_resolve_ambiguity` with a docstring, and the
  winner check computes `ranked = phrase_counts.most_common()` once with explanatory comments.
  Behavior is intentionally unchanged (golden tests still green): if the phrase weights aren't all
  equal, the highest-weighted phrase wins; if they're all tied, there is no winner and the bare
  token is kept. The only remaining edge case — a multi-way tie at the top resolving by `Counter`
  insertion order — is deterministic and considered acceptable, so no behavioral change was made.

### #5 — Top-N cutoff is unreliable — RESOLVED
`if i > number: break` cut off on the sorted `node_weight` *input index*, which (a) is off-by-one,
(b) counts terms not emitted tags, and (c) dropped lower-ranked terms whose contributions still feed
the growth of top phrases — so it actually changed the top-N ordering, not just the count.
- **Resolution:** `_merge_terms` now scores **all** terms (no early cutoff), and `get_tags` treats
  `number` as an **output cap**: sort all tags and return the top `number`. Measurements showed the
  merge loop is sub-millisecond even for the largest fixture (the real cost is spaCy parsing, which
  was never bounded by `number` anyway), so processing everything is essentially free and yields a
  stable, complete ranking. `number` now means "at most N tags" (you may still get fewer, since many
  terms don't emit a tag). This changes output (more complete, re-ordered top-N), so the golden files
  were regenerated to the new baseline.

> Note: #4 becomes much easier to address after Phase 3 isolates the ambiguity logic into its own
> method, and #2/#3 become clearer once `_merge_terms` is a self-contained function. So the cleanup
> is a natural setup for this follow-up pass.