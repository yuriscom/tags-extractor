# Implementation Plan — Easy Bucket (R1, R3, R7)

> **Source spec:** [`entity_extractor_requirements.md`](./entity_extractor_requirements.md).
> This plan covers only the **Easy / mostly-mechanical** slice (~1.5 days): **R1** (rename
> `tags` → `keywords` + structured output), **R3** (separate `entities` from
> keyphrases), **R7** (Elasticsearch/OpenSearch-friendly JSON columns).
>
> **Scope discipline (mirrors the cleanup doc):** this is **additive and non-breaking**. The
> legacy `TagsExtractor.extract()` → `{term: score}` dict and `to_json()` stay **byte-for-byte
> unchanged**, because `lambda_index.py` and `batch_extract.py` depend on them. The new structured
> output is produced by *new* methods alongside the old ones. The existing golden characterization
> test must stay green throughout; new golden files are added for the new output. Anything that
> changes the legacy scores, removes reparsing, adds multi-field/offset evidence, or introduces a
> batch/YAML runner is **out of scope** and tracked in [Deferred](#deferred-to-later-buckets).

## What "Easy bucket" means here

| Req | In this plan | Deferred |
|-----|--------------|----------|
| **R1** rename `tags` → `keywords` | dataclasses + serializers wrapping the current scores | — |
| **R3** separate entity output | `entities` from `doc.ents`, grouped by `(normalized, label)`, configurable label filter/weights, cheap offsets from spaCy | richer sentence context tuning |
| **R7** ES/OpenSearch JSON columns | flattened `*_texts` / `*_labels` / `*_pairs` arrays + `json.dumps`; `version` / `processed_at` / `error` | true Arrow list/struct columns |
| R2 full evidence/offsets for keyphrases | entity offsets only (free from `doc.ents`) | keyphrase offset fallback + `approximate` flag |
| R4 avoid reparsing | entity path already uses `doc.ents` (no reparse) | removing `self.nlp(txt)` in the *keyphrase* path |
| R5 batch Parquet, `nlp.pipe()` | — | full `enrich_parquet.py` runner |
| R6 YAML config, multi-field weighting | new knobs added to the existing `ExtractorConfig` dataclass | YAML loading + per-field weights/aggregation |

Because R2/R4/R5/R6 are deferred, this phase stays **single-field** (one text blob, `source_fields`
defaults to `["content"]`), keyphrase `evidence` stays empty, and the legacy scoring is untouched.

---

## Phase 0 — Safety net (do this first)

Same discipline as the cleanup: prove we changed nothing we didn't mean to, and pin the new output.

1. Confirm the **existing** golden characterization test (`tests/test_characterization.py`) is green — it
   exercises the legacy `extract()` path and is our proof that R1/R3/R7 don't disturb legacy scores.
2. Add **new** golden fixtures for the structured output on the same inputs (`tests/fixtures/*.txt`):
   - `tests/golden/*.keywords.json`
   - `tests/golden/*.entities.json`
   - The R7 columns are **not** goldened (they embed a live timestamp and float-formatted JSON
     strings, which makes an exact-string golden brittle). Instead a **structural consistency test**
     asserts the flattened columns are faithful derivations of the structured result and that the
     metadata is present (`version`, non-empty `processed_at`, `error is None`).
3. Extend `tests/generate_golden.py` to emit these, and add assertions in `test_characterization.py`
   (recursive compare: exact key/label ordering, float scores via `pytest.approx(rel=1e-9, abs=1e-9)`,
   matching the existing tolerance convention).

**Invariant to assert:** the score map derived from `keywords`
(`{ct.text: ct.score}`) must equal the legacy `tags` golden exactly — this guarantees R1 is a pure
wrap, not a behavior change.

---

## Phase 1 — R1: Output models + `keywords` (mechanical)

**Goal:** stop calling the output `tags`; return a list of structured `Keyword` objects that
wrap the *existing* scores.

### Step 1.1 — Add lightweight dataclasses

Add to `text_keyword_extractor.py` (dataclasses, not Pydantic — enough for the POC):

```python
@dataclass
class Evidence:
    field: str
    start: int
    end: int
    text: str
    sentence: str | None = None
    approximate: bool = False        # reserved for R2; always False in this phase

@dataclass
class Keyword:
    text: str                        # readable surface form (current term key)
    normalized: str                  # lowercased, whitespace-collapsed
    type: str = "keyphrase"
    score: float = 0.0
    count: int = 1                   # placeholder in this phase (see Open decisions)
    source_fields: list[str] = field(default_factory=lambda: ["content"])
    evidence: list[Evidence] = field(default_factory=list)   # empty until R2

@dataclass
class EntityMention:
    field: str
    start: int
    end: int
    text: str
    sentence: str | None = None

@dataclass
class Entity:
    text: str
    normalized: str
    label: str
    score: float
    count: int
    label_weight: float
    is_multi_word: bool
    first_seen_field: str = "content"
    source_fields: list[str] = field(default_factory=lambda: ["content"])
    mentions: list[EntityMention] = field(default_factory=list)

@dataclass
class ArticleEnrichmentResult:
    keywords: list[Keyword] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    enrichment_version: str = ENRICHMENT_VERSION
    enrichment_processed_at: str = ""
    enrichment_error: str | None = None
```

### Step 1.2 — Add a new extraction entry point (keep the old one)

- **Leave `extract()`, `get_tags()`, `to_json()` untouched.** They remain the compatibility surface
  for `lambda_index.py` / `batch_extract.py`.
- Add `extract_features(self, text, labels, num=20) -> ArticleEnrichmentResult`:
  1. Runs the same `TextRank4Keyword.analyze()` + `get_tags()` internally (identical scores).
  2. Wraps each `(term, score)` into a `Keyword` (`type="keyphrase"`, `normalized=_normalize(term)`).
  3. Calls the new entity path (Phase 2) to fill `entities`.
  4. Stamps `enrichment_version` / `enrichment_processed_at`; wraps the body in try/except to set
     `enrichment_error` instead of raising (R7 acceptance: failures don't drop the record).

### Step 1.3 — Serializers

Add explicit serializers (no more generic `to_json` for the new shape):

```python
def keywords_to_json(result) -> str: ...
def entities_to_json(result) -> str: ...
```

**Acceptance (R1):** structured `keywords` array returned; the word `tags` does not appear in
the new output shape; score ordering identical to legacy (asserted by the Phase 0 invariant).

---

## Phase 2 — R3: `entities` from `doc.ents` (mechanical)

**Goal:** a dedicated entity path, separate from keyphrases, straight off the already-parsed `Doc`.

### Step 2.1 — Extend `ExtractorConfig` (config already exists — just add fields)

The label weights already live in `ExtractorConfig.entity_label_weights` (moved there during cleanup).
Add the R3 knobs with the spec's recommended defaults:

```python
    # --- R3: entities ---
    include_labels: list = field(default_factory=lambda: [
        "PERSON", "ORG", "NORP", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "FAC", "LOC",
    ])
    exclude_labels: list = field(default_factory=lambda: [
        "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT",
    ])
    remove_leading_the: bool = True
    max_mentions_per_entity: int = 3
    max_entities: int = 50
```

The scoring `label_weight` reuses the existing `entity_label_weights` (falling back to `1.0` for an
included label without an explicit weight, e.g. `FAC`, `LOC`). This keeps one source of truth for
label importance and matches "the label config already exists as a global — just moves into config."

### Step 2.2 — New method `_extract_entities(doc)`

Pure iteration over the already-parsed `doc.ents` (no `self.nlp(...)` reparse — this is why R3
naturally satisfies the entity half of R4):

1. Skip `ent.label_` in `exclude_labels`, or not in `include_labels`.
2. `normalized = _normalize(ent.text)` — trim, collapse internal whitespace, optional
   `remove_leading_the`, lowercase. (Reuse/extend the existing `normalize_entity`.)
3. Group by `(normalized, label_)`.
4. Per group accumulate `count` and up to `max_mentions_per_entity` `EntityMention`s using the
   **free** offsets `ent.start_char` / `ent.end_char`, `ent.text`, and `ent.sent.text` (bounded to
   `max_sentence_chars`, e.g. 300).
5. `label_weight = entity_label_weights.get(label, 1.0)`; `score = label_weight * count`
   (simple + explainable); `is_multi_word = " " in normalized`.
6. Sort by `score` desc, cap at `max_entities`.

**Do not** canonicalize across surface forms (`Target` vs `Target Corp` vs `Target Corporation`) — the
spec explicitly reserves that for the later matching layer.

**Acceptance (R3):** `entities` returned separately from `keywords`; each retains the
spaCy `label`; each has `count`, `score`, `label_weight`, and `mentions` with offsets; filtering and
weights are config-driven.

---

## Phase 3 — R7: ES/OpenSearch JSON columns (pure serialization)

**Goal:** stable, indexable columns built from the `ArticleEnrichmentResult`.

### Step 3.1 — Version constant + column builder

```python
ENRICHMENT_VERSION = "poc-1"

def to_enrichment_columns(result: ArticleEnrichmentResult) -> dict:
    return {
        "keywords_json":        json.dumps([asdict(c) for c in result.keywords]),
        "entities_json":      json.dumps([asdict(e) for e in result.entities]),
        "keyword_texts_json":   json.dumps([c.normalized for c in result.keywords]),
        "entity_texts_json": json.dumps([e.normalized for e in result.entities]),
        "entity_labels_json":json.dumps(sorted({e.label for e in result.entities})),
        "entity_pairs_json": json.dumps([f"{e.label}:{e.normalized}" for e in result.entities]),
        "enrichment_version":          result.enrichment_version,
        "enrichment_processed_at":     result.enrichment_processed_at,   # ISO-8601 UTC string
        "enrichment_error":            result.enrichment_error,          # None on success
    }
```

- JSON **strings** (not Arrow nested types) per the spec's POC guidance.
- `enrichment_error` is populated by the `extract_features` try/except (Step 1.2), never raised.

**Acceptance (R7):** flat keyword arrays + nested JSON available; schema is versioned; parseable into
ES/OpenSearch docs without reverse engineering.

---

## Phase 4 — Sample validation demo (spec Step 6, optional)

Small showcase so we can eyeball output before any indexing. Two low-risk options — pick one:

- **(preferred)** a tiny `tests/sample_features_demo.py` that runs `extract_features` on the fixtures
  and prints `article_id`, top `keywords`, top `entities`, and a couple of mentions.
- **or** extend `batch_extract.py` to *also* write the new R7 columns next to the legacy
  `tags_json`/`top_tags` (still CSV; the Parquet runner is R5/deferred).

This produces a reviewable artifact and doubles as manual QA for the golden files.

---

## Proposed decisions (confirm)

- **(1) Compatibility:** keep legacy `extract()` / `to_json()` **unchanged**; add `extract_features()`
  as the new surface. Migrate callers in a later phase. *(Recommended — zero risk to Lambda/batch.)*
- **(2) Class name:** keep `TagsExtractor` for now (rename to `ArticleTermExtractor` /
  `KeywordExtractor` deferred to the R4/R5 restructure, to avoid churn while callers still
  import the old name).
- **(3) Entity `score`:** `label_weight * count` — simple and explainable. (Not the keyphrase
  TextRank score; entities are a separate signal.)
- **(4) Entity offsets now:** yes — include `mentions` with `start`/`end`/`sentence` since they're
  free from `doc.ents`. Keyphrase offsets remain deferred (R2).
- **(5) `keywords.count`:** placeholder `1` this phase (the legacy score map doesn't carry a
  reliable per-term count); real counts arrive with the R2/R4 rework.
- **(6) `source_fields`:** hard-coded `["content"]` this phase; becomes real once R6 multi-field
  parsing lands.

---

## Input & scale — why R5 stays deferred (decision)

The end goal is running over **large Parquet files**, but that is **explicitly not** this phase:

- This phase's only public surface is `extract_features(text, ...) -> ArticleEnrichmentResult`
  (single article in, single result out). Hardcode `source_fields=["content"]`; no file I/O.
- **Do not hand-roll parallelism / `multiprocessing`** over a Python loop (the `batch_extract.py`
  pattern). At scale the idiomatic path is spaCy's **`nlp.pipe(texts, batch_size=, n_process=)`**
  plus **chunked Parquet reads** (pyarrow row groups) — batching *and* parallelism become config
  knobs, which is exactly what R6's `spacy.batch_size` / `spacy.n_process` already anticipate.
- Requirement: keep the extractor core **I/O-agnostic and `Doc`-ready** now (it already is —
  `get_tags(doc, …)` and `_extract_entities(doc)` take a `Doc`), so the future batch layer
  slots in *on top* with no core rewrite.
- Natural sequence after this bucket: **R4** (remove the per-entity `self.nlp(txt)` reparse so each
  field parses once) **→ R5** (the `enrich_parquet.py` runner using `nlp.pipe`).

## Testing

- Existing legacy golden test stays green (proves R1 is a pure wrap — the Phase 0 invariant).
- New goldens: `*.keywords.json`, `*.entities.json` (the R7 columns are validated structurally, not goldened).
- Float scores compared with `pytest.approx(rel=1e-9, abs=1e-9)`; keys/labels compared for exact order.
- No `self.nlp(...)` calls added in the entity path (grep-assert in review).

## Deferred to later buckets

- **R2** — keyphrase offset/evidence via case-insensitive fallback matching + `approximate: true`.
- **R4** — remove the per-entity `self.nlp(text)` reparse in the *keyphrase* path (`_build_entity_weights`)
  and thread the pre-parsed `Doc` through; add `nlp.pipe()`.
- **R5** — `enrich_parquet.py` batch runner: chunking, `--limit`, per-row error capture, progress/metrics,
  row-count preservation.
- **R6** — `extractor_config.yaml` loading + multi-field parse/weights and cross-field score/count
  aggregation (`final_score = base_score * field_weight * label_weight`).
