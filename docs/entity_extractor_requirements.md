# Requirements: Phase 1 POC - spaCy Extractor Improvements

## 1. Context

This document defines implementation requirements for improving the current `text_keyword_extractor.py` script for the Phase 1 POC of the client-agnostic enrichment job.

The current script combines:

- spaCy parsing and NER
- TextRank-style keyword scoring
- heuristic weighting for multi-word named entities
- final output as a JSON map of `tag -> score`

For the POC, the goal is not to build the final production enrichment service. The goal is to make the current extractor useful as a batch enrichment component that can process article samples, produce structured candidate terms/entities, and generate columns that can be written back to Parquet and indexed into Elasticsearch/OpenSearch.

## 2. Scope



### In scope

- Refactor the extractor output contract.
- Add structured evidence and offsets where available.
- Split keyphrase extraction from entity candidate extraction.
- Avoid repeated spaCy parsing of the same text.
- Make the script usable for batch processing over Parquet files.
- Add configurable input fields and extraction behavior.
- Produce JSON-compatible columns suitable for S3 Parquet and Elasticsearch/OpenSearch indexing.



### Out of scope for this POC

- Replacing spaCy with a different NER model.
- Full entity resolution against a canonical knowledge base.
- Client-specific entity matching rules.
- LLM verification.
- Embedding generation.
- Production orchestration, S3 event listeners, or Elasticsearch ingestion pipeline design.

Those belong to later phases of the enrichment/search architecture.

## 3. Current Script Observations

The current script has two main classes:

- `TextRank4Keyword` - parses text with spaCy, builds token co-occurrence windows, runs a PageRank-like scoring loop, and exposes `doc` and `node_weight`.
- `TagsExtractor` - uses spaCy NER entities and TextRank token weights to produce weighted tags.

Important current behaviors:

- The final public output is named `tags` and serialized with `to_json()` as a simple JSON object: `{term: score}`.
- `TagsExtractor.extract()` creates a `TextRank4Keyword` instance and parses tihe full input text once through `tr4w.analyze()`.
- `TagsExtractor.get_tags()` then reparses each entity text using `self.nlp(txt)` to filter tokens and build multi-word entity mappings.
- Entity labels are used as weights, especially for `PERSON`, `ORG`, `PRODUCT`, `EVENT`, and similar labels.
- The output does not include article ID, source field, offsets, entity label, sentence context, mention count, or evidence.

This is acceptable for a local keyword experiment, but too lossy for an enrichment/indexing flow.

## 4. Target POC Output Contract

The extractor should produce two separate outputs per article:

```text
keywords
entities
```

Optionally, it may also produce flattened helper fields for easier Elasticsearch querying.

### 4.1 `keywords`

Purpose: non-canonical keyphrases and candidate search terms discovered from article text.

Expected JSON shape:

```json
[
  {
    "text": "Target Circle",
    "normalized": "target circle",
    "type": "keyphrase",
    "score": 12.42,
    "count": 3,
    "source_fields": ["title", "content"],
    "evidence": [
      {
        "field": "title",
        "start": 18,
        "end": 31,
        "text": "Target Circle"
      }
    ]
  }
]
```

Notes:

- `text` should preserve the readable surface form.
- `normalized` should be lowercased and normalized for indexing/filtering.
- `score` should preserve the current ranking behavior where possible.
- `count` should count detected occurrences across configured fields.
- `source_fields` should list where the term was found.
- `evidence` should contain a small number of examples, not every occurrence by default.



### 4.2 `entities`

Purpose: named entities detected by spaCy, kept separately from keyword/keyphrase output.

Expected JSON shape:

```json
[
  {
    "text": "Target Corporation",
    "normalized": "target corporation",
    "label": "ORG",
    "score": 20.0,
    "count": 2,
    "source_fields": ["title", "content"],
    "mentions": [
      {
        "field": "content",
        "start": 240,
        "end": 258,
        "text": "Target Corporation",
        "sentence": "Target Corporation announced changes to its loyalty program."
      }
    ]
  }
]
```

Notes:

- This is not final entity resolution.
- `entities` are mentions grouped by normalized text and label.
- Later stages can use client/entity profiles, aliases, exclusions, search rules, vector search, or LLM verification to decide whether an entity candidate is a true match.



### 4.3 Optional flattened fields

For Elasticsearch/OpenSearch convenience, also create simple array fields:

```text
keyword_texts: ["target circle", "loyalty program", "retailer"]
entity_texts: ["target corporation", "brian cornell"]
entity_labels: ["ORG", "PERSON"]
entity_pairs: ["ORG:target corporation", "PERSON:brian cornell"]
```

These fields make it easier to run keyword filters, aggregations, and simple entity presence checks without parsing nested JSON.

## 5. Requirement 1 - Rename output from `tags` to `keywords`



### Current issue

The word `tags` is too vague. In the search-layer architecture, `tags` can mean client topics, taxonomy tags, article labels, entity labels, or analyst-created tags. The current output is closer to extracted keyphrases/candidate terms.

### Required changes

- Rename public output concept from `tags` to `keywords`.
- Keep backward compatibility only if needed by adding a temporary alias, not by keeping `tags` as the primary concept.
- Rename the class only if it helps readability. Suggested names:
  - `KeywordExtractor`
  - `KeyphraseExtractor`
  - `ArticleTermExtractor`



### Implementation steps

1. Introduce a new method that returns a list of structured candidate term objects instead of an `OrderedDict`.
2. Keep the current scoring algorithm initially, but wrap each term and score into the new output schema.
3. Replace `to_json()` with explicit serializers:
  - `keywords_to_json()`
  - or one generic `extract_article_features()` returning a dict.
4. Keep old `extract()` behavior behind a compatibility method only if existing POC code depends on it.



### Acceptance criteria

- The extractor returns `keywords` as an array of JSON objects.
- The output name `tags` is not used in the new enrichment output.
- Existing score ordering is preserved as much as possible.



## 6. Requirement 2 - Add offsets and evidence where possible



### Current issue

The current output only returns the final term and score. It does not explain where the term came from. For debugging search quality, analysts need evidence.

Example questions the output should answer:

- Did the term appear in the title or only in the body?
- Was the detected entity a real article mention or a scoring artifact?
- What exact sentence supported the entity candidate?
- Why did a specific article become a candidate for an entity profile?



### Required changes

Add evidence objects for both `keywords` and `entities`.

### Evidence fields

Minimum evidence fields:

```text
field       - source field name, for example title/content/summary
start       - character offset within that field
end         - character offset within that field
text        - exact matched span text
```

Recommended additional evidence fields:

```text
sentence    - sentence text containing the span
sentence_start
sentence_end
```



### Implementation notes

- For spaCy entities, offsets are available directly from `ent.start_char` and `ent.end_char` relative to the parsed field text.
- For noun chunks or generated keyphrases, offsets may not always be naturally available. In that case:
  - Use token span offsets when the phrase comes from a span.
  - Fall back to a case-insensitive text search in the original field for top terms.
  - Mark evidence as `approximate: true` if offsets were found by fallback matching.
- Limit evidence size to avoid huge JSON columns.

Recommended defaults:

```text
max_evidence_per_term: 3
max_sentence_chars: 300
```



### Acceptance criteria

- Entity candidates include exact offsets for mentions detected by spaCy.
- Candidate terms include evidence when traceable.
- Evidence is bounded and safe to store in Parquet/Elasticsearch.
- Missing evidence does not fail extraction; it should produce an empty evidence array.



## 7. Requirement 3 - Add entity candidate output separately from keyphrases



### Current issue

The current script blends entity mentions and keyphrase scoring into one final weighted output. This loses important distinctions.

For search/entity matching, these are different signals:

```text
Named entity candidate:
  "Target Corporation" labeled ORG by spaCy.

Candidate term/keyphrase:
  "loyalty program" scored by TextRank.
```

Both can help retrieval, but they should not be stored as the same thing.

### Required changes

Create a dedicated entity extraction path using the already parsed spaCy `Doc`.

### Entity candidate fields

Required:

```text
text
normalized
label
count
source_fields
mentions
```

Recommended:

```text
score
label_weight
first_seen_field
is_multi_word
```



### Label handling

The current script already uses label weights:

```text
PERSON: 5
ORG: 5
NORP: 3
GPE: 1.5
EVENT: 5
PRODUCT: 5
WORK_OF_ART: 5
DATE: 0
```

For the POC, keep this idea but make it configurable.

Recommended default labels to keep:

```text
PERSON
ORG
NORP
GPE
EVENT
PRODUCT
WORK_OF_ART
FAC
LOC
```

Recommended labels to exclude or downweight by default:

```text
DATE
TIME
CARDINAL
ORDINAL
QUANTITY
MONEY
PERCENT
```



### Normalization rules

Implement shared normalization for all entity candidates:

- trim whitespace
- collapse internal whitespace
- remove leading `the`  only as a configurable option
- lowercase normalized form
- optionally strip punctuation at boundaries

Do not over-normalize in the POC. For example, do not try to merge `Target`, `Target Corp`, and `Target Corporation` into one canonical entity yet. That belongs to client/entity profile matching.

### Acceptance criteria

- The extractor outputs `entities` separately from `keywords`.
- Entity candidates retain spaCy label information.
- Each entity candidate has mention evidence with offsets.
- Entity label filtering/weighting is configurable.



## 8. Requirement 4 - Avoid reparsing text repeatedly



### Current issue

The current script parses the full article text once in `TextRank4Keyword.analyze()`, but then reparses every entity text in `TagsExtractor.get_tags()` using `self.nlp(txt)`. This is inefficient and will hurt batch performance on large files.

### Required changes

Use the existing spaCy `Doc` and entity/token spans instead of calling `nlp()` repeatedly for entity strings.

### Implementation steps

1. Parse each configured article field exactly once.
2. Pass parsed `Doc` objects into all extraction functions.
3. For entity token filtering, use tokens already inside the `Span`:
  - `for token in ent`
  - avoid `curDoc = self.nlp(txt)`
4. Move TextRank graph building to accept a pre-parsed `Doc`.
5. Use `nlp.pipe()` for batch processing multiple records.



### Proposed internal structure

```text
ArticleFeatureExtractor
  extract_record(record)                         # high-level method
  parse_fields(record, config)                   # one parse per configured field
  extract_keywords(parsed_fields)
  extract_entities(parsed_fields)
  build_output(record, terms, entities)
```

Where `parsed_fields` is something like:

```json
{
  "title": {"text": "...", "doc": "spaCy Doc"},
  "content": {"text": "...", "doc": "spaCy Doc"}
}
```



### Acceptance criteria

- No `nlp()` calls inside per-entity loops.
- The number of spaCy parses per article equals the number of configured text fields, not the number of entities.
- Batch processing uses `nlp.pipe()` where possible.



## 9. Requirement 5 - Make it batch-friendly for large files



### Current issue

The current class works for a single text string. The enrichment job needs to process Parquet files, potentially with many articles and large text fields.

### Required changes

Add a batch-oriented wrapper around the extractor.

### POC batch input

Input should support Parquet files with configurable columns, for example:

```text
article_id
source_name
title
content
published_at
harvest_date
language
source_country_code
```

Do not hardcode these names except for a default POC config.

### POC batch output

The batch job should write a new enriched Parquet file preserving original columns and adding enrichment columns:

```text
keywords                 JSON string or list-compatible Arrow type
entities               JSON string or list-compatible Arrow type
keyword_texts            array<string> or JSON string
entity_texts          array<string> or JSON string
entity_labels         array<string> or JSON string
entity_pairs          array<string> or JSON string
enrichment_version              string
enrichment_model                string
enrichment_processed_at         timestamp
enrichment_error                string/null
```

For POC simplicity, JSON strings are acceptable. For longer-term Parquet usability, Arrow list/struct columns are better.

### Batch processing requirements

- Process rows in chunks to avoid memory pressure.
- Support configurable chunk size.
- Support `limit` for local testing.
- Support skipping records with empty text.
- Capture per-record extraction errors without failing the whole file.
- Log progress every N records.
- Preserve source row count.

Recommended CLI shape:

```bash
python enrich_parquet.py \
  --input-path sample.parquet \
  --output-path sample_enriched.parquet \
  --config extractor_config.yaml \
  --limit 1000
```



### Performance requirements for POC

The POC should report:

```text
records_processed
records_failed
elapsed_seconds
records_per_second
average_text_chars
output_path
```

No strict performance SLA is required yet, but the implementation should avoid obvious per-row and per-entity inefficiencies.

### Acceptance criteria

- The extractor can process a Parquet sample without loading unnecessary columns.
- Output row count matches input row count.
- Failed rows are marked with `enrichment_error` instead of silently dropped.
- Progress and summary metrics are logged.



## 10. Requirement 6 - Add config for which fields to process



### Current issue

The current extractor accepts one text string. Article data usually has multiple fields, and those fields should have different importance.

For example:

```text
title should have higher weight than content
summary may be useful if available
publisher/source fields should usually not be processed as article text
```



### Required changes

Add an extractor config file.

### Suggested `extractor_config.yaml`

```yaml
version: "poc-1"

input:
  id_field: "article_id"
  fields:
    - name: "title"
      enabled: true
      weight: 3.0
      max_chars: 1000
    - name: "content"
      enabled: true
      weight: 1.0
      max_chars: 20000
    - name: "summary"
      enabled: false
      weight: 1.5
      max_chars: 3000

spacy:
  model: "en_core_web_sm"
  batch_size: 100
  n_process: 1
  disable: []

keywords:
  enabled: true
  max_terms: 30
  candidate_pos: ["NOUN", "PROPN"]
  window_size: 4
  lower: false
  stopwords: []
  max_evidence_per_term: 3

entities:
  enabled: true
  max_entities: 50
  include_labels: ["PERSON", "ORG", "NORP", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "FAC", "LOC"]
  exclude_labels: ["DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT"]
  label_weights:
    PERSON: 5.0
    ORG: 5.0
    NORP: 3.0
    GPE: 1.5
    EVENT: 5.0
    PRODUCT: 5.0
    WORK_OF_ART: 5.0
  remove_leading_the: true
  max_mentions_per_entity: 3

output:
  json_as_string: true
  include_flattened_fields: true
```



### Field weighting

Field weights should influence scoring. For POC:

```text
final_score = base_score * field_weight * label_weight_or_1
```

If a term/entity appears across multiple fields, aggregate its score and count.

### Acceptance criteria

- Input fields are not hardcoded.
- Title/content/summary behavior can be changed without code edits.
- Entity labels and weights can be changed without code edits.
- The config is committed next to the POC script and loaded at runtime.



## 11. Requirement 7 - Output JSON columns usable by Elasticsearch/OpenSearch indexing



### Current issue

A simple JSON object of `{term: score}` is not ideal for search indexing, filtering, or debugging.

Elasticsearch/OpenSearch can index arrays and nested objects, but the output should be explicit and stable.

### Required changes

Create stable enrichment columns that can be indexed directly.

### Recommended enriched Parquet columns

```text
keywords_json             - JSON array of structured candidate terms
entities_json           - JSON array of structured entity candidates
keyword_texts_json        - JSON array of normalized term strings
entity_texts_json      - JSON array of normalized entity strings
entity_labels_json     - JSON array of labels
entity_pairs_json      - JSON array of "LABEL:normalized" strings
enrichment_version               - extractor version/config version
enrichment_processed_at          - timestamp
enrichment_error                 - error text or null
```

For POC, use JSON string columns to avoid Arrow nested-type complexity. Later, consider true list/struct columns in Parquet.

### Suggested Elasticsearch/OpenSearch mapping direction

For POC indexing:

```json
{
  "keyword_texts": { "type": "keyword" },
  "entity_texts": { "type": "keyword" },
  "entity_labels": { "type": "keyword" },
  "entity_pairs": { "type": "keyword" },
  "keywords": { "type": "nested" },
  "entities": { "type": "nested" }
}
```

Notes:

- Use flattened keyword arrays for simple filters and aggregations.
- Use nested objects only when evidence-level queries are required.
- Keep the original article text fields indexed separately as text fields.



### Acceptance criteria

- Enriched output can be written to Parquet.
- Enriched JSON can be parsed into Elasticsearch/OpenSearch documents without custom reverse engineering.
- Flat keyword arrays are available for simple entity/term filters.
- The output schema is versioned.



## 12. Proposed Implementation Plan



### Step 1 - Add output models

Add lightweight typed structures using dataclasses or Pydantic models:

```text
Evidence
Keyword
EntityMention
Entity
ArticleEnrichmentResult
```

Pydantic is useful if validation and schema generation are desired. Dataclasses are enough for the POC.

### Step 2 - Separate parsing from extraction

Refactor the code so the high-level flow is:

```text
parse configured fields
extract entities from parsed docs
extract candidate terms from parsed docs
aggregate across fields
serialize result
```



### Step 3 - Refactor TextRank to accept `Doc`

Current `TextRank4Keyword.analyze(text, ...)` should be complemented or replaced by:

```text
analyze_doc(doc, field_name, field_weight, ...)
```

This avoids reprocessing the same text and makes field-level evidence possible.

### Step 4 - Add entity extraction from `doc.ents`

Create a dedicated method that groups entity mentions by normalized text and label.

### Step 5 - Add batch Parquet wrapper

Implement a separate runner script rather than overloading the extractor class.

Suggested files:

```text
text_keyword_extractor.py          # core extraction logic
extractor_config.yaml              # POC config
enrich_parquet.py                  # batch runner
requirements.md                    # this document
```



### Step 6 - Add sample output validation

Create a small test or script that runs on 10-100 articles and prints:

```text
article_id
top keywords
top entities
evidence examples
```

This is important before indexing into Elasticsearch/OpenSearch.

## 13. POC Success Criteria

The Phase 1 POC is successful when:

1. A sample Parquet file can be enriched with the improved extractor.
2. The enriched Parquet output contains `keywords` and `entities` separately.
3. Entity candidates include label, count, score, and mention evidence with offsets.
4. Candidate terms include score, count, source fields, and bounded evidence where available.
5. The job can process records in batches without reparsing entity text repeatedly.
6. The input fields and entity labels are configurable.
7. The output can be indexed into Elasticsearch/OpenSearch with simple keyword arrays and optional nested JSON objects.



## 14. Non-Goals and Known Limitations

The improved spaCy extractor is still only a first-stage enrichment tool.

Known limitations:

- spaCy NER will miss some organizations, products, and brand aliases.
- spaCy may mislabel common words used as brands, such as `Target`.
- Candidate terms are not canonical entities.
- The extractor does not decide whether an article truly matches a client entity profile.
- Alias logic, exclusions, semantic search, and LLM confirmation are later matching-layer responsibilities.

This is expected. The extractor should produce useful candidate signals, not final entity truth.

## 15. Suggested Next Document

After this extractor improvement spec, the next design document should define the client-agnostic enrichment job flow:

```text
S3 daily raw Parquet upload
  -> enrichment trigger
  -> batch enrichment job
  -> enriched Parquet write
  -> Elasticsearch/OpenSearch indexing
  -> daily client/entity matching job
```

