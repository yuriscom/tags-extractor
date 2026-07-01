import datetime
import json
import re
from collections import OrderedDict, Counter
from dataclasses import asdict, dataclass, field

import numpy as np
from spacy.lang.en.stop_words import STOP_WORDS

# Stamped onto enrichment output so downstream consumers can tell which extractor
# version produced a row. Bump when the output schema or scoring changes.
ENRICHMENT_VERSION = "poc-1"

_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_THE_RE = re.compile(r"^the\s+", flags=re.IGNORECASE)


@dataclass
class ExtractorConfig:
    """Tunable parameters for TagsExtractor, grouped by which output they affect.

    The two outputs are independent and so are most of their settings:
      - keywords: the TextRank keyphrase path.
      - entities: the spaCy NER path.
    The one shared setting is entity_label_weights (see below).
    """

    # --- keywords (keyphrase path) only ---
    exclude_pos: list = field(default_factory=lambda: ["PUNCT", "PART"])
    multiple_occurrence_multiplier: bool = True
    full_term_weight_bonus: dict = field(default_factory=lambda: {"PERSON": 2})

    # --- shared by both paths ---
    # Per-label importance. In the keyphrase path it weights multi-word entities during
    # merging; in the entity path it is the entity's label_weight (score = weight * count).
    entity_label_weights: dict = field(default_factory=lambda: {
        "PERSON": 5, "ORG": 5, "NORP": 3, "GPE": 1.5, "EVENT": 5, "PRODUCT": 5,
        "WORK_OF_ART": 5, "DATE": 0,
    })

    # --- entities path only ---
    # Which entities to keep: label must be in include_labels (empty = keep any) and
    # not in exclude_labels.
    include_labels: list = field(default_factory=lambda: [
        "PERSON", "ORG", "NORP", "GPE", "EVENT", "PRODUCT", "WORK_OF_ART", "FAC", "LOC",
    ])
    exclude_labels: list = field(default_factory=lambda: [
        "DATE", "TIME", "CARDINAL", "ORDINAL", "QUANTITY", "MONEY", "PERCENT",
    ])
    remove_leading_the: bool = True
    max_mentions_per_entity: int = 3
    max_sentence_chars: int = 300
    max_entities: int = 50


@dataclass
class Evidence:
    field: str
    start: int
    end: int
    text: str
    sentence: str | None = None
    approximate: bool = False


@dataclass
class Keyword:
    text: str
    normalized: str
    type: str = "keyphrase"
    score: float = 0.0
    # TODO: placeholder, always 1. The keyphrase scoring folds occurrence into the score
    # and keeps no separate count; populate with real occurrence counts when added.
    count: int = 1
    source_fields: list = field(default_factory=lambda: ["content"])
    evidence: list = field(default_factory=list)


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
    source_fields: list = field(default_factory=lambda: ["content"])
    mentions: list = field(default_factory=list)


@dataclass
class ArticleEnrichmentResult:
    keywords: list = field(default_factory=list)
    entities: list = field(default_factory=list)
    enrichment_version: str = ENRICHMENT_VERSION
    enrichment_processed_at: str = ""
    enrichment_error: str | None = None


class TextRank4Keyword():
    def __init__(self, nlp):
        self.nlp = nlp
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.doc = None

    def set_stopwords(self, stopwords):
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = self.nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, labels, lower):
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with candidate POS tag
                is_candidate_pos = token.pos_ in candidate_pos and token.is_stop is False and token.is_alpha is True
                is_candidate_label = len(labels) == 0 or (len(labels) > 0 and token.ent_type_ in labels)
                if is_candidate_pos and is_candidate_label:
                    if lower is True:
                        selected_words.append(token.lemma_.lower())
                    else:
                        selected_words.append(token.lemma_)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i + 1, i + window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get symmetric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, out=np.zeros_like(g), where=norm != 0)

        return g_norm

    def analyze(self, text,
                labels=[],
                candidate_pos=['NOUN', 'PROPN'],
                window_size=3, lower=False, stopwords=list()):

        # Set stop words
        self.set_stopwords(stopwords)

        # Parse text with spaCy
        doc = self.nlp(text)

        self.doc = doc

        # Filter sentences (here put the logic for what we include in the sentence and what not)
        sentences = self.sentence_segment(doc, candidate_pos, labels, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initialization for weight (pagerank value)
        pr = np.ones(len(vocab), dtype='float')

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1 - self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


class TagsExtractor():
    def __init__(self, nlp, config=None):
        self.nlp = nlp
        self.config = config or ExtractorConfig()
        self.remove_ambiguity = True
        self.tags = OrderedDict()
        # Parsed Doc from the last extract() call, reused by extract_features().
        self.doc = None
        # Per-instance copy so extract() can merge request labels without mutating the config.
        self.entity_label_weights = self.config.entity_label_weights.copy()

    # labels: https://spacy.io/api/annotation#dependency-parsing
    def extract(self, text, labels, num=20):
        if isinstance(labels, dict):
            self.entity_label_weights.update(labels)
            labels = list(labels.keys())

        tr4w = TextRank4Keyword(self.nlp)
        tr4w.analyze(text, labels, candidate_pos=['NOUN', 'PROPN'], window_size=4, lower=False)
        self.doc = tr4w.doc
        return self.get_tags(tr4w.doc, tr4w.node_weight, num)

    def extract_features(self, text, labels, num=20):
        """Return keywords and entities as an ``ArticleEnrichmentResult``.

        ``keywords`` holds the ranked keyphrases produced by ``extract()``; ``entities``
        holds spaCy entity mentions grouped by normalized text and label (read from the
        Doc that ``extract()`` just parsed). Any failure is recorded in
        ``enrichment_error`` instead of raised, so a single bad article never aborts a
        batch run.
        """
        result = ArticleEnrichmentResult(
            enrichment_version=ENRICHMENT_VERSION,
            enrichment_processed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )
        try:
            tags = self.extract(text, labels, num)
            result.keywords = [
                Keyword(text=term, normalized=self._normalize_text(term), score=float(score))
                for term, score in tags.items()
            ]
            result.entities = self._extract_entities(self.doc)
        except Exception as exc:
            result.enrichment_error = str(exc)
        return result

    def _normalize_text(self, text):
        """Lowercased, whitespace-collapsed form used for indexing/filtering."""
        return _WHITESPACE_RE.sub(" ", text).strip().lower()

    def _clean_surface(self, text):
        """Readable surface form: collapse whitespace, optionally drop a leading 'the'."""
        cleaned = _WHITESPACE_RE.sub(" ", text).strip()
        if self.config.remove_leading_the:
            cleaned = _LEADING_THE_RE.sub("", cleaned)
        return cleaned

    def _extract_entities(self, doc, field_name="content"):
        """Group the entities in an already-parsed ``Doc`` into Entity objects.

        Mentions are grouped by ``(normalized, label)`` and filtered/weighted by config.
        Works off the existing ``Doc`` (no reparsing) and reads mention offsets straight
        from each ``Span``.
        """
        include = set(self.config.include_labels)
        exclude = set(self.config.exclude_labels)
        groups = OrderedDict()

        for ent in doc.ents:
            label = ent.label_
            if label in exclude:
                continue
            if include and label not in include:
                continue
            surface = self._clean_surface(ent.text)
            if not surface:
                continue
            normalized = surface.lower()
            key = (normalized, label)

            group = groups.get(key)
            if group is None:
                group = {"text": surface, "normalized": normalized, "label": label,
                         "count": 0, "mentions": []}
                groups[key] = group
            group["count"] += 1

            if len(group["mentions"]) < self.config.max_mentions_per_entity:
                sentence = _WHITESPACE_RE.sub(" ", ent.sent.text).strip() if ent.sent is not None else ""
                sentence = sentence[: self.config.max_sentence_chars]
                group["mentions"].append(EntityMention(
                    field=field_name,
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                    sentence=sentence or None,
                ))

        entities = []
        for group in groups.values():
            label_weight = float(self.entity_label_weights.get(group["label"], 1.0))
            entities.append(Entity(
                text=group["text"],
                normalized=group["normalized"],
                label=group["label"],
                score=label_weight * group["count"],
                count=group["count"],
                label_weight=label_weight,
                is_multi_word=(" " in group["normalized"]),
                first_seen_field=field_name,
                source_fields=[field_name],
                mentions=group["mentions"],
            ))
        # Stable, deterministic ordering: score desc, then normalized/label for ties.
        entities.sort(key=lambda e: (-e.score, e.normalized, e.label))
        return entities[: self.config.max_entities]

    def normalize_entity(self, ent):
        return re.sub(r'^the\s+', "", ent, flags=re.IGNORECASE)

    def get_tags(self, doc, node_weight, number=20):
        entity_weights, full_term_map = self._build_entity_weights(doc)
        tags = self._merge_terms(node_weight, entity_weights, full_term_map)
        # `number` is an output cap: return at most the top-N tags (often fewer, since
        # many terms don't emit a tag). All terms are scored first so the ranking is stable.
        ranked = sorted(tags.items(), key=lambda t: t[1], reverse=True)[:number]
        self.tags = OrderedDict(ranked)
        return self.tags

    def _build_entity_weights(self, doc):
        """Walk the spaCy entities and accumulate per-entity weights.

        Returns:
            entity_weights: Counter of normalized entity text -> accumulated weight.
            full_term_map: dict of lemma -> set(FullTerm) for multi-word entities,
                used later to promote single tokens to their full phrase.
        """
        entity_weights = Counter()
        occurrence = Counter()
        full_term_map = dict()

        for entity in doc.ents:
            weight = 1
            text = self.normalize_entity(entity.text)
            entity_doc = self.nlp(text)
            words = []
            token_count = len(entity_doc)

            for token in entity_doc:
                if token.pos_ not in self.config.exclude_pos:
                    words.append(token.text)

            text = " ".join(words)
            occurrence[text] += 1

            if token_count > 1:
                weight = self.entity_label_weights[entity.label_] if entity.label_ in self.entity_label_weights else 1
                for token in entity_doc:
                    if token.is_stop is False:
                        bonus_multiplier = self.config.full_term_weight_bonus[entity.label_] \
                            if entity.label_ in self.config.full_term_weight_bonus else 1

                        full_term = FullTerm(text, bonus_multiplier)
                        full_term_map.setdefault(token.lemma_, set())
                        full_term_map[token.lemma_].add(full_term)

            if self.config.multiple_occurrence_multiplier:
                weight = weight * occurrence[text]

            entity_weights[text] += weight

        return entity_weights, full_term_map

    def _resolve_ambiguity(self, full_terms, entity_weights):
        """Pick the winning full phrase when a lemma maps to several of them.

        Returns (is_ambiguity, ambiguity_winner). ambiguity_winner is None when the
        phrases are tied (no clear winner) or when ambiguity resolution is disabled.
        """
        is_ambiguity = False
        ambiguity_winner = None
        if len(full_terms) > 1 and self.remove_ambiguity is True:
            is_ambiguity = True
            phrase_counts = Counter()

            for full_term in full_terms:
                text = full_term.text
                phrase_counts[text] += entity_weights[text] or 1
            ranked = phrase_counts.most_common()  # highest weight first
            if ranked[0][1] != ranked[-1][1]:  # weights aren't all equal -> clear winner
                ambiguity_winner = ranked[0][0]
        return is_ambiguity, ambiguity_winner

    def _merge_terms(self, node_weight, entity_weights, full_term_map):
        """Combine TextRank scores with entity weights, promoting single tokens to
        their multi-word entity phrase where that scores higher.

        A multi-word phrase is amplified: each of its tokens compounds the phrase's
        score. This is tracked with a per-phrase growth factor (starts at 1.0) that
        multiplies up as tokens contribute, so the phrase weight (from entity_weights)
        and the rank-based growth stay as separate factors.

        All terms are scored (no early cutoff) so every contribution lands and the final
        ranking is stable; the caller caps the number of tags returned.
        """
        tags = dict()
        phrase_growth = dict()
        node_weight = OrderedDict(sorted(node_weight.items(), key=lambda t: t[1], reverse=True))
        for term, rank_score in node_weight.items():
            # we are only interested in replacing the ones that have multi worded version
            if term in full_term_map:
                is_ambiguity, ambiguity_winner = self._resolve_ambiguity(full_term_map[term], entity_weights)

                single_weight = 0
                if term in entity_weights:
                    single_weight = entity_weights[term]

                if self.remove_ambiguity is True and is_ambiguity and ambiguity_winner is None:
                    # Ambiguous tie.
                    # Example: Morgan Stanley and Morgan Freeman get the same weight.
                    # The shared token "Morgan" maps to several equally-weighted entities,
                    # so no single phrase wins. Keep the bare token "Morgan" with its
                    # TextRank score only (no entity-weight multiplier), which is why this
                    # score is small (a single term "Morgan" gets a small score).
                    # The full phrases still get their proper score via
                    # their own unique tokens (e.g. "Stanley" -> "Morgan Stanley", "Freeman" -> "Morgan Freeman")
                    # so this just preserves the shared token instead of dropping it entirely.
                    tags[term] = tags.setdefault(term, 0) + rank_score
                    continue

                for full_term in full_term_map[term]:
                    text = full_term.text
                    weight_bonus = full_term.weight
                    if ambiguity_winner and ambiguity_winner != text:
                        continue

                    growth = phrase_growth.get(text, 1.0)
                    multi_weight = entity_weights[text] * growth * weight_bonus

                    if single_weight >= multi_weight:
                        tags[term] = tags.setdefault(term, 0) + (rank_score * single_weight)
                    else:
                        tags[text] = tags.setdefault(text, 0) + (rank_score * multi_weight)
                        phrase_growth[text] = growth * (1 + rank_score * weight_bonus)

            else:
                weight = entity_weights[term]
                if weight > 0:
                    tags[term] = rank_score * weight

        return tags

    def to_json(self):
        return json.dumps(self.tags)


def keywords_to_json(result):
    return json.dumps([asdict(keyword) for keyword in result.keywords], ensure_ascii=False)


def entities_to_json(result):
    return json.dumps([asdict(entity) for entity in result.entities], ensure_ascii=False)


def to_enrichment_dict(result):
    """Full result as one nested, JSON-serializable dict — handy for debugging.

    Keeps the objects nested (unlike to_enrichment_columns, which returns JSON strings),
    so ``json.dumps(to_enrichment_dict(result), indent=2, ensure_ascii=False)`` gives a
    single readable, paste-ready document.
    """
    return {
        "keywords": [asdict(keyword) for keyword in result.keywords],
        "entities": [asdict(entity) for entity in result.entities],
        "enrichment_version": result.enrichment_version,
        "enrichment_processed_at": result.enrichment_processed_at,
        "enrichment_error": result.enrichment_error,
    }


def to_enrichment_columns(result):
    """Flatten a result into JSON-string columns for storage and search indexing.

    Each value is a JSON string: the ``*_json`` fields carry the full nested objects,
    while the flat text/label/pair arrays support simple keyword filters and aggregations.
    """
    return {
        "keywords_json": keywords_to_json(result),
        "entities_json": entities_to_json(result),
        "keyword_texts_json": json.dumps(
            [keyword.normalized for keyword in result.keywords], ensure_ascii=False),
        "entity_texts_json": json.dumps(
            [entity.normalized for entity in result.entities], ensure_ascii=False),
        "entity_labels_json": json.dumps(
            sorted({entity.label for entity in result.entities}), ensure_ascii=False),
        "entity_pairs_json": json.dumps(
            [f"{entity.label}:{entity.normalized}" for entity in result.entities],
            ensure_ascii=False),
        "enrichment_version": result.enrichment_version,
        "enrichment_processed_at": result.enrichment_processed_at,
        "enrichment_error": result.enrichment_error,
    }


class FullTerm:
    def __init__(self, text, weight):
        self.text = text
        self.weight = weight

    def __eq__(self, other):
        return isinstance(other, FullTerm) and self.text == other.text

    def __hash__(self):
        return hash(self.text)

    def __str__(self):
        return f"{self.text}:{self.weight}"
