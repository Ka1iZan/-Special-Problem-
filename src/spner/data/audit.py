from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from statistics import median

from .bio import bio_spans
from .io import validate_unique
from .overlap import normalize_surface, training_entity_vocabulary
from .schema import NERSentence


def content_hash(tokens: list[str]) -> str:
    return hashlib.sha256(normalize_surface(tokens).encode("utf-8")).hexdigest()


def audit_records(records: list[NERSentence]) -> dict:
    validate_unique(records)
    groups, duplicate_groups, document_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in records:
        groups[(row.language, row.dataset, row.split)].append(row)
        duplicate_groups[content_hash(row.tokens)].append(row)
        if row.source_document_id:
            document_groups[(row.dataset, row.source_document_id)].append(row)
    summaries = []
    for (language, dataset, split), rows in sorted(groups.items()):
        lengths = [len(row.tokens) for row in rows]
        entity_counts = Counter(span.label for row in rows for span in bio_spans(row.labels_coarse))
        original_counts = Counter(span.label for row in rows for span in bio_spans(row.labels_original))
        summaries.append({"language": language, "dataset": dataset, "split": split,
                          "sentences": len(rows), "tokens": sum(lengths),
                          "entities": dict(entity_counts), "original_entities": dict(original_counts),
                          "ignored_tokens": sum(row.labels_coarse.count("IGNORE") for row in rows),
                          "missing_document_ids": sum(row.source_document_id is None for row in rows),
                          "word_length_min": min(lengths), "word_length_median": median(lengths),
                          "word_length_max": max(lengths)})
    duplicates = [{"hash": digest, "example_ids": [row.example_id for row in rows],
                   "cross_split": len({row.split for row in rows}) > 1,
                   "datasets": sorted({row.dataset for row in rows})}
                  for digest, rows in sorted(duplicate_groups.items()) if len(rows) > 1]
    document_leaks = [{"dataset": key[0], "document_id": key[1],
                       "splits": sorted({row.split for row in rows})}
                      for key, rows in sorted(document_groups.items())
                      if len({row.split for row in rows}) > 1]
    overlap = []
    languages = sorted({row.language for row in records})
    for source in languages:
        training = [row for row in records if row.language == source and row.split == "train"]
        if not training:
            continue
        vocabulary = training_entity_vocabulary(training)
        for target in languages:
            if target == source:
                continue
            mentions = [(span.label, normalize_surface(row.tokens[span.start:span.end]))
                        for row in records if row.language == target and row.split == "test"
                        for span in bio_spans(row.labels_coarse)]
            if not mentions:
                continue
            seen = sum(surface in vocabulary for _, surface in mentions)
            overlap.append({"source": source, "target": target, "evaluation_split": "test",
                            "source_train_unique_forms": len(vocabulary),
                            "gold_mentions": len(mentions), "seen_mentions": seen,
                            "unseen_mentions": len(mentions) - seen,
                            "seen_fraction": seen / len(mentions)})
    canonical = json.dumps([r.to_dict() for r in sorted(records, key=lambda r: r.example_id)],
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"manifest_id": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "summaries": summaries, "duplicates": duplicates,
            "cross_split_duplicate_groups": sum(row["cross_split"] for row in duplicates),
            "document_leaks": document_leaks, "source_train_target_test_overlap": overlap,
            "normalization": "NFC + casefold + collapsed whitespace; punctuation retained",
            "limitations": ["Word lengths are not XLM-R subword lengths.",
                            "Exact normalized duplicates only; near-duplicates and translations need review.",
                            "Missing document IDs limit both leakage checks and cluster resampling.",
                            "Overlap is descriptive; never select a model using test overlap results."]}
