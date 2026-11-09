"""Explicit generic adapters; corpus-specific release formats remain to be audited."""
from __future__ import annotations

import json
from pathlib import Path

from .harmonize import project_sequence
from .schema import NERSentence


def read_jsonl(path: Path) -> list[NERSentence]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                records.append(NERSentence.from_dict(json.loads(line)))
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    validate_unique(records)
    return records


def validate_unique(records: list[NERSentence]) -> None:
    ids = set()
    for record in records:
        record.validate()
        if record.example_id in ids:
            raise ValueError(f"Duplicate example_id: {record.example_id}")
        ids.add(record.example_id)
    if not records:
        raise ValueError("No records")


def write_jsonl(records: list[NERSentence], path: Path) -> None:
    validate_unique(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def read_conll(path: Path, *, dataset: str, language: str, split: str,
               mapping: dict[str, str], token_column: int = 0,
               label_column: int = -1, expected_columns: int = 2) -> list[NERSentence]:
    """Whitespace columns; blank sentence boundaries; optional # newdoc id / sent_id.

    Never synthesize a document ID from a sentence ID. Nested, flat, or span-offset
    formats need a release-specific adapter and are not guessed here.
    """
    records, tokens, labels = [], [], []
    document_id, sentence_id = None, None

    def flush():
        nonlocal tokens, labels, sentence_id
        if not tokens:
            return
        record = NERSentence(
            example_id=f"{dataset}:{split}:{sentence_id or str(len(records) + 1)}",
            language=language, dataset=dataset, split=split, tokens=tokens,
            labels_original=labels, labels_coarse=project_sequence(labels, mapping),
            sentence_text=" ".join(tokens), source_document_id=document_id,
        )
        record.validate()
        records.append(record)
        tokens, labels, sentence_id = [], [], None

    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("# newdoc id ="):
            flush()
            document_id = line.split("=", 1)[1].strip()
        elif line.startswith("# sent_id ="):
            flush()
            sentence_id = line.split("=", 1)[1].strip()
        elif line.startswith("#"):
            continue
        elif line.startswith("-DOCSTART-"):
            raise ValueError("DOCSTART needs a release-specific document-ID rule; do not guess IDs")
        else:
            fields = line.split()
            if len(fields) != expected_columns:
                raise ValueError(f"Expected {expected_columns} columns at {path}:{number}, found {len(fields)}; review annotation layers")
            try:
                tokens.append(fields[token_column])
                labels.append(fields[label_column])
            except IndexError as exc:
                raise ValueError(f"Missing columns at {path}:{number}") from exc
    flush()
    validate_unique(records)
    return records
