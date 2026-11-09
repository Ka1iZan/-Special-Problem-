"""Adapter for the inspected Hugging Face default Parquet release."""
import json
from pathlib import Path
from .io import validate_unique
from .schema import NERSentence


def load_tlunified_parquet(path: Path, split: str) -> list[NERSentence]:
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    metadata = json.loads(table.schema.metadata[b"huggingface"])
    feature = metadata["info"]["features"]["ner_tags"]
    names = feature.get("feature", feature.get("element"))["names"]
    if set(names) != {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}:
        raise ValueError("Release ontology changed; audit before conversion")
    records = []
    for row in table.to_pylist():
        labels = [names[index] for index in row["ner_tags"]]
        record = NERSentence(f"tlunified_ner:{split}:{row['id']}", "tag", "tlunified_ner", split,
                             row["tokens"], labels.copy(), labels.copy(), " ".join(row["tokens"]), None)
        record.validate()
        records.append(record)
    validate_unique(records)
    return records
