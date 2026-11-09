from dataclasses import dataclass, asdict
from typing import Optional

from .bio import bio_spans, split_label

@dataclass
class NERSentence:
    example_id: str
    language: str
    dataset: str
    split: str
    tokens: list[str]
    labels_original: list[str]
    labels_coarse: list[str]
    sentence_text: str
    source_document_id: Optional[str] = None

    def validate(self) -> None:
        for name in ("example_id", "dataset", "sentence_text"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Missing {name} in {self.example_id!r}")
        if self.language not in {"tag", "ceb", "hil"}:
            raise ValueError(f"Unknown language: {self.language}")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError(f"Unknown split: {self.split}")
        if not isinstance(self.tokens, list) or not self.tokens:
            raise ValueError(f"Empty/non-list tokens in {self.example_id}")
        if not all(isinstance(t, str) and t.strip() for t in self.tokens):
            raise ValueError(f"Invalid token in {self.example_id}")
        if not (len(self.tokens) == len(self.labels_original) == len(self.labels_coarse)):
            raise ValueError(f"Length mismatch in {self.example_id}")
        if self.source_document_id is not None and (
            not isinstance(self.source_document_id, str) or not self.source_document_id.strip()
        ):
            raise ValueError("source_document_id must be a non-empty string or null")
        bio_spans(self.labels_original)
        bio_spans(self.labels_coarse)
        for label in self.labels_coarse:
            _, kind = split_label(label)
            if kind is not None and kind not in {"PER", "LOC", "ORG"}:
                raise ValueError(f"Non-coarse label {label} in {self.example_id}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict) -> "NERSentence":
        result = cls(**row)
        result.validate()
        return result
