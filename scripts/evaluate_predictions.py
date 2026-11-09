import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.data.io import read_jsonl
from spner.data.overlap import training_entity_vocabulary
from spner.metrics import score_sentence, aggregate_scores
from spner.utils import object_sha256, file_sha256, save_json


def main():
    parser = argparse.ArgumentParser(description="Strict word-level BIO entity evaluation")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True,
                        help='JSONL: {"example_id": "...", "labels": ["O", "B-PER", ...]}')
    parser.add_argument("--source-train", type=Path, nargs="*")
    parser.add_argument("--ignore-policy", choices=["reject", "exclude_touching_spans"], default="reject")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = read_jsonl(args.gold)
    if len({(r.language, r.split) for r in records}) != 1:
        parser.error("Evaluate one language and split at a time")
    predictions = {}
    for line in args.predictions.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["example_id"] in predictions:
                raise ValueError("Duplicate prediction ID")
            predictions[row["example_id"]] = row["labels"]
    if set(predictions) != {r.example_id for r in records}:
        raise ValueError("Predictions must cover exactly the gold example IDs")
    vocabulary = None
    if args.source_train:
        source = [r for p in args.source_train for r in read_jsonl(p)]
        vocabulary = training_entity_vocabulary(source)
    scores, units = [], []
    for record in records:
        score = score_sentence(record.tokens, record.labels_coarse, predictions[record.example_id],
                               source_vocabulary=vocabulary, ignore_policy=args.ignore_policy)
        scores.append(score)
        units.append({"example_id": record.example_id, "dataset": record.dataset,
                      "source_document_id": record.source_document_id,
                      "gold_hash": object_sha256([record.tokens, record.labels_coarse]),
                      "counts": {k: score["all"][k] for k in ("tp", "fp", "fn")}})
    report = {"metrics": aggregate_scores(scores), "units": units,
              "language": records[0].language, "split": records[0].split,
              "source_vocabulary_hash": object_sha256(sorted(vocabulary)) if vocabulary is not None else None,
              "gold_file_hash": file_sha256(args.gold), "prediction_file_hash": file_sha256(args.predictions)}
    save_json(args.out, report)
    print(f"Evaluation saved: {args.out}")


if __name__ == "__main__":
    main()
