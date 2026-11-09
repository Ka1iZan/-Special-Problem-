import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.data.io import read_conll, read_jsonl, write_jsonl
from spner.utils import file_sha256, save_json


def main():
    parser = argparse.ArgumentParser(description="Explicit canonical JSONL / simple CoNLL conversion")
    parser.add_argument("input", type=Path)
    parser.add_argument("--format", choices=["canonical", "conll"], required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--language", choices=["tag", "ceb", "hil"])
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--token-column", type=int, default=0)
    parser.add_argument("--label-column", type=int, default=-1)
    parser.add_argument("--expected-columns", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.format == "canonical":
        records = read_jsonl(args.input)
    else:
        if not all([args.dataset, args.language, args.split, args.mapping]):
            parser.error("CoNLL requires dataset, language, split and an audited mapping JSON")
        mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
        records = read_conll(args.input, dataset=args.dataset, language=args.language,
                             split=args.split, mapping=mapping["mapping"],
                             token_column=args.token_column, label_column=args.label_column,
                             expected_columns=args.expected_columns)
    write_jsonl(records, args.out)
    save_json(args.out.with_suffix(".provenance.json"), {
        "input": str(args.input.resolve()), "input_sha256": file_sha256(args.input),
        "output_sha256": file_sha256(args.out), "n_records": len(records),
        "mapping_sha256": file_sha256(args.mapping) if args.mapping else None,
        "format": args.format, "token_column": args.token_column, "label_column": args.label_column,
        "expected_columns": args.expected_columns})
    print(f"Wrote {len(records)} validated records to {args.out}")


if __name__ == "__main__":
    main()
