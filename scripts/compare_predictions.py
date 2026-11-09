import argparse
import json
from pathlib import Path
import _bootstrap  # noqa: F401
from spner.metrics import Counts
from spner.statistics import paired_document_bootstrap
from spner.utils import save_json, file_sha256


def main():
    parser = argparse.ArgumentParser(description="Paired document / crossed run-document bootstrap")
    parser.add_argument("pairs", type=Path, help='JSON list: [{"pair_id":"1","a":"eval_a.json","b":"eval_b.json"}]')
    parser.add_argument("--resample-pairs", action="store_true")
    parser.add_argument("--unit", choices=["document", "sentence"], default="document")
    parser.add_argument("--draws", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.pairs.read_text(encoding="utf-8"))
    pairs, signature, gold_signature = {}, None, None
    for pair in spec:
        if pair["pair_id"] in pairs:
            raise ValueError("Duplicate pair_id")
        evaluations = []
        for key in ("a", "b"):
            path = args.pairs.parent / pair[key]
            report = json.loads(path.read_text(encoding="utf-8"))
            current = (report["language"], report["split"], report["source_vocabulary_hash"],
                       report["metrics"]["ignore_policy"], report["metrics"]["bio_policy"])
            if signature is not None and signature != current:
                raise ValueError("Incompatible evaluation settings")
            signature = current
            units = {unit["example_id"]: unit for unit in report["units"]}
            if len(units) != len(report["units"]):
                raise ValueError("Duplicate evaluation example")
            current_gold = {key: (unit["gold_hash"], unit["dataset"], unit["source_document_id"])
                            for key, unit in units.items()}
            if gold_signature is not None and current_gold != gold_signature:
                raise ValueError("Gold labels, examples or clusters differ across compared predictions")
            gold_signature = current_gold
            evaluations.append(units)
        documents = {}
        for example_id, left in evaluations[0].items():
            doc = left["source_document_id"] if args.unit == "document" else example_id
            if not doc:
                raise ValueError("Missing true document IDs. Sentence resampling requires explicit --unit sentence")
            cluster = f"{left['dataset']}:{doc}"
            previous = documents.get(cluster, (Counts(), Counts()))
            documents[cluster] = (previous[0] + Counts(**left["counts"]),
                                  previous[1] + Counts(**evaluations[1][example_id]["counts"]))
        pairs[pair["pair_id"]] = documents
    result = paired_document_bootstrap(pairs, resample_pairs=args.resample_pairs,
                                       draws=args.draws, seed=args.seed)
    result["unit"] = args.unit
    result["pair_spec_hash"] = file_sha256(args.pairs)
    if args.unit == "sentence":
        result["warnings"].append("Assumes independent sentences; can understate uncertainty when documents are unknown.")
    save_json(args.out, result)
    print(f"Saved paired interval: {args.out}")


if __name__ == "__main__":
    main()
