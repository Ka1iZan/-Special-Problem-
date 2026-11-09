"""Descriptive release audit, without silently choosing/repairing gold annotations."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import statistics
import _bootstrap  # noqa: F401
from spner.data.bio import bio_spans, invalid_i_count
from spner.data.audit import content_hash
from spner.utils import save_json, file_sha256


def conll_inventory(path):
    blocks, tokens, labels = [], [], []
    widths, tag_counts = Counter(), Counter()
    docstart, malformed = 0, []
    def flush():
        nonlocal tokens, labels
        if tokens:
            blocks.append((tokens, labels))
        tokens, labels = [], []
    for number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("-DOCSTART-"):
            flush()
            docstart += 1
        else:
            fields = line.split()
            widths[len(fields)] += 1
            if len(fields) != 4:
                malformed.append(number)
                continue
            tokens.append(fields[0])
            labels.append(fields[-1])
            tag_counts[fields[-1]] += 1
    flush()
    invalid, unsupported = [], []
    for index, (_, tags) in enumerate(blocks):
        try:
            bio_spans(tags)
        except ValueError:
            invalid.append(index + 1)
        if any(tag != "O" and not tag.startswith(("B-", "I-")) for tag in tags):
            unsupported.append(index + 1)
    hashes = Counter(content_hash(tokens) for tokens, _ in blocks)
    return {"blocks": len(blocks), "tokens": sum(len(tokens) for tokens, _ in blocks),
            "column_widths": dict(widths), "label_counts": dict(sorted(tag_counts.items())),
            "docstart_markers": docstart, "malformed_line_numbers": malformed,
            "invalid_bio_block_ids": invalid, "invalid_bio_blocks": len(invalid),
            "non_bio_block_ids": unsupported,
            "duplicate_text_groups": sum(n > 1 for n in hashes.values()),
            "extra_duplicate_records": sum(n - 1 for n in hashes.values()),
            "max_words": max(len(tokens) for tokens, _ in blocks)}, blocks


def hiligaynon_inventory(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels, fields, lengths = Counter(), set(), []
    invalid_offsets, overlap_rows, partial_word_rows, invalid_bio_rows = [], [], [], []
    duplicates = defaultdict(list)
    for number, row in enumerate(rows, 1):
        fields.update(row)
        text = row["text"]
        words = list(re.finditer(r"\S+", text))
        lengths.append(len(words))
        duplicates[content_hash([text])].append(number)
        starts = {m.start(): i for i, m in enumerate(words)}
        ends = {m.end(): i for i, m in enumerate(words)}
        word_tags = ["O"] * len(words)
        valid, overlaps, partial, last_end = True, False, False, -1
        for start, end, label in sorted(row["entities"], key=lambda entry: (entry[0], entry[1])):
            labels[label] += 1
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                valid = False
                continue
            if start < last_end:
                overlaps = True
            last_end = max(last_end, end)
            if start not in starts or end not in ends:
                partial = True
                continue
            first, last = starts[start], ends[end]
            for word in range(first, last + 1):
                word_tags[word] = label if word == first else "I-" + label.split("-", 1)[-1]
        if not valid:
            invalid_offsets.append(number)
        if overlaps:
            overlap_rows.append(number)
        if partial:
            partial_word_rows.append(number)
        if valid and not overlaps and not partial:
            try:
                bio_spans(word_tags)
            except ValueError:
                invalid_bio_rows.append(number)
    return {"records": len(rows), "record_fields": sorted(fields),
            "annotation_entry_counts": dict(sorted(labels.items())),
            "word_tokens": sum(lengths), "word_length_median": statistics.median(lengths),
            "word_length_max": max(lengths), "invalid_offset_rows": invalid_offsets,
            "overlapping_annotation_rows": overlap_rows,
            "whitespace_boundary_mismatch_rows": partial_word_rows,
            "invalid_bio_on_unambiguous_rows": invalid_bio_rows,
            "duplicate_text_groups": [ids for ids in duplicates.values() if len(ids) > 1],
            "note": "Annotation entries may be BIO-labelled pieces, not whole entities; no split metadata in text/entities alone."}


def tl_inventory(root):
    import pyarrow.parquet as pq
    summaries, duplicates, by_split = [], defaultdict(list), {}
    for split in ("train", "validation", "test"):
        path = root / f"{split}.parquet"
        table = pq.read_table(path)
        hf = json.loads(table.schema.metadata[b"huggingface"])
        feature = hf["info"]["features"]["ner_tags"]
        label_names = feature.get("feature", feature.get("element"))["names"]
        rows = table.to_pylist()
        by_split[split] = (rows, label_names)
        counts, lengths, bad = Counter(), [], []
        for row in rows:
            labels = [label_names[index] for index in row["ner_tags"]]
            lengths.append(len(row["tokens"]))
            duplicates[content_hash(row["tokens"])].append(f"{split}:{row['id']}")
            if len(labels) != len(row["tokens"]):
                bad.append(row["id"])
                continue
            try:
                counts.update(span.label for span in bio_spans(labels))
            except ValueError:
                bad.append(row["id"])
        summaries.append({"split": split, "records": len(rows), "tokens": sum(lengths),
                          "entity_counts_valid_sequences": dict(counts), "invalid_record_ids": bad,
                          "word_length_max": max(lengths), "label_names": label_names})
    duplicate_sets = [ids for ids in duplicates.values() if len(ids) > 1]
    cross_split = [ids for ids in duplicate_sets if len({x.split(":")[0] for x in ids}) > 1]
    return {"splits": summaries, "duplicate_groups": duplicate_sets,
            "cross_split_duplicate_groups": cross_split,
            "note": "Released row IDs are retained; no parent document IDs were invented."}, by_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/public_v1"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/data_audit/public_releases.json"))
    args = parser.parse_args()
    tl, _ = tl_inventory(args.root / "tlunified_ner")
    ceb1, blocks1 = conll_inventory(args.root / "cebuaner/final-1.txt")
    ceb2, blocks2 = conll_inventory(args.root / "cebuaner/final-2.txt")
    aligned = len(blocks1) == len(blocks2) and all(a[0] == b[0] for a, b in zip(blocks1, blocks2))
    comparison = {"exact_token_and_block_alignment": aligned}
    if aligned:
        total = sum(len(a[1]) for a in blocks1)
        differences = sum(x != y for a, b in zip(blocks1, blocks2) for x, y in zip(a[1], b[1]))
        comparison.update({"tokens_compared": total, "different_token_labels": differences,
                           "raw_token_agreement": 1 - differences / total,
                           "note": "Descriptive token agreement, dominated by O; not entity-level annotation reliability."})
    hil = hiligaynon_inventory(args.root / "hiligayner/final_combined_all.jsonl")
    report = {"status": "descriptive_raw_data_audit_no_model_results", "tlunified_ner": tl,
              "cebuaner": {"final_1": ceb1, "final_2": ceb2, "annotator_comparison": comparison},
              "hiligayner": hil, "raw_manifest_hash": file_sha256(args.root / "manifest.json")}
    save_json(args.out, report)
    print(json.dumps({"tlunified_splits": [{k: v for k, v in s.items() if k != "label_names"} for s in tl["splits"]],
                      "tlunified_cross_split_duplicate_groups": len(tl["cross_split_duplicate_groups"]),
                      "cebuaner": {"blocks": ceb1["blocks"], "invalid_bio_blocks": ceb1["invalid_bio_blocks"],
                                    "annotation_comparison": comparison, "labels": list(ceb1["label_counts"])},
                      "hiligayner": {"records": hil["records"], "labels": hil["annotation_entry_counts"],
                                      "invalid_offsets": len(hil["invalid_offset_rows"]),
                                      "overlapping_rows": len(hil["overlapping_annotation_rows"]),
                                      "boundary_mismatch_rows": len(hil["whitespace_boundary_mismatch_rows"]),
                                      "invalid_bio_rows": len(hil["invalid_bio_on_unambiguous_rows"]),
                                      "duplicate_groups": len(hil["duplicate_text_groups"])}}, indent=2))


if __name__ == "__main__":
    main()
