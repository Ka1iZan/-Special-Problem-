"""Synthetic integration fixtures only; no experimental results are generated."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CommandLineTests(unittest.TestCase):
    def run_script(self, script, *args, expected=0):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_conversion_scoring_and_document_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            conll, mapping = root / "test.conll", root / "mapping.json"
            conll.write_text("# newdoc id = d1\nAda B-PER\n\n# newdoc id = d2\nManila B-LOC\n\n", encoding="utf-8")
            mapping.write_text(json.dumps({"mapping": {"PER": "PER", "LOC": "LOC"}}), encoding="utf-8")
            gold = root / "gold.jsonl"
            self.run_script("prepare_data.py", conll, "--format", "conll", "--dataset", "toy",
                            "--language", "tag", "--split", "test", "--mapping", mapping, "--out", gold)
            self.run_script("audit_data.py", gold, "--out", root / "audit.json")
            for method, labels in (("a", ["B-PER", "B-LOC"]), ("b", ["O", "B-LOC"])):
                prediction = root / f"{method}.jsonl"
                prediction.write_text("\n".join(json.dumps({"example_id": f"toy:test:{i+1}", "labels": [label]})
                                                 for i, label in enumerate(labels)), encoding="utf-8")
                self.run_script("evaluate_predictions.py", "--gold", gold, "--predictions", prediction,
                                "--out", root / f"{method}.evaluation.json")
            spec = root / "pairs.json"
            spec.write_text(json.dumps([{"pair_id": "synthetic", "a": "a.evaluation.json", "b": "b.evaluation.json"}]), encoding="utf-8")
            self.run_script("compare_predictions.py", spec, "--draws", 100, "--out", root / "interval.json")
            interval = json.loads((root / "interval.json").read_text())
            self.assertAlmostEqual(interval["delta_f1_pp"], 100 / 3)
            self.assertEqual(interval["n_documents"], 2)

    def test_extra_annotation_columns_are_not_silently_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data.tsv").write_text("Name\tB-PER\tB-ORG\n", encoding="utf-8")
            (root / "map.json").write_text(json.dumps({"mapping": {"PER": "PER", "ORG": "ORG"}}), encoding="utf-8")
            result = self.run_script("prepare_data.py", root / "data.tsv", "--format", "conll",
                                    "--dataset", "toy", "--language", "tag", "--split", "train",
                                    "--mapping", root / "map.json", "--out", root / "out.jsonl", expected=1)
            self.assertIn("review annotation layers", result.stderr)
            self.assertFalse((root / "out.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
