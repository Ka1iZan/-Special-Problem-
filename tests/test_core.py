import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spner.data.bio import Span, bio_spans
from spner.data.harmonize import project_sequence
from spner.data.schema import NERSentence
from spner.data.io import read_conll, read_jsonl, write_jsonl
from spner.data.audit import audit_records
from spner.data.overlap import normalize_surface, training_entity_vocabulary
from spner.data.sampling import nested_target_subsets
from spner.data.tokenize import align_first_subwords
from spner.metrics import Counts, score_sentence, aggregate_scores, forgetting
from spner.statistics import sign_flip_test, paired_run_summary, holm_adjust, paired_document_bootstrap
from spner.replay import MemoryItem, ReplayIndex
from spner.analysis import compare_planned_runs


def record(identifier="one", split="train", tokens=None, labels=None, doc=None):
    tokens = tokens or ["Si", "Ada"]
    labels = labels or ["O", "B-PER"]
    return NERSentence(identifier, "tag", "toy", split, tokens, labels, labels,
                       " ".join(tokens), doc)


class DataTests(unittest.TestCase):
    def test_adjacent_entities_remain_distinct(self):
        self.assertEqual(bio_spans(["B-PER", "B-PER", "I-PER"]),
                         [Span(0, 1, "PER"), Span(1, 3, "PER")])

    def test_gold_invalid_transition_rejected(self):
        with self.assertRaises(ValueError):
            bio_spans(["O", "I-PER"])

    def test_projection_preserves_boundaries_and_ignores(self):
        self.assertEqual(project_sequence(["B-CITY", "I-CITY", "B-EVENT", "O"],
                                          {"CITY": "LOC", "EVENT": "IGNORE"}),
                         ["B-LOC", "I-LOC", "IGNORE", "O"])

    def test_unknown_mapping_rejected(self):
        with self.assertRaises(KeyError):
            project_sequence(["B-UNKNOWN"], {})

    def test_mapping_cannot_hide_invalid_bio(self):
        with self.assertRaises(ValueError):
            project_sequence(["I-EVENT"], {"EVENT": "IGNORE"})

    def test_schema_length_and_language(self):
        bad = record()
        bad.labels_coarse = ["O"]
        with self.assertRaises(ValueError):
            bad.validate()
        bad = record()
        bad.language = "unknown"
        with self.assertRaises(ValueError):
            bad.validate()

    def test_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_jsonl([record()], path)
            self.assertEqual(read_jsonl(path), [record()])
            with self.assertRaises(FileExistsError):
                write_jsonl([record()], path)

    def test_conll_keeps_document_id(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.conll"
            path.write_text("# newdoc id = doc1\n# sent_id = s1\nSi O\nAda B-PER\n\n", encoding="utf-8")
            rows = read_conll(path, dataset="toy", language="tag", split="train", mapping={"PER": "PER"})
            self.assertEqual(rows[0].source_document_id, "doc1")
            self.assertEqual(rows[0].example_id, "toy:train:s1")

    def test_leakage_across_datasets_and_splits(self):
        a, b = record("a"), record("b", "test")
        b.dataset = "other"
        report = audit_records([a, b])
        self.assertEqual(report["cross_split_duplicate_groups"], 1)

    def test_document_leak_without_duplicate_text(self):
        report = audit_records([record("a", doc="shared"),
                                record("b", "test", ["Ben"], ["B-PER"], "shared")])
        self.assertEqual(len(report["document_leaks"]), 1)

    def test_overlap_does_not_use_evaluation_labels(self):
        with self.assertRaises(ValueError):
            training_entity_vocabulary([record(split="test")])
        self.assertEqual(normalize_surface(["  MANILA "]), "manila")
        self.assertNotEqual(normalize_surface(["Manila,"]), "manila")

    def test_nested_order_invariant_subsets(self):
        rows = [record(str(i), tokens=[f"Name{i}"], labels=["B-PER"]) for i in range(40)]
        a = nested_target_subsets(rows, [5, 10, 25], 42)
        b = nested_target_subsets(rows[::-1], [25, 5, 10], 42)
        self.assertEqual(a, b)
        self.assertLessEqual(set(a["5"]["example_ids"]), set(a["10"]["example_ids"]))
        with self.assertRaises(ValueError):
            nested_target_subsets([record(split="test")], [10], 42)

    def test_first_subword_alignment_keeps_ignored_word(self):
        result = align_first_subwords([None, 0, 1, 1, 2, None], ["O", "B-PER", "IGNORE"])
        self.assertEqual(result["labels"], [-100, 0, 1, -100, -100, -100])
        self.assertEqual(result["first_subword_positions"], [1, 2, 4])

    def test_truncation_rejected(self):
        with self.assertRaises(ValueError):
            align_first_subwords([None, 0, None], ["O", "B-PER"])


class MetricTests(unittest.TestCase):
    def test_boundary_error_is_fp_and_fn(self):
        score = score_sentence(["Ada", "Lovelace", "Manila"], ["B-PER", "I-PER", "B-LOC"],
                                ["B-PER", "O", "B-LOC"])
        self.assertEqual([score["all"][k] for k in ("tp", "fp", "fn")], [1, 1, 1])
        self.assertEqual(score["all"]["f1"], .5)

    def test_invalid_predicted_i_is_not_repaired(self):
        score = score_sentence(["Ada"], ["B-PER"], ["I-PER"])
        self.assertEqual(score["all"]["fn"], 1)
        self.assertEqual(score["invalid_predicted_i_tags"], 1)

    def test_masks_do_not_glue_or_truncate_predicted_spans(self):
        score = score_sentence(["Ada", "event", "Ben"], ["B-PER", "IGNORE", "B-PER"],
                                ["B-PER", "I-PER", "I-PER"], ignore_policy="exclude_touching_spans")
        self.assertEqual(score["all"]["fn"], 2)
        self.assertEqual(score["all"]["tp"], 0)
        self.assertEqual(score["excluded_predicted_spans"], 1)

    def test_masks_require_explicit_policy(self):
        with self.assertRaises(ValueError):
            score_sentence(["event"], ["IGNORE"], ["O"])

    def test_overlap_false_positives_partition(self):
        score = score_sentence(["Ada", "Ben", "Office"], ["B-PER", "B-PER", "O"],
                                ["B-PER", "O", "B-ORG"], source_vocabulary={"ada", "office"})
        self.assertEqual(score["by_overlap"]["seen"]["fp"], 1)
        self.assertEqual(score["by_overlap"]["unseen"]["fn"], 1)
        for key in ("tp", "fp", "fn"):
            self.assertEqual(score["all"][key], sum(s[key] for s in score["by_overlap"].values()))

    def test_aggregate_uses_counts_not_sentence_f1(self):
        one = score_sentence(["Ada"], ["B-PER"], ["B-PER"])
        many = score_sentence(["B", "C", "D"], ["B-PER"] * 3, ["O"] * 3)
        self.assertAlmostEqual(aggregate_scores([one, many])["all"]["f1"], .4)

    def test_empty_scope_is_undefined_and_forgetting_signed(self):
        self.assertIsNone(Counts().metrics()["f1"])
        self.assertLess(forgetting(.6, .7), 0)


class StatisticsTests(unittest.TestCase):
    def test_five_pairs_cannot_reach_point05(self):
        self.assertEqual(sign_flip_test([1, 2, 3, 4, 5])["p_value"], .0625)

    def test_six_pairs_and_zero_pairs(self):
        self.assertEqual(sign_flip_test([1] * 6)["p_value"], .03125)
        self.assertEqual(sign_flip_test([0] * 5)["p_value"], 1)
        self.assertEqual(sign_flip_test([1, -1])["p_value"], 1)

    def test_holm_monotonic_adjustment(self):
        adjusted = holm_adjust([.03, .001, .04])
        for a, b in zip(adjusted, [.06, .003, .06]):
            self.assertAlmostEqual(a, b)

    def test_paired_summary_units_and_reproducibility(self):
        a, b = [.6, .7, .8], [.5, .6, .7]
        result = paired_run_summary(a, b, draws=200)
        self.assertAlmostEqual(result["delta_f1_pp"], 10)
        self.assertAlmostEqual(result["ci95_low_pp"], 10)
        self.assertEqual(result, paired_run_summary(a, b, draws=200))
        with self.assertRaises(ValueError):
            paired_run_summary([60, 70], [50, 60])

    def test_document_bootstrap_recomputes_micro_f1(self):
        pairs = {"seed1": {"d1": (Counts(1), Counts(0, 0, 1)),
                            "d2": (Counts(0, 0, 9), Counts(0, 0, 9))}}
        result = paired_document_bootstrap(pairs, draws=200)
        self.assertAlmostEqual(result["delta_f1_pp"], 100 * 2 / 11)
        self.assertEqual(result["n_documents"], 2)

    def test_crossed_bootstrap_identical_methods(self):
        pairs = {str(i): {"d1": (Counts(2, 1, 1), Counts(2, 1, 1)),
                          "d2": (Counts(1), Counts(1))} for i in range(3)}
        result = paired_document_bootstrap(pairs, resample_pairs=True, draws=100)
        self.assertEqual(result["ci95_low_pp"], 0)
        self.assertEqual(result["ci95_high_pp"], 0)

    def test_bootstrap_rejects_mismatched_gold_support(self):
        with self.assertRaises(ValueError):
            paired_document_bootstrap({"s": {"d1": (Counts(1), Counts(2)),
                                              "d2": (Counts(1), Counts(1))}}, draws=100)

    def test_analysis_rejects_unmatched_subsets_and_missing_family(self):
        plan = {"plan_id": "toy", "status": "synthetic", "targets": ["hil"], "budget_percent": 10,
                "pair_match_fields": ["subset_id"], "analysis_seed": 1, "bootstrap_draws": 100,
                "alpha": .05, "contrasts": [{"id": "toy", "a": "a", "b": "b", "endpoint": "target_f1", "extra_match_fields": []}]}
        rows = [{"run_id": f"{m}{s}", "method": m, "pair_id": str(s), "target": "hil",
                 "budget_percent": 10, "subset_id": str(s), "target_f1": .7 if m == "a" else .6}
                for m in ("a", "b") for s in range(3)]
        self.assertEqual(len(compare_planned_runs(rows, plan)), 1)
        with self.assertRaises(ValueError):
            compare_planned_runs(rows[:-1], plan)
        rows[-1]["subset_id"] = "wrong"
        with self.assertRaises(ValueError):
            compare_planned_runs(rows, plan)


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.index = ReplayIndex([MemoryItem("a", "tag", frozenset({"PER", "LOC"})),
                                  MemoryItem("b", "tag", frozenset({"ORG"}))])

    def test_zero_drift_equals_uniform_groups(self):
        groups = self.index.group_probabilities()
        self.assertEqual(groups, self.index.group_probabilities({group: 0 for group in groups}))
        self.assertAlmostEqual(sum(groups.values()), 1)

    def test_uniform_groups_is_not_uniform_sentences(self):
        rng = random.Random(5)
        n_a = sum(self.index.sample_group(rng).example_id == "a" for _ in range(3000))
        self.assertTrue(1800 < n_a < 2200)

    def test_missing_drift_and_duplicate_items_rejected(self):
        with self.assertRaises(ValueError):
            self.index.group_probabilities({("tag", "PER"): .1})
        with self.assertRaises(ValueError):
            ReplayIndex([MemoryItem("a", "tag", frozenset()), MemoryItem("a", "tag", frozenset())])


if __name__ == "__main__":
    unittest.main()
