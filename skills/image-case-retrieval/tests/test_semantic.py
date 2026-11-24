import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import boundary_review
import pipeline as p
from lexical_facts import family_inventory, signature
from score_weighted import label_sets
from semantic_features import augment, graph
from semantic_projection import grouped_folds
from example_rerank import corroboration
from lexical_ranker import train
from cross_rerank import select
from dense_candidates import top_ids
from weighted_retrieval import sha


class SemanticRegressionTests(unittest.TestCase):
    def test_grouped_folds_keep_transitive_aliases_together(self):
        folds = grouped_folds(
            [["bird"], ["bird", "avian"], ["avian"], ["train"], ["bus"]]
        )
        self.assertEqual(len(set(folds[:3])), 1)
        self.assertGreater(len(set(folds)), 1)

    def test_parse_grades_accepts_digit_strings_and_rejects_ambiguous_results(self):
        ask = {"0": "a", "1": "b"}
        self.assertEqual(
            boundary_review.parse_grades({"grades": {"0": "4", "1": 1}}, ask),
            {"0": 4, "1": 1},
        )
        for values in (
            {"0": True, "1": 4},
            {"0": 1, "1": "1 or 3"},
            {"0": "4"},
            {"0": 5, "1": 2},
        ):
            self.assertEqual(boundary_review.parse_grades({"grades": values}, ask), {})

    def test_dense_candidates_never_return_missing_images(self):
        result = top_ids(
            np.array([0.5, 0.8, 0.8]), np.array([True, False, True]), 2, ["a", "b", "c"]
        )
        self.assertEqual(result, ["c", "a"])

    def test_cross_encoder_preserves_volume_when_prefix_is_shorter(self):
        scores = np.array([0.91, 0.8, 0.7, 0.6, 0.1])
        self.assertEqual(select(scores, [0.1, 0.9], 4, 1).tolist(), [1, 0, 2, 3])
        self.assertEqual(select(scores, [0.1, 0.9], 0, 0).tolist(), [])

    def test_visual_examples_require_positive_and_negative_evidence(self):
        vectors = np.asarray([[1, 0], [-1, 0], [0, 1]], np.float32)
        self.assertTrue(
            np.array_equal(corroboration([vectors], [0, 1, 2], [0], [4]), [0, 0, 0])
        )
        signal = corroboration([vectors], [0, 1, 2], [0, 1], [4, 1])
        self.assertGreater(signal[0], 0)
        self.assertLess(signal[1], 0)
        self.assertEqual(signal[2], 0)

    def test_projection_training_rejects_a_different_cross_fit_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p.write_json(d / "ids.json", ["A"])
            p.write_json(d / "plans.json", {})
            p.write_json(d / "split.json", {"plans_sha256": sha(d / "plans.json")})
            p.write_json(
                d / "manifest.json",
                dict(
                    ids_hash=sha(d / "ids.json"),
                    protocol={"plans": sha(d / "plans.json"), "projection_index": "p"},
                ),
            )
            p.write_json(
                d / "projection_mode.json",
                {
                    "cross_fit": True,
                    "split": "other",
                    "manifest": sha(d / "manifest.json"),
                },
            )
            with self.assertRaisesRegex(ValueError, "cross-fitted"):
                train(
                    SimpleNamespace(
                        out=d / "out",
                        features=d,
                        plans=d / "plans.json",
                        split=d / "split.json",
                    )
                )

    def test_graph_excludes_self_and_missing_images(self):
        vectors = np.array([[1, 0], [0.9, 0.1], [-1, 0], [0, 0]], dtype=np.float32)
        neighbors = graph(vectors, np.array([True, True, True, False]), k=9)
        self.assertEqual(neighbors.shape, (4, 2))
        self.assertEqual(neighbors[0, 0], 1)
        for i in range(3):
            self.assertNotIn(i, neighbors[i])
            self.assertNotIn(3, neighbors[i])

    def test_rejects_cross_gallery_feature_join(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            p.write_json(
                d / "manifest.json",
                {"protocol": {"gallery": "old", "source_index": "same"}},
            )
            p.write_json(d / "ids.json", ["A"])
            p.write_json(d / "plans.json", {})
            p.write_json(d / "gallery.json", {"docs_sha256": "new", "index": "same"})
            with self.assertRaisesRegex(ValueError, "documents"):
                augment(d, d, d / "plans.json", d / "out", SimpleNamespace(), "text")

    def test_named_workbook_does_not_require_legacy_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            book = openpyxl.Workbook()
            book.active.append(["用例编号", "查询", "高权重预期fid"])
            book.active.append(["Q", "一张图", "x,y"])
            book.save(d / "new.xlsx")
            self.assertEqual(
                label_sets(d, {"Q": "一张图"}, workbook=d / "new.xlsx"),
                {"Q": {"x", "y"}},
            )

    def test_visual_abstention_is_preserved_and_crossings_are_auditable(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            source, review, out = d / "source", d / "review", d / "final"
            inv = {
                "images": [
                    dict(id=i, path=i + ".jpg", size=1, mtime_ns=2)
                    for i in ("i1", "i2", "i3")
                ]
            }
            p.write_json(d / "inventory.json", inv)
            proto = {"gallery_metadata": {"inventory": family_inventory(inv)}}
            p.write_json(source / "prediction_protocol.json", proto)
            pred = {
                "queries": {"q": "鸟"},
                "ranker": {"q": ["i1", "i2"]},
                "candidates": {"q": ["i1", "i2", "i3"]},
                "literal": {"q": []},
            }
            p.write_json(source / "predictions.json", pred)
            with (source / "matches.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "case_id",
                        "query",
                        "image_id",
                        "relative_path",
                        "score",
                        "evidence",
                    ]
                )
                writer.writerows(
                    [["q", "鸟", i, i + ".jpg", 3, "{}"] for i in ("i1", "i2")]
                )
            seal = {
                "predictions": sha(source / "predictions.json"),
                "protocol": signature(proto),
                "matches": sha(source / "matches.csv"),
            }
            p.write_json(source / "seal.json", seal)
            p.write_json(review / "review_protocol.json", {"parent": seal, "depth": 2})
            p.write_json(
                review / "reviews.json",
                {
                    "q": [
                        {"image": "i1", "grade": None, "offset": -2, "score": 3},
                        {"image": "i2", "grade": 1, "offset": -1, "score": 3},
                        {"image": "i3", "grade": 4, "offset": 0, "score": 1},
                    ]
                },
            )
            p.write_json(
                review / "review_seal.json",
                {
                    "data": sha(review / "reviews.json"),
                    "protocol": sha(review / "review_protocol.json"),
                },
            )
            boundary_review.apply(d, source, review, out, 1, 4, 2)
            final, _, _ = boundary_review.read_parent(out)
            self.assertEqual(final["ranker"]["q"], ["i1", "i3"])
            self.assertEqual(p.read_json(source / "predictions.json"), pred)


if __name__ == "__main__":
    unittest.main()
