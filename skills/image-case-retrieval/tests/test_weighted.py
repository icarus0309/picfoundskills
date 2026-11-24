import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline
import score_weighted
import weighted_retrieval as weighted


class WeightedTests(unittest.TestCase):
    def test_high_weight_header_can_move_and_is_deduplicated(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for header in (
                ["id", "query", "预期fid", "extra", "高权重预期fid"],
                ["id", "query", "extra", "预期fid", "other", "高权重预期fid"],
            ):
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.append(header)
                ws.append(
                    ["a", "red"] + ["nonreference"] * (len(header) - 3) + ["a；b，a"]
                )
                wb.save(root / weighted.CASES)
                wb.close()
                self.assertEqual(
                    score_weighted.label_sets(root, {"a": "red"}), {"a": {"a", "b"}}
                )

    def test_missing_references_and_extra_returns_stay_in_closed_set_metrics(self):
        out = score_weighted.measure(
            {"a": {"good", "missing"}, "b": set()},
            {"good": "x"},
            {"a": {"x", "y"}, "b": {"unjudged"}},
        )
        self.assertEqual(
            (out["total"]["precision"], out["total"]["coverage"]), (0.5, 0.5)
        )
        self.assertEqual(out["total"]["unscored_cases"], 1)
        self.assertEqual(out["cases"]["a"]["unmapped_fids"], 1)

    def test_prior_visual_decision_only_nominates_a_candidate(self):
        result = weighted.extend_pool(
            {"q": ["i1"]},
            {"q": "object"},
            {"i1": {}, "i2": {"summary": "rare object"}, "i3": {}},
            {"q": {"recall": ["rare"]}},
            {"q": {"i1": "0", "i3": "1"}},
        )
        self.assertEqual(result, {"q": ["i1", "i2", "i3"]})

    def test_blocked_image_never_reaches_decoder_or_provider(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            cases = {"q": "red"}
            inv = {"images": [{"id": "i", "path": "test.png"}]}
            api = Mock()
            weighted.finish(
                out, out, inv, cases, {"i": {}}, {"q": {"i": 3}}, api, 1, {"i"}
            )
            api.call.assert_not_called()
            self.assertIsNone(
                pipeline.read_json(out / "weighted_predictions.json")["q"]["i"]
            )
            self.assertEqual(
                len((out / "weighted_matches.csv").read_text("utf-8-sig").splitlines()),
                1,
            )


if __name__ == "__main__":
    unittest.main()
