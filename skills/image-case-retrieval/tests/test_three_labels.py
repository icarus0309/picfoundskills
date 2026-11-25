import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pipeline as p
import three_labels as t
from providers import ProviderContentBlocked
from score_weighted import measure


class FakeAPI:
    calls = []
    response = dict(label="hard_negative", reason="黑猫不满足白色", similarity="同为猫", exclusion="毛色为黑色")
    error = None

    def __init__(self, *args, **kwargs):
        pass

    def call(self, stage, prompt, images):
        self.calls.append(prompt)
        if self.error:
            raise self.error
        return self.response.copy()


class ThreeLabelsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.out = self.root / "out"
        photos = self.root / "图片"
        photos.mkdir()
        Image.new("RGB", (40, 40), "black").save(photos / "a.jpg")
        stat = (photos / "a.jpg").stat()
        inv = dict(cases=[dict(id="Q1", query="白猫"), dict(id="Q2", query="白色宠物")],
                   images=[dict(id="I0", path="a.jpg", size=stat.st_size, mtime_ns=stat.st_mtime_ns)])
        p.write_json(self.source / "inventory.json", inv)
        self.write_source({"Q1": {"I0": "0"}, "Q2": {"I0": "S"}})
        (self.root / "key").write_text("fake-local-test-key")
        self.config = self.root / "config.json"
        p.write_json(self.config, dict(base_url="https://fixture.invalid", model="fixture", key_file="key"))
        FakeAPI.calls = []
        FakeAPI.error = None

    def write_source(self, states):
        p.write_json(self.source / "predictions.json", dict(states=states))
        p.write_json(self.source / "predictions_seal.json", dict(sha256=t.sha(self.source / "predictions.json")))

    def run_classify(self, **kwargs):
        return t.classify(self.root, self.source, self.out, self.config, api_factory=FakeAPI, **kwargs)

    def test_hard_negative_evidence_required_and_all_labels_accepted(self):
        for label in t.LABELS:
            self.assertIsNotNone(t.parse(dict(FakeAPI.response, label=label)))
        self.assertIsNone(t.parse(dict(FakeAPI.response, exclusion="")))
        self.assertIsNone(t.parse(dict(FakeAPI.response, similarity="")))
        self.assertIsNone(t.parse(dict(FakeAPI.response, label="almost_positive")))

    def test_end_to_end_resume_and_unsearched(self):
        self.run_classify()
        result = p.read_json(self.out / "predictions.json")
        self.assertEqual(result["states"], {"Q1": {"I0": "H"}, "Q2": {"I0": "S"}})
        self.assertEqual(result["evidence"][0]["original_state"], "0")
        self.assertIn("黑猫", (self.out / "hard_negatives.csv").read_text(encoding="utf-8-sig"))
        self.assertEqual(len(FakeAPI.calls), 1)
        self.run_classify()
        self.assertEqual(len(FakeAPI.calls), 1)
        self.assertNotIn("original_state", FakeAPI.calls[0])
        self.assertFalse((self.out / "running.lock").exists())

    def test_complete_gallery_and_tamper_rejection(self):
        self.run_classify(all_images=True)
        self.assertEqual(len(FakeAPI.calls), 2)
        (self.out / "labels.csv").write_text("tampered")
        with self.assertRaisesRegex(ValueError, "Sealed"):
            self.run_classify(all_images=True)

    def test_source_changes_require_new_output(self):
        self.run_classify()
        self.write_source({"Q1": {"I0": "1"}, "Q2": {"I0": "S"}})
        with self.assertRaisesRegex(ValueError, "Inputs changed"):
            self.run_classify()

    def test_decode_failure_and_provider_block_are_not_negatives(self):
        with patch.object(p, "thumbnail", side_effect=OSError("decode")):
            self.run_classify()
        self.assertEqual(p.read_json(self.out / "predictions.json")["states"]["Q1"]["I0"], "E")
        self.out = self.root / "blocked"
        FakeAPI.error = ProviderContentBlocked()
        self.run_classify(all_images=True)
        self.assertEqual(len(FakeAPI.calls), 1)
        self.assertEqual(p.read_json(self.out / "predictions.json")["states"], {"Q1": {"I0": "?"}, "Q2": {"I0": "?"}})

    def test_failed_request_resumes_and_invalid_schema_abstains(self):
        FakeAPI.error = RuntimeError("temporary failure")
        with self.assertRaises(RuntimeError):
            self.run_classify()
        self.assertFalse((self.out / "running.lock").exists())
        self.assertFalse((self.out / "three_labels_seal.json").exists())
        FakeAPI.error = None
        with patch.object(FakeAPI, "response", dict(label="hard_negative")):
            self.run_classify()
        self.assertEqual(p.read_json(self.out / "predictions.json")["states"]["Q1"]["I0"], "?")

    def test_only_positive_counts_as_retrieved(self):
        self.run_classify()
        states = p.read_json(self.out / "predictions.json")["states"]
        outputs = {q: {i for i, v in row.items() if v == "1"} for q, row in states.items()}
        score = measure({"Q1": {"fid"}}, {"fid": "I0"}, outputs)
        self.assertEqual(score["total"]["tp"], 0)
        self.assertEqual(score["total"]["fn"], 1)

    def test_ranker_source_and_positive_output(self):
        inv = p.read_json(self.source / "inventory.json")
        protocol = dict(gallery_metadata=dict(inventory=t.family_inventory(inv)))
        p.write_json(self.source / "prediction_protocol.json", protocol)
        p.write_json(self.source / "predictions.json", dict(
            queries={"Q1": "白猫"}, ranker={"Q1": ["I0"]}, candidates={"Q1": ["I0"]}))
        (self.source / "matches.csv").write_text("case_id,image_id\n")
        p.write_json(self.source / "seal.json", dict(
            predictions=t.sha(self.source / "predictions.json"), protocol=t.signature(protocol),
            matches=t.sha(self.source / "matches.csv")))
        with patch.object(FakeAPI, "response", dict(FakeAPI.response, label="positive")):
            self.run_classify()
        self.assertEqual(p.read_json(self.out / "predictions.json")["states"], {"Q1": {"I0": "1"}})
        self.assertEqual(len(FakeAPI.calls), 0)

    def test_negative_positive_conflict_requires_review(self):
        with patch.object(FakeAPI, "response", dict(FakeAPI.response, label="positive")):
            self.run_classify()
        row = p.read_json(self.out / "predictions.json")["evidence"][0]
        self.assertEqual(row["state"], "?")
        self.assertIn("source_negative_conflict", row["reason"])

    def test_lock_prevents_concurrent_writer(self):
        self.out.mkdir()
        lock = self.out / "running.lock"
        lock.write_text("other process")
        with self.assertRaises(FileExistsError):
            self.run_classify()
        self.assertEqual(lock.read_text(), "other process")


if __name__ == "__main__":
    unittest.main()
