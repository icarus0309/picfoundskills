import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lexical_venue import proof
import pipeline as p
import lexical_features as features
from lexical_facts import prepare
from lexical_index import LexicalIndex
from lexical_ranker import check_gallery, threshold
from lexical_album import augment
from lexical_split import partition
from weighted_retrieval import INDEX


def plan(text, domain="other", aliases=()):
    return dict(
        query=text,
        heads=[[text, *aliases]],
        examples=[],
        context=[],
        facets=[],
        places=[],
        years=[],
        any_head=False,
        kind="object",
        domains=[domain],
    )


class LexicalTests(unittest.TestCase):
    def test_scope_purges_cross_theme_synonyms(self):
        split = partition(
            {
                "a": plan("白虎", "animal", ["白老虎"]),
                "b": plan("白老虎"),
                "c": plan("小车", "vehicle"),
                "d": plan("窗帘"),
            }
        )
        self.assertEqual(
            (split["test"], split["calibration"], split["train"]), (["a"], ["c"], ["d"])
        )
        self.assertIn("b", split["purged"])

    def test_confuser_alone_does_not_nominate(self):
        idx = LexicalIndex(
            {
                "i1": {"summary": "餐桌"},
                "i2": {"summary": "鸟居"},
                "i3": {"summary": "大门"},
            }
        )
        q = dict(
            plan("大门"),
            contrast={
                "equivalent": ["鸟居"],
                "confuser": ["餐桌"],
                "typical": [],
                "detail": [],
            },
        )
        doc, _, names = idx.features(q)
        self.assertEqual([idx.ids[j] for j in doc], ["i2", "i3"])
        self.assertIn("confuser_title", names)

    def test_gallery_without_cases_and_damaged_cache(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            folder = root / "图片" / "随手拍"
            folder.mkdir(parents=True)
            (folder / "one.jpg").write_bytes(b"one")
            b = openpyxl.Workbook()
            b.active.append(["fid", "fileName", "fileSize", "summary"])
            b.active.append(["private", "one.jpg", 3, "入口"])
            b.save(root / INDEX)
            b.close()
            g = root / "gallery"
            f = root / "features"
            plans = root / "plans.json"
            p.write_json(plans, {"new": plan("入口")})
            prepare(root, g)
            features.build(g, plans, f)
            m = p.read_json(f / "manifest.json")
            self.assertEqual(len(features.load_block(f, m, "new")[0]), 1)
            block = f / features.block_name("new")
            block.write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                features.load_block(f, m, "new")
            features.build(g, plans, f)
            features.load_block(f, p.read_json(f / "manifest.json"), "new")
            inv = p.read_json(g / "inventory.json")
            inv["images"][0]["path"] = "elsewhere.jpg"
            p.write_json(g / "inventory.json", inv)
            with self.assertRaises(ValueError):
                check_gallery(g, m["protocol"])

    def test_album_never_drops_candidates(self):
        x = np.array([[0], [1], [0]], dtype=np.float32)
        doc = np.array([1, 2, 3], np.int32)
        result, cols = augment(
            x, doc, ["head_title"], np.array([0, 0, 1, 1]), np.array([2, 2])
        )
        self.assertEqual(len(result), 3)
        np.testing.assert_array_equal(result[:, 0], x[:, 0])
        self.assertGreater(
            result[2, cols.index("album_head_title_rate")],
            result[0, cols.index("album_head_title_rate")],
        )

    def test_named_venue_requires_a_witness_for_each_place(self):
        request = {"places": [["六盘水动物园", "六盘水市动物园"]]}
        self.assertFalse(proof({"summary": "其他城市的动物园"}, request))
        self.assertIn("六盘水动物园", proof({"summary": "六盘水动物园正门"}, request))
        request["places"].append(["其他明确地点"])
        self.assertFalse(proof({"ocr": "六盘水动物园"}, request))

    def test_equal_scores_cross_threshold_together(self):
        selected = threshold(np.array([0.9, 0.9]), np.array([True, False]), 1)
        self.assertEqual((selected["tp"], selected["fp"]), (1, 1))
