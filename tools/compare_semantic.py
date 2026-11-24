"""Reproduce the six reranking comparisons from a prepared gallery and frozen query split."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = (
    Path(__file__).resolve().parents[1] / "skills" / "image-case-retrieval" / "scripts"
)
sys.path.insert(0, str(SCRIPTS))
import pipeline as p
from weighted_retrieval import sha

MODES = ("baseline", "text", "projection", "pixel", "hard", "feedback")


def command(script, *arguments):
    subprocess.run(
        [sys.executable, "-X", "utf8", str(SCRIPTS / script), *map(str, arguments)],
        check=True,
    )


def compare(a):
    a.out.mkdir(parents=True, exist_ok=True)
    base = a.prepared
    gallery, plans, split = (
        base / "gallery",
        base / "plans.json",
        a.split or base / "split.json",
    )
    text, pixel = a.out / "text", a.out / "pixel"
    bank, projection = a.out / "private-bank", a.out / "projection-model"
    inputs = ["--gallery", gallery, "--plans", plans]
    labels = ["--root", a.root, "--reference-workbook", a.workbook]
    text_args = ["--index", a.text_index, "--model", a.text_model]
    features = dict(
        baseline=base / "visual-album",
        text=text,
        projection=a.out / "projection",
        pixel=pixel,
        hard=pixel,
        feedback=a.out / "feedback",
    )
    command(
        "semantic_index.py",
        "--gallery",
        gallery,
        "--model",
        a.text_model,
        "--out",
        a.text_index,
    )
    command(
        "semantic_features.py",
        "text",
        *inputs,
        "--source",
        features["baseline"],
        *text_args,
        "--out",
        text,
    )
    command(
        "semantic_features.py",
        "pixel",
        *inputs,
        "--source",
        text,
        "--index",
        a.clip_index,
        "--model",
        a.clip_model,
        "--out",
        pixel,
    )
    command(
        "semantic_feedback.py",
        "bank",
        *inputs,
        *labels,
        "--model",
        a.text_model,
        "--split",
        split,
        "--out",
        bank,
    )
    command(
        "semantic_projection.py",
        "fit",
        "--bank",
        bank,
        "--index",
        a.text_index,
        "--out",
        projection,
    )
    command(
        "semantic_projection.py",
        "augment",
        *inputs,
        *text_args,
        "--source",
        text,
        "--projection",
        projection,
        "--cross-fit",
        "--out",
        features["projection"],
    )
    command(
        "semantic_feedback.py",
        "augment",
        *inputs,
        "--source",
        pixel,
        "--bank",
        bank,
        "--model",
        a.text_model,
        "--out",
        features["feedback"],
    )
    decisions = {}
    for name in MODES:
        model = a.out / ("fit-" + name)
        command(
            "lexical_ranker.py",
            "fit",
            *labels,
            *inputs,
            "--features",
            features[name],
            "--split",
            split,
            "--out",
            model,
            *(["--mine-semantic"] if name == "hard" else []),
        )
        decisions[name] = p.read_json(model / "model.json")["decision"]

    def calibrated(name):
        v = decisions[name]
        return min(v["precision"], v["coverage"]), v["f1"]

    choice = dict(
        winner=max(MODES, key=calibrated),
        portable=max(MODES[:-1], key=calibrated),
        decisions=decisions,
        split=sha(split),
        reference=sha(a.workbook),
    )
    lock = a.out / "calibration_choice.json"
    if lock.exists() and p.read_json(lock) != choice:
        raise ValueError("The frozen experiment has different inputs or decisions")
    if not lock.exists() and any(
        (a.out / ("pred-" + n) / "scores.json").exists() for n in MODES
    ):
        raise ValueError("Use an output directory without previous test scoring")
    p.write_json(lock, choice)
    for name in MODES:
        command(
            "lexical_ranker.py",
            "predict",
            *inputs,
            "--features",
            features[name],
            "--model",
            a.out / ("fit-" + name),
            "--split",
            split,
            "--contrast",
            base / "contrast.json",
            "--out",
            a.out / ("pred-" + name),
        )
    for name in MODES:
        command(
            "lexical_ranker.py",
            "score",
            *labels,
            "--gallery",
            gallery,
            "--out",
            a.out / ("pred-" + name),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for k in (
        "root",
        "workbook",
        "prepared",
        "out",
        "text-index",
        "text-model",
        "clip-index",
        "clip-model",
    ):
        parser.add_argument("--" + k, type=Path, required=True)
    parser.add_argument(
        "--split", type=Path, help="Frozen split, defaults to prepared/split.json"
    )
    compare(parser.parse_args())
