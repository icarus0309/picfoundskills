"""Diagnostic replay, followed by a separate refit excluding every additional eligible venue query."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace as Args

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills/image-case-retrieval/scripts")
)
import pipeline as p
import lexical_ranker as rank
import lexical_venue as venue
from lexical_split import keys
from weighted_retrieval import sha


def run(root, d, out):
    g = d / "gallery"
    plans_file = d / "plans.json"
    plans = p.read_json(plans_file)
    s = p.read_json(d / "split.json")
    name = p.read_json(d / "choice.json")["chosen"]
    features = d / name
    e = d / "venue-followup"
    e.mkdir(exist_ok=True)
    held = [q for q in s["train"] + s["calibration"] if venue.applies(plans[q])]
    seen = set().union(*(keys(plans[q]) for q in held))
    kept = {
        k: [q for q in s[k] if q not in held and not seen.intersection(keys(plans[q]))]
        for k in ("train", "calibration")
    }
    revised = dict(**kept, test=held, plans_sha256=sha(plans_file))
    split = e / "split.json"
    if split.exists() and p.read_json(split) != revised:
        raise ValueError("Changed follow-up split")
    p.write_json(split, revised)
    common = dict(gallery=g, features=features, plans=plans_file, split=split)
    rank.train(Args(**common, root=root, out=e / "model"))
    rank.predict(
        Args(
            **common,
            model=e / "model",
            out=e / "original",
            contrast=d / "contrast.json",
            facts=g / "facts.json",
        )
    )
    for source, target in (
        (e / "original", e / "filtered"),
        (d / ("pred-" + name), e / "diagnostic"),
    ):
        venue.apply(g, plans_file, source, target)
    for target in (e / "filtered", e / "original", e / "diagnostic"):
        rank.score(Args(root=root, gallery=g, out=target))

    def get(path):
        return p.read_json(path / "scores.json")["ranker"]

    data = {
        "rule": "venue kind with explicit place groups; at least one written alias from each group",
        "new_split": {k: len(revised[k]) for k in ("train", "calibration", "test")},
        "refit_split_sha256": sha(split),
        "additional_without_rule": get(e / "original"),
        "additional_with_rule": get(e / "filtered"),
        "diagnostic_155": get(e / "diagnostic")["total"],
        "diagnostic_venue_cases": {
            q: r
            for q, r in get(e / "diagnostic")["cases"].items()
            if venue.applies(plans[q]) and q in s["test"]
        },
    }
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode())


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "run", "out"):
        cli.add_argument("--" + k, type=Path, required=True)
    a = cli.parse_args()
    run(a.root, a.run, a.out)
