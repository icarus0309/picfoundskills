"""Freeze a topic holdout and purge cross-split exact head/alias collisions before loading labels."""

import argparse
import re
import unicodedata
from pathlib import Path
import pipeline as p
from weighted_retrieval import sha

TEST = {"animal", "zoo", "gate", "bamboo", "bridge", "instrument"}
CALIBRATION = {"vehicle", "flower"}


def keys(plan):
    def clean(t):
        t = unicodedata.normalize("NFKC", t).casefold()
        return re.sub(
            r"[\s，。？！、]|^(帮我找下|帮我搜下|搜下|查找)|(?:的照片|照片|图片)$",
            "",
            t,
        )

    return {
        k
        for t in [plan["query"], *(t for g in plan["heads"] for t in g)]
        if (k := clean(t))
    }


def partition(plans):
    raw = {"test": [], "calibration": [], "train": []}
    purged = {}
    for q, plan in plans.items():
        ds = set(plan["domains"])
        bucket = "test" if ds & TEST else "calibration" if ds & CALIBRATION else "train"
        raw[bucket].append(q)
    used = set()
    result = {}
    for bucket in ("test", "calibration", "train"):
        retained = []
        for q in raw[bucket]:
            if used & keys(plans[q]):
                purged[q] = "shared_head_with_higher_priority_split"
            else:
                retained.append(q)
        result[bucket] = retained
        for q in retained:
            used.update(keys(plans[q]))
    return {**result, "purged": purged}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--plans", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    a = cli.parse_args()
    result = partition(p.read_json(a.plans))
    result["plans_sha256"] = sha(a.plans)
    if a.out.exists() and p.read_json(a.out) != result:
        raise ValueError("Frozen split differs; use a new directory")
    p.write_json(a.out, result)
    print({k: len(result[k]) for k in ("train", "calibration", "test", "purged")})
