"""Recompute published precision and F-column coverage directly from anonymous case counts."""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


def verify(directory):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))
    groups = defaultdict(dict)
    with (directory / "cases.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = row["cohort"] + "/" + row["method"]
            case = row["case_id"]
            if case in groups[key]:
                raise ValueError(f"Duplicate {key}/{case}")
            counts = {
                k: int(row[k])
                for k in ("tp", "fp", "fn", "reference_fids", "unmapped_fids")
            }
            if (
                min(counts.values()) < 0
                or counts["tp"] + counts["fn"] != counts["reference_fids"]
                or counts["unmapped_fids"] > counts["fn"]
            ):
                raise ValueError(f"Inconsistent counts for {key}/{case}")
            groups[key][case] = counts
    if set(groups) != set(summary["methods"]):
        raise ValueError("Missing or unexpected variant")
    cohorts = {}
    for key, cases in groups.items():
        cohort = key.split("/")[0]
        refs = {q: c["reference_fids"] for q, c in cases.items()}
        if refs != cohorts.setdefault(cohort, refs):
            raise ValueError(
                f"Comparison mixes different labels or queries in {cohort}"
            )
        subset = [c for c in cases.values() if c["reference_fids"]]
        tp, fp, fn = (sum(c[k] for c in subset) for k in ("tp", "fp", "fn"))

        def ratio(n, d):
            return n / d if d else 0.0

        expected = dict(
            tp=tp,
            fp=fp,
            fn=fn,
            precision=ratio(tp, tp + fp),
            coverage=ratio(tp, tp + fn),
            f1=ratio(2 * tp, 2 * tp + fp + fn),
            labeled_cases=len(subset),
            unscored_cases=len(cases) - len(subset),
            hit_cases=sum(c["tp"] > 0 for c in subset),
        )
        expected["both90"] = sum(
            c["tp"] > 0
            and 10 * c["tp"] >= 9 * (c["tp"] + c["fp"])
            and 10 * c["tp"] >= 9 * c["reference_fids"]
            for c in subset
        )
        expected["unmapped"] = sum(c["unmapped_fids"] for c in subset)
        actual = summary["methods"][key]
        if any(
            not math.isclose(actual[k], v, rel_tol=1e-12, abs_tol=1e-12)
            for k, v in expected.items()
        ):
            raise ValueError(f"Aggregate mismatch for {key}")
    assets = (
        Path(__file__).resolve().parents[1]
        / "skills"
        / "image-case-retrieval"
        / "assets"
    )
    for name, receipt in summary["public_models"].items():
        folder = assets / name
        for f, pinned in (
            ("model.json", receipt["model"]),
            ("review_recipe.json", receipt["recipe"]),
        ):
            if hashlib.sha256((folder / f).read_bytes()).hexdigest() != pinned:
                raise ValueError(
                    f"Packaged model differs from the public benchmark: {name}/{f}"
                )
        model = json.loads((folder / "model.json").read_text(encoding="utf8"))
        for kind, pinned in model["weights"].items():
            if (
                hashlib.sha256((folder / (kind + ".txt")).read_bytes()).hexdigest()
                != pinned
            ):
                raise ValueError(f"Changed numerical weights: {name}/{kind}")
    print(
        f"Verified {sum(map(len, groups.values()))} case records across {len(groups)} comparisons."
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--directory",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "semantic-ranker",
    )
    verify(cli.parse_args().directory)
