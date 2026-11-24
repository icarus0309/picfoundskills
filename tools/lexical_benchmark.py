"""Freeze the choice on calibration scores. Export anonymous held-out counts and independently verify them."""

import argparse
import csv
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills/image-case-retrieval/scripts")
)
import pipeline as p
from score_weighted import metrics
from weighted_retrieval import sha

MODES = ("base", "profile", "album", "enriched", "visual-album")
PUBLIC = Path(__file__).resolve().parents[1] / "benchmarks" / "lexical-ranker"


def freeze(run):
    choices = {}
    for mode in MODES:
        path = run / ("fit-" + mode) / "model.json"
        choices[mode] = {
            "decision": p.read_json(path)["decision"],
            "model_sha256": sha(path),
        }

    def key(name):
        a = choices[name]["decision"]
        return min(a["precision"], a["coverage"]), a["f1"]

    result = {
        "chosen": max(MODES, key=key),
        "variants": choices,
        "split_sha256": sha(run / "split.json"),
    }
    path = run / "choice.json"
    if path.exists() and p.read_json(path) != result:
        raise ValueError("Different protocol already frozen")
    if not path.exists() and any(
        (run / ("pred-" + m) / "scores.json").exists() for m in MODES
    ):
        raise ValueError("Scores already opened before freezing the choice")
    p.write_json(path, result)


def aggregate(rows):
    scored = [r for r in rows if int(r["reference_fids"]) > 0]
    return dict(
        **metrics(*(sum(int(r[k]) for r in scored) for k in ("tp", "fp", "fn"))),
        labeled=len(scored),
        unscored=len(rows) - len(scored),
        hit_cases=sum(int(r["tp"]) > 0 for r in scored),
        unmapped=sum(int(r["unmapped_fids"]) for r in scored),
    )


def export(run, out):
    choice = p.read_json(run / "choice.json")
    split = p.read_json(run / "split.json")
    chosen = choice["chosen"]
    if sha(run / "split.json") != choice["split_sha256"]:
        raise ValueError("Changed split")
    totals = {}
    rows = []
    seals = {}
    for m, item in choice["variants"].items():
        pred = run / ("pred-" + m)
        spec = p.read_json(pred / "prediction_protocol.json")
        seal = p.read_json(pred / "seal.json")
        if (
            spec["model"] != item["model_sha256"]
            or spec["queries"] != split["test"]
            or sha(pred / "predictions.json") != seal["predictions"]
        ):
            raise ValueError("Held-out prediction mismatch")
        seals[m] = seal["predictions"]
        scores = p.read_json(pred / "scores.json")
        for method in ("ranker", "literal", "candidates"):
            key = m if method == "ranker" else m + "-" + method
            for q, r in scores[method]["cases"].items():
                rows.append(
                    dict(
                        method=key,
                        case_id=q,
                        **{
                            k: r[k]
                            for k in (
                                "tp",
                                "fp",
                                "fn",
                                "unmapped_fids",
                                "reference_fids",
                            )
                        },
                    )
                )
            totals[key] = aggregate([r for r in rows if r["method"] == key])
    partitions = {
        "first10": {f"TC{i:04d}" for i in range(1, 11)},
        "second10": {f"TC{i:04d}" for i in range(11, 21)},
    }
    partitions["remaining"] = (
        set(split["test"]) - partitions["first10"] - partitions["second10"]
    )
    if not all((partitions[k] <= set(split["test"])) for k in partitions):
        raise ValueError("Historical subset missing")
    stats = {
        k: aggregate([r for r in rows if r["method"] == chosen and r["case_id"] in qs])
        for k, qs in partitions.items()
    }
    facts = p.read_json(run / "gallery" / "facts.json")
    info = dict(
        choice=choice,
        split_counts={
            k: len(split[k]) for k in ("train", "calibration", "test", "purged")
        },
        images=len(p.read_json(run / "gallery" / "inventory.json")["images"]),
        visual_status=dict(Counter(f["status"] for f in facts.values())),
        training={
            m: p.read_json(run / ("fit-" + m) / "calibration.json")["training"]
            for m in MODES
        },
        sealed_predictions=seals,
        results=totals,
        subsets=stats,
    )
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (out / "summary.json").write_bytes(
        (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode()
    )
    verify(out)


def verify(out):
    info = p.read_json(out / "summary.json")
    with (out / "cases.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for k, t in info["results"].items():
        selected = [r for r in rows if r["method"] == k]
        if (
            len({r["case_id"] for r in selected}) != info["split_counts"]["test"]
            or len(selected) != info["split_counts"]["test"]
            or any(
                int(r["tp"]) + int(r["fn"]) != int(r["reference_fids"])
                for r in selected
            )
            or aggregate(selected) != t
        ):
            raise ValueError("Invalid per-query counts for " + k)
    variants = info["choice"]["variants"]
    chosen = max(
        variants,
        key=lambda k: (
            min(variants[k]["decision"][n] for n in ("precision", "coverage")),
            variants[k]["decision"]["f1"],
        ),
    )
    if chosen != info["choice"]["chosen"]:
        raise ValueError("Selection does not match calibration")
    chosen_rows = [r for r in rows if r["method"] == chosen]
    for group, counts in info["subsets"].items():

        def bucket(q):
            i = int(q[2:])
            return "first10" if i <= 10 else "second10" if i <= 20 else "remaining"

        if (
            aggregate([r for r in chosen_rows if bucket(r["case_id"]) == group])
            != counts
        ):
            raise ValueError("Historical subset counts do not reconcile")
    diagnostic = out / "diagnostic.json"
    if diagnostic.exists():
        diag = p.read_json(diagnostic)
        tot = info["results"][chosen]
        if (
            sum(diag["extra_categories"].values()) != tot["fp"]
            or sum(
                row["missed_before_rank"] + row["missed_at_rank"]
                for row in diag["errors"].values()
            )
            != tot["fn"]
        ):
            raise ValueError("Failure-stage totals do not reconcile")
    if (out / "venue.json").exists():
        supplemental = p.read_json(out / "venue.json")
        for key in ("additional_with_rule", "additional_without_rule"):
            case = supplemental[key]
            check = aggregate(list(case["cases"].values()))
            if (
                any(
                    case["total"][k] != check[k]
                    for k in ("tp", "fp", "fn", "precision", "coverage", "f1")
                )
                or len(case["cases"]) != supplemental["new_split"]["test"]
            ):
                raise ValueError("Supplementary venue counts do not reconcile")
    if out.resolve() == PUBLIC.resolve():
        assets = PUBLIC.parents[1] / "skills/image-case-retrieval/assets/lexical-ranker"
        if (assets / "model.json").exists():
            if sha(assets / "model.json") != variants[chosen]["model_sha256"]:
                raise ValueError("Packaged model differs from selected version")
            if any(
                sha(assets / (k + ".txt")) != v
                for k, v in p.read_json(assets / "model.json")["weights"].items()
            ):
                raise ValueError("Packaged weights have changed")
    print("Verified", len(rows), "method/query rows. Calibration winner:", chosen)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("freeze", "export", "verify"))
    cli.add_argument("--run", type=Path)
    cli.add_argument("--out", type=Path, default=PUBLIC)
    a = cli.parse_args()
    if a.command == "freeze":
        freeze(a.run)
    elif a.command == "export":
        export(a.run, a.out)
    else:
        verify(a.out)
