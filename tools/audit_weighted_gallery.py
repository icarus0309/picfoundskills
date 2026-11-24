"""Trace missed reference images from a colleague's local HTML gallery against an independent sealed run."""

import argparse
import collections
import csv
from pathlib import Path, PureWindowsPath
import sys

from bs4 import BeautifulSoup

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills/image-case-retrieval/scripts")
)
import pipeline as p
from score_weighted import label_sets
from weighted_retrieval import sha


def key(path):
    return str(PureWindowsPath(path)).casefold()


def audit(root, gallery, pilot, output):
    inv = p.read_json(pilot / "inventory.json")
    pred = p.read_json(pilot / "predictions.json")
    if (
        sha(pilot / "predictions.json")
        != p.read_json(pilot / "predictions_seal.json")["sha256"]
    ):
        raise ValueError("Local comparison predictions changed")
    bypath = {key(i["path"]): i for i in inv["images"]}
    evidence = {(r["case_id"], r["image_id"]): r["status"] for r in pred["evidence"]}
    names = collections.defaultdict(set)
    mapped = collections.defaultdict(set)
    for r in p.read_json(pilot / "corpus_metadata.json"):
        names[str(r["filename"]).casefold()].add(r["fid"])
        if r["mapping"] in ("size_and_name", "unique_name"):
            for i in r["image_ids"]:
                mapped[i].add(r["fid"])
    soup = BeautifulSoup(gallery.read_text("utf-8-sig"), "html.parser")
    sections = soup.select("section[id]")
    ids = {s["id"] for s in sections}
    cases = {r["id"]: r["query"] for r in inv["cases"] if r["id"] in ids}
    high, low, negative = (
        label_sets(root, cases, h) for h in ("高权重预期fid", "低权重fid", "非预期fid")
    )
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    detail = []
    for section in sections:
        q = section["id"]
        declared = [
            int(b.get_text(strip=True)) for b in section.select_one("p").select("b")
        ]
        if (
            len(declared) != 4
            or declared[0] != len(high[q])
            or declared[0] != declared[1] + declared[2]
        ):
            raise ValueError(f"Workbook and HTML counts disagree for {q}")
        tally = collections.Counter()
        for card in section.select("article"):
            path = card.select_one(".path").get_text(strip=True)
            image = bypath.get(key(path))
            fids = (
                mapped.get(image["id"], set())
                if image
                else names[PureWindowsPath(path).name.casefold()]
            )
            if "找漏" in card.select_one(".id").get_text():
                if not high[q] & fids:
                    raise ValueError(f"Cannot reconcile a missed reference for {q}")
                if not image:
                    stage = "unmapped"
                else:
                    i = image["id"]
                    stage = evidence.get((q, i))
                    if stage is None:
                        state = pred["states"][q].get(i, "S")
                        stage = {
                            "S": "outside_route",
                            "E": "decode",
                            "0": "screen_rejected",
                        }[state]
                category = "miss"
            else:
                if high[q] & fids:
                    raise ValueError(
                        f"Extra overlaps a local high-weight reference for {q}"
                    )
                stage = (
                    "low_weight"
                    if low[q] & fids
                    else "explicit_negative"
                    if negative[q] & fids
                    else "unlisted"
                    if fids
                    else "unknown_mapping"
                )
                category = "extra"
            detail.append(
                [q, cases[q], category, path, image["id"] if image else "", stage]
            )
            tally[stage] += 1
        if len(section.select("article")) != declared[2] + declared[3]:
            raise ValueError(f"HTML detail count disagrees for {q}")
        counts[q] = {
            "expected": declared[0],
            "hits": declared[1],
            "missed": declared[2],
            "extra": declared[3],
            "local_comparison_or_reference_status": dict(tally),
        }
    with (output / "private_gallery_trace.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "query",
                "gallery_status",
                "private_relative_path",
                "local_image",
                "local_trace_or_reference",
            ]
        )
        w.writerows(detail)
    p.write_json(output / "gallery_counts.json", counts)
    return counts


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "gallery", "pilot", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    a = cli.parse_args()
    audit(a.root, a.gallery, a.pilot, a.out)
