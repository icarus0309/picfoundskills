"""Optionally require written evidence for each explicit place in a venue query. Preserve the original ranking."""

import argparse
import csv
import json
from pathlib import Path
import pipeline as p
from lexical_index import norm
from lexical_facts import signature, family_inventory
from weighted_retrieval import sha

FIELDS = (
    "summary",
    "abstract",
    "aiDescription",
    "search_term",
    "keyword",
    "tagIds",
    "ocr",
    "country",
    "province",
    "city",
    "business_terms",
    "place_terms",
)


def applies(plan):
    return plan["kind"] == "venue" and bool(plan["places"])


def proof(doc, plan):
    evidence = {}
    for group in plan["places"]:
        words = [norm(w) for w in group if len(norm(w)) >= 2] or [
            norm(w) for w in group if norm(w)
        ]
        hits = {t: [k for k in FIELDS if t in norm(doc.get(k, ""))] for t in words}
        hits = {t: fields for t, fields in hits.items() if fields}
        if not hits:
            return {}
        evidence.update(hits)
    return evidence


def apply(gallery, plans_file, source, out):
    if source.resolve() == out.resolve():
        raise ValueError("Use a separate output directory")
    prev = p.read_json(source / "seal.json")
    stamp = p.read_json(source / "prediction_protocol.json")
    plans = p.read_json(plans_file)
    if (
        sha(source / "predictions.json") != prev["predictions"]
        or sha(source / "matches.csv") != prev["matches"]
        or signature(stamp) != prev["protocol"]
    ):
        raise ValueError("Parent predictions changed")
    meta = p.read_json(gallery / "gallery.json")
    if (
        meta != stamp["gallery_metadata"]
        or sha(gallery / "weighted_docs.json") != meta["docs_sha256"]
        or family_inventory(p.read_json(gallery / "inventory.json"))
        != meta["inventory"]
    ):
        raise ValueError("Gallery mismatch")
    parent = p.read_json(source / "predictions.json")
    if any(plans[q]["query"] != text for q, text in parent["queries"].items()):
        raise ValueError("Query text mismatch")
    docs = p.read_json(gallery / "weighted_docs.json")
    protocol = dict(
        stamp,
        venue_filter=sha(Path(__file__)),
        parent_predictions=prev["predictions"],
        venue_plans=sha(plans_file),
    )
    out.mkdir(parents=True, exist_ok=True)
    if (out / "prediction_protocol.json").exists() and p.read_json(
        out / "prediction_protocol.json"
    ) != protocol:
        raise ValueError("Different venue filter inputs")
    p.write_json(out / "prediction_protocol.json", protocol)
    approved = {}
    for q, images in parent["ranker"].items():
        if applies(plans[q]):
            good = {i: e for i in images if (e := proof(docs[i], plans[q]))}
            approved[q] = good
            parent["ranker"][q] = list(good)
    with (
        (source / "matches.csv").open(encoding="utf-8-sig", newline="") as a,
        (out / "matches.csv").open("w", encoding="utf-8-sig", newline="") as b,
    ):
        reader = csv.DictReader(a)
        writer = csv.DictWriter(
            b, fieldnames=[*reader.fieldnames, "required_place_evidence"]
        )
        writer.writeheader()
        for row in reader:
            q = row["case_id"]
            i = row["image_id"]
            if q in approved and i not in approved[q]:
                continue
            row["required_place_evidence"] = json.dumps(
                approved.get(q, {}).get(i, {}), ensure_ascii=False
            )
            writer.writerow(row)
    p.write_json(out / "predictions.json", parent)
    p.write_json(
        out / "seal.json",
        {
            "predictions": sha(out / "predictions.json"),
            "matches": sha(out / "matches.csv"),
            "protocol": signature(protocol),
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("gallery", "plans", "source", "out"):
        cli.add_argument("--" + k, type=Path, required=True)
    a = cli.parse_args()
    apply(a.gallery, a.plans, a.source, a.out)
