"""Re-evaluate sealed visual review results and their cached responses locally, with no network or provider credentials."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pipeline as p
from boundary_review import (
    RUBRIC,
    parse_grades,
    read_parent,
    request_text,
    response_digest,
)
from lexical_facts import signature
from weighted_retrieval import sha


def replay(gallery, source, review_dir, out, legacy_endpoint=None):
    parent, proto, parent_seal = read_parent(source)
    review = p.read_json(review_dir / "review_protocol.json")
    old_seal = p.read_json(review_dir / "review_seal.json")
    if (
        review["rubric"] != RUBRIC
        or review["parent"] != parent_seal
        or old_seal
        != {
            "data": sha(review_dir / "reviews.json"),
            "protocol": sha(review_dir / "review_protocol.json"),
        }
        or p.read_json(gallery / "gallery.json") != proto["gallery_metadata"]
        or sha(gallery / "weighted_docs.json")
        != proto["gallery_metadata"]["docs_sha256"]
    ):
        raise ValueError("Review, gallery or predictions changed")
    if review.get("raw_responses_sha256") and review[
        "raw_responses_sha256"
    ] != response_digest(review_dir):
        raise ValueError("Sealed provider response cache changed")
    endpoint = review["endpoint"]
    if legacy_endpoint is not None:
        if hashlib.sha256(legacy_endpoint.encode()).hexdigest() != endpoint:
            raise ValueError("Legacy endpoint does not match the original review")
        endpoint = hashlib.sha256(legacy_endpoint.rstrip("/").encode()).hexdigest()
    first = next(
        (review_dir / "responses" / "cache").glob("qwen-boundary-v1-*.json"), None
    )
    if first is not None and p.read_json(first)["endpoint_sha256"] != endpoint:
        raise ValueError(
            "Cached endpoint differs. For older protocols supply the exact --legacy-endpoint-url"
        )
    docs = p.read_json(gallery / "weighted_docs.json")
    evidence = p.read_json(review_dir / "reviews.json")
    groups = defaultdict(dict)
    for q, items in evidence.items():
        for item in items:
            groups[item["image"]][q] = item
    files = {}
    counts = dict(known=0, recovered=0, missing=0)
    for image, by_query in groups.items():
        queries = list(by_query)
        thumb = review_dir / "thumbs" / (image + "-1200.jpg")
        image_hash = sha(thumb) if thumb.is_file() else None
        for begin in range(0, len(queries), 8):
            batch = queries[begin : begin + 8]
            ask = {str(k): parent["queries"][q] for k, q in enumerate(batch)}
            grades = {}
            if image_hash and image not in review["excluded"]:
                key = [
                    "provider-json-v2",
                    "qwen",
                    review["model"],
                    endpoint,
                    "boundary-v1",
                    request_text(docs[image], ask),
                    [image_hash],
                ]
                digest = hashlib.sha256(
                    json.dumps(key, ensure_ascii=False).encode()
                ).hexdigest()
                cache = (
                    review_dir
                    / "responses"
                    / "cache"
                    / ("qwen-boundary-v1-" + digest + ".json")
                )
                if cache.exists():
                    data = p.read_json(cache)
                    if (
                        data["input_image_hashes"] != [image_hash]
                        or data["requested_model"] != review["model"]
                        or data["endpoint_sha256"] != endpoint
                        or data["stage"] != "boundary-v1"
                    ):
                        raise ValueError("Provider response identity changed")
                    files[digest] = sha(cache)
                    grades = parse_grades(data["result"], ask)
            for k, q in enumerate(batch):
                old = by_query[q]["grade"]
                val = grades.get(str(k))
                if old is not None and val != old:
                    raise ValueError(
                        "Original visual decision could not be reproduced from its raw response"
                    )
                counts[
                    "known"
                    if old is not None
                    else "recovered"
                    if val is not None
                    else "missing"
                ] += 1
                by_query[q]["grade"] = val
    out.mkdir(exist_ok=True, parents=True)
    lock = out / "review_protocol.json"
    record = dict(
        review,
        grade_parser=2,
        replay_of=old_seal,
        cache_sha256=signature(sorted(files.items())),
    )
    if lock.exists() and p.read_json(lock) != record:
        raise ValueError("Replay inputs changed")
    p.write_json(lock, record)
    p.write_json(out / "reviews.json", evidence)
    p.write_json(
        out / "review_seal.json",
        dict(data=sha(out / "reviews.json"), protocol=sha(lock)),
    )
    p.write_json(out / "recovery.json", counts)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("gallery", "source", "review-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--legacy-endpoint-url",
        help="Original URL as entered in an older configuration",
    )
    a = parser.parse_args()
    print(replay(a.gallery, a.source, a.review_dir, a.out, a.legacy_endpoint_url))
