"""Apply photo grades gathered by a scout ranker to another ranker's candidates in the same gallery."""

import argparse
from pathlib import Path

import pipeline as p
from boundary_review import apply, read_parent
from lexical_facts import signature
from weighted_retrieval import sha


def transfer(gallery, scout, target, reviews, out, depth, drop, add):
    original, original_proto, scout_seal = read_parent(scout)
    other, other_proto, target_seal = read_parent(target)
    meta = p.read_json(gallery / "gallery.json")
    if (
        original["queries"] != other["queries"]
        or meta != other_proto["gallery_metadata"]
        or meta != original_proto["gallery_metadata"]
    ):
        raise ValueError("Scout and target require identical gallery and queries")
    review = p.read_json(reviews / "review_protocol.json")
    cert = dict(
        data=sha(reviews / "reviews.json"),
        protocol=sha(reviews / "review_protocol.json"),
    )
    if (
        cert != p.read_json(reviews / "review_seal.json")
        or review["parent"] != scout_seal
        or not 1 <= depth <= review["depth"]
    ):
        raise ValueError("The votes must belong to the scout's sealed run")
    scoped = out / "scout-evidence"
    scoped.mkdir(parents=True, exist_ok=True)
    evidence = p.read_json(reviews / "reviews.json")
    narrowed = {}
    for q, items in evidence.items():
        allowed = set(other["candidates"][q])
        # Relative offsets always refer to the scout boundary. A newly admitted photo
        # is listed after the target's original selections, which retain their own scores.
        narrowed[q] = [
            dict(v, score=None, scout_score=v["score"])
            for v in items
            if v["image"] in allowed
        ]
    stamp = dict(
        review,
        parent=target_seal,
        scout=scout_seal,
        original_review=cert,
        operation="transfer-v1",
        candidates=signature(other["candidates"]),
    )
    lock = scoped / "review_protocol.json"
    if lock.exists() and p.read_json(lock) != stamp:
        raise ValueError("Use a fresh directory for different feedback")
    p.write_json(lock, stamp)
    p.write_json(scoped / "reviews.json", narrowed)
    p.write_json(
        scoped / "review_seal.json",
        dict(data=sha(scoped / "reviews.json"), protocol=sha(lock)),
    )
    apply(gallery, target, scoped, out / "result", drop, add, depth)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("gallery", "scout", "target", "reviews", "out"):
        cli.add_argument("--" + k, type=Path, required=True)
    for k in ("depth", "drop", "add"):
        cli.add_argument("--" + k, type=int, required=True)
    a = cli.parse_args()
    transfer(a.gallery, a.scout, a.target, a.reviews, a.out, a.depth, a.drop, a.add)
