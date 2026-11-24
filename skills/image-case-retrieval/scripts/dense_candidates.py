"""Propose gallery-wide images from caption and pixel indexes without consulting directory plans."""

import argparse
from pathlib import Path

import numpy as np
import pipeline as p
from lexical_facts import family_inventory, signature
from semantic_features import FieldScores, PixelScores
from weighted_retrieval import sha


def top_ids(values, available, count, ids):
    eligible = np.flatnonzero(available)
    order = np.argsort(-values[eligible], kind="stable")[:count]
    return [ids[i] for i in eligible[order]]


def propose(a):
    if a.top < 1:
        raise ValueError("The per-channel limit must be positive")
    meta = p.read_json(a.gallery / "gallery.json")
    inv = p.read_json(a.gallery / "inventory.json")
    if (
        family_inventory(inv) != meta["inventory"]
        or sha(a.gallery / "weighted_docs.json") != meta["docs_sha256"]
    ):
        raise ValueError("Gallery inputs changed")
    ids = sorted(v["id"] for v in inv["images"])
    text = FieldScores(a.text_index, a.text_model, ids, a.gallery, a.threads)
    pix = PixelScores(a.clip_index, a.clip_model, ids, a.gallery, a.threads)
    qs = p.read_json(a.queries)
    if (
        not isinstance(qs, dict)
        or not qs
        or not all(isinstance(q, str) and q.strip() for q in qs.values())
    ):
        raise ValueError("Expected a mapping of case IDs to nonempty search queries")
    stamp = dict(
        version=1,
        gallery=meta,
        query_hash=sha(a.queries),
        per_channel=a.top,
        text=text.index_sha,
        pixel=pix.index_sha,
    )
    lock = a.out / "protocol.json"
    a.out.mkdir(parents=True, exist_ok=True)
    if lock.exists() and p.read_json(lock) != stamp:
        raise ValueError("Use separate output for different candidate inputs")
    p.write_json(lock, stamp)
    result = {}
    for q, query in qs.items():
        vector = text.encoder.encode(
            ["为这个句子生成表示以用于检索相关文章：" + query]
        )[0]
        similarity = text.vec @ vector
        image_scores = pix.photos @ pix.encoder.texts([query])[0]
        result[q] = {
            field: top_ids(similarity[lookup], lookup != 0, a.top, ids)
            for field, lookup in (("title", text.title), ("abstract", text.abstract))
        }
        result[q]["pixel"] = top_ids(image_scores, pix.valid, a.top, ids)
    target = a.out / "proposals.json"
    p.write_json(target, result)
    p.write_json(
        a.out / "seal.json", dict(predictions=sha(target), protocol=signature(stamp))
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in (
        "gallery",
        "text-index",
        "text-model",
        "clip-index",
        "clip-model",
        "queries",
        "out",
    ):
        cli.add_argument("--" + k, type=Path, required=True)
    cli.add_argument("--top", type=int, default=50)
    cli.add_argument("--threads", type=int, default=3)
    propose(cli.parse_args())
