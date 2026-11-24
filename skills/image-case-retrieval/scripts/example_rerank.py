"""Propagate graded examples to similar photo captions, preserving explicit visual decisions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pipeline as p
import lexical_ranker as lr
from boundary_review import read_parent
from lexical_facts import signature
from weighted_retrieval import sha


def caption_views(index, gallery, ids):
    meta = p.read_json(gallery / "gallery.json")
    stamp = p.read_json(index / "text_protocol.json")
    checksum = sha(index / "text.npz")
    if (stamp["docs"] != meta["docs_sha256"] or sha(gallery / "weighted_docs.json") != stamp["docs"]
            or checksum != p.read_json(index / "text_complete.json")["sha256"]):
        raise ValueError("Caption vectors belong to a different gallery")
    with np.load(index / "text.npz", allow_pickle=False) as data:
        if data["ids"].tolist() != ids:
            raise ValueError("Caption vector IDs differ")
        vectors = data["vectors"]
        result = []
        for field in ("title", "abstract"):
            rows = vectors[data[field]]
            present = data[field] != 0
            if present.any():
                rows = rows - rows[present].mean(axis=0)
            rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-8)
            rows[~present] = 0
            result.append(rows)
    return result, checksum


def corroboration(views, rows, seeds, grades):
    """Locally compare positive and negative examples after centering each caption field."""
    positive = np.asarray(grades) >= 3
    if not positive.any() or positive.all():
        return np.zeros(len(rows), np.float32)
    similarity = sum(v[rows] @ v[seeds].T for v in views) / len(views)
    # The nearest supporting and contradicting examples dominate. A distant match carries little weight.
    best_pos = similarity[:, positive].max(axis=1)
    best_neg = similarity[:, ~positive].max(axis=1)
    confidence = np.clip((np.maximum(best_pos, best_neg) - .15) / .45, 0, 1)
    return 4 * (best_pos - best_neg) * confidence


def predict(gallery, features, model, source, reviews, index, out, strength, depth, drop, add):
    if not np.isfinite(strength) or strength < 0 or depth < 1 or not 0 <= drop < add <= 5:
        raise ValueError("Invalid feedback configuration")
    parent, proto, seal = read_parent(source)
    evidence = p.read_json(reviews / "reviews.json")
    review = p.read_json(reviews / "review_protocol.json")
    check = dict(data=sha(reviews / "reviews.json"), protocol=sha(reviews / "review_protocol.json"))
    if (review["parent"] != seal or review["depth"] < depth or p.read_json(reviews / "review_seal.json") != check
            or sha(features / "manifest.json") != proto["features"] or sha(model / "model.json") != proto["model"]
            or p.read_json(gallery / "gallery.json") != proto["gallery_metadata"]):
        raise ValueError("Model, features, gallery or review differs from sealed prediction")
    manifest, ids = lr.read_features(features)
    inv, _ = lr.check_gallery(gallery, manifest["protocol"])
    paths = {r["id"]: r["path"] for r in inv["images"]}
    views, checksum = caption_views(index, gallery, ids)
    pos = {v: n for n, v in enumerate(ids)}
    spec = p.read_json(model / "model.json")
    cfg = spec["decision"]
    models = []
    for k in (*lr.KINDS, "count"):
        path = model / (k + ".txt")
        if sha(path) != spec["weights"][k]:
            raise ValueError("Model weights changed")
        models.append(lr.lgb.Booster(model_str=path.read_text(encoding="utf8")))
    out.mkdir(parents=True, exist_ok=True)
    stamp = dict(proto, example_feedback=dict(parent=seal, review=check, text_index=checksum,
                 strength=strength, depth=depth, drop=drop, add=add, version=1))
    lock = out / "prediction_protocol.json"
    if lock.exists() and p.read_json(lock) != stamp:
        raise ValueError("Different run already occupies output")
    p.write_json(lock, stamp)
    with (source / "matches.csv").open(encoding="utf-8-sig", newline="") as f:
        prior = {(v["case_id"], v["image_id"]): v["evidence"] for v in csv.DictReader(f)}
    selected = {}
    with (out / "matches.csv").open("w", encoding="utf-8-sig", newline="") as out_file:
        writer = csv.writer(out_file)
        writer.writerow(("case_id", "query", "image_id", "relative_path", "score", "evidence"))
        for q, query in parent["queries"].items():
            doc, x = lr.load_block(features, manifest, q)
            prob = lr.probabilities(models, x, cfg["iterations"])
            count = models[-1].predict([lr.count_features(x)], num_threads=1)[0]
            scores = lr.relative_score(prob, cfg["pairwise_weight"], count, cfg["count_adjustment"])
            retained = scores >= cfg["threshold"]
            if {ids[j] for j in doc[retained]} != set(parent["ranker"][q]):
                raise ValueError("Incompatible model decision")
            reviewed = {v["image"]: v["grade"] for v in evidence[q] if -depth <= v["offset"] < depth}
            known = {i: grade for i, grade in reviewed.items() if grade is not None}
            signal = corroboration(views, doc, [pos[i] for i in known], list(known.values()))
            adjusted = scores + strength * signal
            returned = adjusted >= cfg["threshold"]
            for j, n in enumerate(doc):
                image_id = ids[n]
                if image_id in reviewed:
                    grade = reviewed[image_id]
                    returned[j] = retained[j] if grade is None or drop < grade < add else grade >= add
            selected[q] = []
            for j in np.argsort(-adjusted, kind="stable"):
                if returned[j]:
                    i = ids[doc[j]]
                    selected[q].append(i)
                    details = json.loads(prior.get((q, i), "{}"))
                    details["caption_example_support"] = round(float(signal[j]), 4)
                    if i in reviewed:
                        details["visual_grade"] = reviewed[i]
                    writer.writerow((q, query, i, paths[i], round(float(adjusted[j]), 6),
                                     json.dumps(details, ensure_ascii=False)))
    p.write_json(out / "predictions.json", dict(parent, ranker=selected))
    p.write_json(out / "seal.json", dict(predictions=sha(out / "predictions.json"),
                                         matches=sha(out / "matches.csv"), protocol=signature(stamp)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("gallery", "features", "model", "source", "reviews", "index", "out"):
        cli.add_argument("--" + k, type=Path, required=True)
    for k in ("depth", "add", "drop"):
        cli.add_argument("--" + k, type=int, required=True)
    cli.add_argument("--strength", type=float, required=True)
    a = cli.parse_args()
    predict(a.gallery, a.features, a.model, a.source, a.reviews, a.index, a.out,
            a.strength, a.depth, a.drop, a.add)
