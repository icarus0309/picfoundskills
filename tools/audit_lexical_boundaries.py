"""Post-evaluation audit of a fixed ranker. Emit a private blind review sheet and anonymous aggregates."""

import argparse
import collections
import csv
import random
import sys
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills/image-case-retrieval/scripts")
)
import numpy as np
import lexical_ranker as lr
import pipeline as p
from score_weighted import label_sets, file_ids
from weighted_retrieval import sha


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        w = csv.writer(file)
        w.writerow(header)
        w.writerows(rows)


def audit(root, run, out):
    choice = p.read_json(run / "choice.json")
    mode = choice["chosen"]
    model = run / ("fit-" + mode)
    pred_dir = run / ("pred-" + mode)
    g = run / "gallery"
    if sha(model / "model.json") != choice["variants"][mode]["model_sha256"]:
        raise ValueError("Changed chosen model")
    seal = p.read_json(pred_dir / "seal.json")
    if sha(pred_dir / "predictions.json") != seal["predictions"]:
        raise ValueError("Changed predictions")
    pred = p.read_json(pred_dir / "predictions.json")
    sql = pred["queries"]
    ref, low, neg = [label_sets(root, sql, header) for header in lr.HEADERS]
    inv = p.read_json(g / "inventory.json")
    mapping = file_ids(root, inv)
    paths = {i["id"]: i["path"] for i in inv["images"]}
    m, ids = lr.read_features(run / mode)
    spec = p.read_json(model / "model.json")
    cfg = spec["decision"]
    models = []
    for k in (*lr.KINDS, "count"):
        f = model / (k + ".txt")
        if sha(f) != spec["weights"][k]:
            raise ValueError("Changed ranker weights")
        models.append(lr.lgb.Booster(model_str=f.read_text(encoding="utf-8")))
    vals = []
    truth = []
    total = 0
    categories = collections.Counter()
    review = []
    errors = {}
    rng = random.Random(20261124)
    for q in sql:
        if not ref[q]:
            continue
        expected = {mapping[f] for f in ref[q] if f in mapping}
        total += len(ref[q])
        predicted = set(pred["ranker"][q])
        candidates = set(pred["candidates"][q])
        extra = predicted - expected
        missed = expected - predicted
        known_low = {mapping[f] for f in low[q] if f in mapping}
        known_neg = {mapping[f] for f in neg[q] if f in mapping}
        for i in sorted(extra):
            categories[
                "low"
                if i in known_low
                else "explicit_negative"
                if i in known_neg
                else "unlisted"
            ] += 1
        errors[q] = {
            "extra": len(extra),
            "missed_before_rank": len(ref[q]) - len(expected & candidates),
            "missed_at_rank": len(missed & candidates),
        }
        ix, x = lr.load_block(run / mode, m, q)
        both = lr.probabilities(models, x, cfg["iterations"])
        scores = lr.relative_score(
            both,
            cfg["pairwise_weight"],
            models[2].predict([lr.count_features(x)], num_threads=1)[0],
            cfg["count_adjustment"],
        )
        images = [ids[i] for i in ix]
        if set(i for i, v in zip(images, scores) if v >= cfg["threshold"]) != predicted:
            raise ValueError("Replayed score differs")
        vals.extend(scores)
        truth.extend(i in expected for i in images)
        if extra and missed:
            for category, pool in (("extra-F", extra), ("missing-F", missed)):
                review.append((q, sql[q], rng.choice(sorted(pool)), category))
    scores = np.asarray(vals)
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
    tp = np.cumsum(np.array(truth)[order])[ends]
    n = ends + 1
    prec = tp / n
    rec = tp / total
    outcomes = {
        "max_coverage_at_90_precision": float(rec[prec >= 0.9].max())
        if any(prec >= 0.9)
        else None,
        "max_precision_at_90_coverage": float(prec[rec >= 0.9].max())
        if any(rec >= 0.9)
        else None,
    }
    out.mkdir(parents=True, exist_ok=True)
    rng.shuffle(review)
    blind = []
    key = []
    for j, (q, text, i, category) in enumerate(review, 1):
        ticket = f"R{j:04}"
        blind.append([ticket, text, str(root / "图片" / paths[i]), "", ""])
        key.append([ticket, q, i, category])
    write_csv(
        out / "review.csv",
        ["ticket", "query", "local_path", "verdict", "evidence"],
        blind,
    )
    write_csv(
        out / "review_key.csv",
        ["ticket", "case_id", "image_id", "original_status"],
        key,
    )
    p.write_json(
        out / "diagnostic.json",
        dict(
            mode=mode,
            extra_categories=dict(categories),
            threshold_sweep=outcomes,
            errors=errors,
            review_rows=len(blind),
            predictions_sha256=seal["predictions"],
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "run", "out"):
        cli.add_argument("--" + k, type=Path, required=True)
    a = cli.parse_args()
    audit(a.root, a.run, a.out)
