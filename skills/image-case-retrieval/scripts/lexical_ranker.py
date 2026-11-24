"""Train a grouped relevance ranker; apply a frozen model to new queries without reading any labels."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb

import pipeline as p
from lexical_features import load_block
from score_weighted import file_ids, label_sets, measure
from weighted_retrieval import sha
from lexical_facts import signature, family_inventory
from lexical_index import FIELDS, doc_fields, norm
from weighted_retrieval import INDEX

HEADERS = ("高权重预期fid", "低权重fid", "非预期fid")
KINDS = ("binary", "pairwise")
ITERATIONS = (70, 150)
MIX = (0.0, 0.5, 1.0)


def read_features(directory):
    m = p.read_json(directory / "manifest.json")
    if sha(directory / "ids.json") != m["ids_hash"]:
        raise ValueError("Image IDs changed")
    return m, p.read_json(directory / "ids.json")


def check_gallery(gallery, protocol):
    inv = p.read_json(gallery / "inventory.json")
    meta = p.read_json(gallery / "gallery.json")
    if (
        family_inventory(inv) != meta["inventory"]
        or meta["docs_sha256"] != sha(gallery / "weighted_docs.json")
        or meta["docs_sha256"] != protocol["gallery"]
        or meta["inventory"] != protocol.get("inventory")
        or meta["index"] != protocol.get("source_index")
    ):
        raise ValueError("Gallery identity mismatch")
    return inv, meta


def fit_rows(directory, manifest, qids, labels, mapping, ids, mine_semantic=False):
    rows = []
    grades = []
    weights = []
    groups = []
    stats = dict(high=0, low=0, negative=0, unlisted=0, train_queries=0)
    columns = manifest["columns"]
    for q in qids:
        if not labels[0][q]:
            continue
        mapped = [{mapping[f] for f in ref[q] if f in mapping} for ref in labels]
        doc, x = load_block(directory, manifest, q)
        cat = np.array(
            [
                3
                if ids[j] in mapped[0]
                else 1
                if ids[j] in mapped[1]
                else 0
                if ids[j] in mapped[2]
                else -1
                for j in doc
            ],
            np.int8,
        )
        if not np.any(cat == 3) or np.all(cat == 3):
            continue
        unknown = np.flatnonzero(cat < 0)
        high = int((cat == 3).sum())
        limit = max(80, min(600, high * 4))
        order = (
            2 * x[:, columns.index("head_title")]
            + x[:, columns.index("alias_title")]
            + x[:, columns.index("object_anyfield_fraction")]
            + x[:, columns.index("instance_title")]
            + 0.2 * x[:, columns.index("bm25")]
        )
        if mine_semantic:
            # Mine unlisted photos with strong independent text or image evidence.
            # Reciprocal ranks make the channels comparable across queries.
            channels = [order] + [
                x[:, columns.index(c)]
                for c in (
                    "semantic_query_abstract",
                    "semantic_query_strongest_tag",
                    "pixel_query",
                )
            ]
            fused = np.zeros(len(unknown))
            for channel in channels:
                values = channel[unknown]
                if len(values):
                    # Equal scores receive equal ranks, including empty evidence.
                    ranks = len(values) - np.searchsorted(
                        np.sort(values), values, side="right"
                    )
                    fused += 1 / (20 + ranks)
            order = order.copy()
            order[unknown] = fused
        hard = unknown[np.argsort(-order[unknown], kind="stable")[:limit]]
        rest = np.setdiff1d(unknown, hard)
        rng = np.random.default_rng(
            int(hashlib.sha256(q.encode()).hexdigest()[:12], 16)
        )
        sample = rng.choice(rest, size=min(len(rest), limit // 2), replace=False)
        keep = np.sort(np.r_[np.flatnonzero(cat >= 0), hard, sample])
        c = cat[keep]
        rows.append(x[keep])
        grades.append(np.maximum(c, 0))
        weights.append(
            np.select([c == 3, c == 1, c == 0], [1.0, 1.0, 1.0], default=0.35)
        )
        groups.append(len(keep))
        stats["train_queries"] += 1
        for name, v in (("high", 3), ("low", 1), ("negative", 0), ("unlisted", -1)):
            stats[name] += int((c == v).sum())
    if not rows:
        raise ValueError(
            "No training queries have both reachable high-weight and contrast images"
        )
    return (
        np.vstack(rows),
        np.concatenate(grades),
        np.concatenate(weights),
        groups,
        stats,
    )


def probabilities(models, x, iterations):
    if not len(x):
        return np.empty((0, 2))
    binary = models[0].predict(x, num_iteration=iterations, num_threads=2)
    pair = models[1].predict(x, num_iteration=iterations, num_threads=2)
    return np.column_stack((binary, 1 / (1 + np.exp(-np.clip(pair, -25, 25)))))


def threshold(scores, is_hit, expected):
    if not len(scores) or not expected:
        raise ValueError("Calibration requires labeled queries and candidates")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(order) - 1]
    tp = np.cumsum(is_hit[order])[ends]
    returned = ends + 1
    prec = tp / returned
    cov = tp / expected
    f1 = 2 * tp / (returned + expected)
    key = np.minimum(prec, cov)
    best = np.lexsort((-returned, f1, key))[-1]
    return {
        "threshold": float(ordered[ends[best]]),
        "precision": float(prec[best]),
        "coverage": float(cov[best]),
        "f1": float(f1[best]),
        "tp": int(tp[best]),
        "fp": int(returned[best] - tp[best]),
        "fn": int(expected - tp[best]),
    }


def configkey(v):
    return min(v["precision"], v["coverage"]), v["f1"]


def count_features(x):
    # Estimate demand from the evidence distribution on this gallery without image or word identities.
    if len(x) == 0:
        return np.zeros(1 + 3 * x.shape[1], np.float32)
    return np.r_[
        np.log1p(len(x)),
        np.log1p(np.count_nonzero(x, axis=0)),
        x.mean(axis=0),
        x.max(axis=0),
    ]


def count_model(directory, manifest, labels):
    rows = []
    sizes = []
    for q, refs in labels.items():
        if refs:
            _, features = load_block(directory, manifest, q)
            rows.append(count_features(features))
            sizes.append(np.log1p(len(refs)))
    data = lgb.Dataset(np.vstack(rows), label=sizes)
    return lgb.train(
        dict(
            objective="regression_l1",
            num_leaves=7,
            min_data_in_leaf=20,
            learning_rate=0.05,
            verbosity=-1,
            num_threads=4,
            deterministic=True,
            force_col_wise=True,
            seed=418,
        ),
        data,
        num_boost_round=65,
    )


def relative_score(both, mix, count, adjustment):
    if not len(both):
        return np.empty(0)
    prob = np.clip((1 - mix) * both[:, 0] + mix * both[:, 1], 1e-10, 1 - 1e-10)
    logits = np.log(prob / (1 - prob))
    k = min(len(prob), max(1, int(round(np.expm1(np.clip(count, 0, 12))))))
    anchor = np.partition(logits, len(logits) - k)[len(logits) - k]
    return logits - adjustment * anchor


def train(args):
    dest = args.out
    dest.mkdir(exist_ok=True, parents=True)
    m, ids = read_features(args.features)
    plans = p.read_json(args.plans)
    split = p.read_json(args.split)
    if m["protocol"]["plans"] != sha(args.plans) or split["plans_sha256"] != sha(
        args.plans
    ):
        raise ValueError("Split, plans and features do not match")
    if "projection_index" in m["protocol"]:
        proof = args.features / "projection_mode.json"
        record = p.read_json(proof) if proof.is_file() else {}
        if (
            not record.get("cross_fit")
            or record.get("split") != sha(args.split)
            or record.get("manifest") != sha(args.features / "manifest.json")
        ):
            raise ValueError(
                "Projection training requires cross-fitted features for this split"
            )
    train_ids = split["train"]
    cal_ids = split["calibration"]
    if set(train_ids) & set(cal_ids) or set(split["test"]) & (
        set(train_ids) | set(cal_ids)
    ):
        raise ValueError("Overlapping split")
    workbook = (
        getattr(args, "reference_workbook", None)
        or args.root / "E2E_1321cases_1w_data_v19.xlsx"
    )
    ident = {
        "split": sha(args.split),
        "features": sha(args.features / "manifest.json"),
        "workbook": sha(workbook),
        "training_code": sha(Path(__file__)),
        "mine_semantic": getattr(args, "mine_semantic", False),
    }
    lock = dest / "training_inputs.json"
    if lock.exists() and p.read_json(lock) != ident:
        raise ValueError("Changed fit inputs; use a fresh output directory")
    p.write_json(lock, ident)
    inv, meta = check_gallery(args.gallery, m["protocol"])
    if sha(args.root / INDEX) != meta["index"]:
        raise ValueError("Source image index changed")
    mapping = file_ids(args.root, inv)
    training = {q: plans[q]["query"] for q in train_ids}
    labels = [label_sets(args.root, training, header, workbook) for header in HEADERS]
    x, y, w, groups, stats = fit_rows(
        args.features,
        m,
        train_ids,
        labels,
        mapping,
        ids,
        getattr(args, "mine_semantic", False),
    )
    models = []
    print("Training observations:", stats, flush=True)
    for name in KINDS:
        rank = name == "pairwise"
        data = lgb.Dataset(
            x,
            label=y if rank else (y == 3).astype(int),
            weight=w,
            group=groups if rank else None,
            feature_name=m["columns"],
        )
        params = dict(
            objective="lambdarank" if rank else "binary",
            num_leaves=15,
            learning_rate=0.06,
            min_data_in_leaf=45,
            lambda_l2=6,
            deterministic=True,
            force_col_wise=True,
            num_threads=4,
            verbosity=-1,
            seed=307,
        )
        if rank:
            params.update(label_gain=[0, 1, 3, 10], lambdarank_truncation_level=100)
        model = lgb.train(params, data, num_boost_round=max(ITERATIONS))
        (dest / (name + ".txt")).write_bytes(model.model_to_string().encode("utf-8"))
        models.append(model)
    volume = count_model(args.features, m, labels[0])
    (dest / "count.txt").write_bytes(volume.model_to_string().encode("utf-8"))
    validation = {q: plans[q]["query"] for q in cal_ids}
    ref = label_sets(args.root, validation, workbook=workbook)
    expected = sum(map(len, ref.values()))
    decision = []
    for nt in ITERATIONS:
        preds = []
        hits = []
        counts = []
        for q in cal_ids:
            if not ref[q]:
                continue
            j, xx = load_block(args.features, m, q)
            preds.append(probabilities(models, xx, nt))
            counts.append(volume.predict([count_features(xx)], num_threads=1)[0])
            correct = {mapping[f] for f in ref[q] if f in mapping}
            hits.extend(ids[i] in correct for i in j)
        hit = np.array(hits, dtype=bool)
        for mix in MIX:
            for adjustment in (0, 0.5, 1):
                scores = np.concatenate(
                    [
                        relative_score(pair, mix, c, adjustment)
                        for pair, c in zip(preds, counts)
                    ]
                )
                decision.append(
                    dict(
                        iterations=nt,
                        pairwise_weight=mix,
                        count_adjustment=adjustment,
                        **threshold(scores, hit, expected),
                    )
                )
    winner = max(decision, key=configkey)
    weights = {n: sha(dest / (n + ".txt")) for n in (*KINDS, "count")}
    p.write_json(
        dest / "model.json",
        {
            "version": 1,
            "columns": m["columns"],
            "decision": winner,
            "weights": weights,
            **(
                {"feedback_index": m["protocol"]["feedback_index"]}
                if "feedback_index" in m["protocol"]
                else {}
            ),
            "feature_encoder": m["protocol"]["encoder"],
            "visual_enrichment": bool(m["protocol"]["facts"]),
            "contrast": bool(m["protocol"].get("contrast")),
        },
    )
    p.write_json(
        dest / "calibration.json",
        {
            "training": stats,
            "train_cases": len(train_ids),
            "calibration_cases": len(cal_ids),
            "labeled_calibration": sum(bool(v) for v in ref.values()),
            "grid": decision,
            "chosen": winner,
        },
    )
    print("Calibration:", winner, flush=True)


def predict(args):
    m, ids = read_features(args.features)
    spec = p.read_json(args.model / "model.json")
    if spec.get("feedback_index") != m["protocol"].get("feedback_index"):
        raise ValueError("Private feedback bank differs from the fitted model")
    if (
        spec["columns"] != m["columns"]
        or spec["feature_encoder"] != m["protocol"]["encoder"]
    ):
        raise ValueError("Feature schema or encoder differs from training")
    if bool(m["protocol"]["facts"]) != spec["visual_enrichment"]:
        raise ValueError("Use matching gallery enrichment mode")
    plans = p.read_json(args.plans)
    if sha(args.plans) != m["protocol"]["plans"]:
        raise ValueError("Mismatch between queries and feature blocks")
    queries = p.read_json(args.split)["test"] if args.split else list(plans)
    if args.split and p.read_json(args.split)["plans_sha256"] != sha(args.plans):
        raise ValueError("Changed test scope")
    inv, meta = check_gallery(args.gallery, m["protocol"])
    if bool(m["protocol"].get("contrast")) != spec.get("contrast", False):
        raise ValueError("Contrast mode mismatch")
    stamp = {
        "gallery_metadata": meta,
        "features": sha(args.features / "manifest.json"),
        "model": sha(args.model / "model.json"),
        "queries": queries,
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    proto = out / "prediction_protocol.json"
    if proto.exists() and p.read_json(proto) != stamp:
        raise ValueError("Existing predictions have different inputs")
    p.write_json(proto, stamp)
    models = []
    for name in KINDS:
        path = args.model / (name + ".txt")
        if sha(path) != spec["weights"][name]:
            raise ValueError("Model weights changed")
        models.append(lgb.Booster(model_str=path.read_text(encoding="utf-8")))
    path = args.model / "count.txt"
    if sha(path) != spec["weights"]["count"]:
        raise ValueError("Count model changed")
    volume = lgb.Booster(model_str=path.read_text(encoding="utf-8"))
    decision = spec["decision"]
    results = {}
    literal = {}
    candidates = {}
    docs = p.read_json(args.gallery / "weighted_docs.json")
    facts = {}
    profiles = {}
    if spec["visual_enrichment"]:
        fact_path = getattr(args, "facts", None) or args.gallery / "facts.json"
        if sha(fact_path) != m["protocol"]["facts"]:
            raise ValueError("Changed visual facts")
        facts = p.read_json(fact_path)
    if spec.get("contrast"):
        if not args.contrast or sha(args.contrast) != m["protocol"]["contrast"]:
            raise ValueError("Provide matching --contrast")
        profiles = p.read_json(args.contrast)["profiles"]
    paths = {i["id"]: i["path"] for i in inv["images"]}
    fields = {i: doc_fields(d, facts.get(i, {}))[0] for i, d in docs.items()}
    with (out / "matches.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ("case_id", "query", "image_id", "relative_path", "score", "evidence")
        )
        for q in queries:
            doc, x = load_block(args.features, m, q)
            both = probabilities(models, x, decision["iterations"])
            mix = decision["pairwise_weight"]
            count = volume.predict([count_features(x)], num_threads=1)[0]
            scores = relative_score(both, mix, count, decision["count_adjustment"])
            indices = np.flatnonzero(scores >= decision["threshold"])
            selected = indices[np.argsort(-scores[indices], kind="stable")]
            results[q] = [ids[doc[i]] for i in selected]
            candidates[q] = [ids[j] for j in doc]
            literal[q] = [
                ids[doc[i]]
                for i in np.flatnonzero(
                    x[:, m["columns"].index("literal_exact_search")]
                )
            ]
            plan = plans[q]
            words = [
                (role, t)
                for role in ("heads", "facets", "places")
                for g in plan[role]
                for t in g
            ]
            words += [(role, t) for role in ("examples", "context") for t in plan[role]]
            words += [
                (role, t) for role, items in profiles.get(q, {}).items() for t in items
            ]
            words = [(role + ":" + word, norm(word)) for role, word in words if word]
            for i in selected:
                image = ids[doc[i]]
                evidence = {
                    key: matches
                    for key, term in words
                    if (
                        matches := [
                            name for name, f in zip(FIELDS, fields[image]) if term in f
                        ]
                    )
                }
                if x[i, m["columns"].index("year_support")]:
                    evidence["matched_year"] = sorted(
                        set(map(str, plan["years"])) & doc_fields(docs[image], {})[3]
                    )
                if (
                    "nearby_strength" in m["columns"]
                    and x[i, m["columns"].index("nearby_strength")] > 0
                ):
                    evidence["nearby_evidence_strength"] = round(
                        float(x[i, m["columns"].index("nearby_strength")]), 3
                    )
                evidence = json.dumps(evidence, ensure_ascii=False)
                writer.writerow(
                    (
                        q,
                        plans[q]["query"],
                        image,
                        paths[image],
                        round(float(scores[i]), 6),
                        evidence,
                    )
                )
    path = out / "predictions.json"
    p.write_json(
        path,
        {
            "queries": {q: plans[q]["query"] for q in queries},
            "ranker": results,
            "literal": literal,
            "candidates": candidates,
        },
    )
    p.write_json(
        out / "seal.json",
        {
            "predictions": sha(path),
            "protocol": signature(stamp),
            "matches": sha(out / "matches.csv"),
        },
    )


def score(args):
    path = args.out / "predictions.json"
    seal = p.read_json(args.out / "seal.json")
    if seal["predictions"] != sha(path) or seal["protocol"] != signature(
        p.read_json(args.out / "prediction_protocol.json")
    ):
        raise ValueError("Predictions changed")
    protocol = p.read_json(args.out / "prediction_protocol.json")
    inv = p.read_json(args.gallery / "inventory.json")
    meta = protocol["gallery_metadata"]
    if (
        sha(args.out / "matches.csv") != seal["matches"]
        or family_inventory(inv) != meta["inventory"]
        or sha(args.root / INDEX) != meta["index"]
    ):
        raise ValueError("Scoring inputs or returned paths changed")
    pred = p.read_json(path)
    if any(
        set(pred[key]) != set(protocol["queries"])
        for key in ("queries", "ranker", "literal", "candidates")
    ):
        raise ValueError("Incomplete prediction scope")
    labels = label_sets(
        args.root, pred["queries"], workbook=getattr(args, "reference_workbook", None)
    )
    mapping = file_ids(args.root, p.read_json(args.gallery / "inventory.json"))
    results = {
        method: measure(labels, mapping, {q: set(v) for q, v in pred[method].items()})
        for method in ("ranker", "literal", "candidates")
    }
    p.write_json(args.out / "scores.json", results)
    print(results["ranker"]["total"])


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="command", required=True)
    for name, keys in (
        ("fit", ("root", "gallery", "features", "plans", "split", "out")),
        ("predict", ("gallery", "features", "plans", "model", "out")),
        ("score", ("root", "gallery", "out")),
    ):
        sub = subs.add_parser(name)
        for k in keys:
            sub.add_argument("--" + k, type=Path, required=True)
        if name in ("fit", "score"):
            sub.add_argument("--reference-workbook", type=Path)
        if name == "fit":
            sub.add_argument("--mine-semantic", action="store_true")
        if name == "predict":
            for key in ("split", "contrast", "facts"):
                sub.add_argument("--" + key, type=Path)
    a = cli.parse_args()
    {"fit": train, "predict": predict, "score": score}[a.command](a)
