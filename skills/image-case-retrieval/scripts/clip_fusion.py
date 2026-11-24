"""Fuse independently encoded pixels with sealed caption/VLM evidence, without reference fids."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pipeline as p
from clip_index import Encoder
from weighted_retrieval import export_matches, sha


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def normalize(values, positives, negatives):
    hi = float(np.median(values[positives]))
    lo = float(np.median(values[negatives]))
    scale = max(hi - lo, float(np.std(values)) * 0.4, 1e-5)
    return sigmoid((values - (hi + lo) / 2) * 4 / scale)


def votes(vectors, anchor_pos, anchor_neg, text_vector, ids):
    text_cos = vectors @ text_vector
    lang = normalize(text_cos, anchor_pos, anchor_neg)
    n = len(vectors)
    folds = np.array(
        [int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 4 for i in ids]
    )
    linear = np.zeros(n)
    near = np.zeros(n)
    for fold in range(4):
        test = folds == fold
        pos = anchor_pos & ~test
        neg = anchor_neg & ~test
        if not test.any():
            continue
        if pos.sum() < 2 or neg.sum() < 2:
            linear[test] = lang[test]
            near[test] = lang[test]
            continue
        vp, vn = vectors[pos], vectors[neg]
        # Estimate a regularized, class-balanced direction. References never enter the fit.
        mp = vp.mean(axis=0)
        mn = vn.mean(axis=0)
        centered_p, centered_n = vp - mp, vn - mn
        cov = (
            centered_p.T @ centered_p / len(vp) + centered_n.T @ centered_n / len(vn)
        ) / 2
        cov.flat[:: cov.shape[0] + 1] += 0.015
        direction = np.linalg.solve(cov, mp - mn)
        projected = vectors @ direction
        linear[test] = normalize(projected, pos, neg)[test]

        # Compare to a small number of closest positives and equally many negatives.
        # The held-out item cannot serve as its own visual exemplar.
        def top_mean(reference):
            k = min(3, len(reference))
            similarity = vectors[test] @ reference.T
            return np.partition(similarity, -k, axis=1)[:, -k:].mean(axis=1)

        near[test] = sigmoid((top_mean(vp) - top_mean(vn)) * 18)
    return (0.40 * linear + 0.40 * near + 0.20 * lang).clip(0, 1)


def run(args):
    if not 0 < args.threshold <= 1:
        raise ValueError("Threshold must be in (0,1]")
    source = args.source
    if args.out.resolve() == source.resolve():
        raise ValueError("Fusion needs its own output directory")
    completion = p.read_json(args.index / "clip_completion.json")
    if (
        not completion.get("complete")
        or completion["requested"] != completion["encoded"] + completion["errors"]
    ):
        raise ValueError("Finish building the pixel index before fusion")
    stamp = p.read_json(source / "weighted_protocol.json")
    seal = p.read_json(source / "weighted_seal.json")
    if sha(source / "weighted_predictions.json") != seal["sha256"]:
        raise ValueError("Input predictions do not match seal")
    manifest = p.read_json(args.index / "clip_protocol.json")
    inv = p.read_json(source / "inventory.json")
    import json

    if (
        hashlib.sha256(json.dumps(inv["images"], sort_keys=True).encode()).hexdigest()
        != manifest["images"]
    ):
        raise ValueError("Pixel index and prediction inventory differ")
    for name, value in manifest["model"].items():
        if sha(args.model_dir / name) != value:
            raise ValueError("Text and image encoders differ")
    encoder = Encoder(args.model_dir, args.threads)
    base = p.read_json(source / "weighted_predictions.json")
    cases = seal["cases"]
    vector_files = sorted((args.index / "vectors").glob("*.npy"))
    vector_digest = hashlib.sha256()
    vectors = {}
    import io

    for f in vector_files:
        data = f.read_bytes()
        vector_digest.update(f.stem.encode() + hashlib.sha256(data).digest())
        vectors[f.stem] = np.load(io.BytesIO(data), allow_pickle=False)
    eligible = {
        i for rows in base.values() for i, n in rows.items() if n is not None and n >= 1
    }
    result = {q: {} for q in base}
    diagnostics = {}
    for q, rows in base.items():
        ids = sorted(i for i, n in rows.items() if n is not None and i in vectors)
        count = len(ids)
        background = sorted(
            (eligible & vectors.keys()) - rows.keys(),
            key=lambda i: hashlib.sha256((q + i).encode()).digest(),
        )[:128]
        ids += background
        raw = np.array([rows.get(i, 0) for i in ids])
        pos = raw >= 3.5
        neg = raw <= 1.5
        result[q] = {i: (None if n is None else n / 4) for i, n in rows.items()}
        if pos.sum() < 3 or neg.sum() < 3:
            diagnostics[q] = {
                "mode": "caption_vlm_fallback",
                "anchors": [int(pos.sum()), int(neg.sum())],
            }
            continue
        matrix = np.stack([vectors[i] for i in ids])
        query = cases[q]
        text = encoder.texts([query, "一张关于" + query + "的照片"])[0:2].mean(axis=0)
        text /= max(np.linalg.norm(text), 1e-8)
        visual = votes(matrix, pos, neg, text, ids)
        # Moderate semantic matches benefit most from an independent pixel vote.
        for j, i in enumerate(ids[:count]):
            result[q][i] = float(0.45 * (raw[j] / 4) + 0.55 * visual[j])
        diagnostics[q] = {
            "mode": "fusion",
            "indexed_pairs": count,
            "anchors": [int(pos.sum()), int(neg.sum())],
        }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pred = out / "weighted_predictions.json"
    existing = out / "weighted_seal.json"
    if existing.exists():
        frozen = p.read_json(existing)
        previous = p.read_json(out / "weighted_protocol.json")
        if (
            frozen["threshold"] != args.threshold
            or previous.get("source_sha256") != seal["sha256"]
            or previous.get("vectors_sha256") != vector_digest.hexdigest()
            or previous.get("fusion_code") != sha(Path(__file__))
        ):
            raise ValueError("Existing fusion output differs; use another directory")
    p.write_json(pred, result)
    export_matches(out, inv, cases, result, args.threshold)
    p.write_json(
        out / "weighted_protocol.json",
        {
            **stamp,
            "source_sha256": seal["sha256"],
            "fusion_code": sha(Path(__file__)),
            "index_protocol": sha(args.index / "clip_protocol.json"),
            "vector_count": len(vectors),
            "vectors_sha256": vector_digest.hexdigest(),
        },
    )
    p.write_json(out / "weighted_fusion_diagnostics.json", diagnostics)
    p.write_json(out / "inventory.json", inv)
    p.write_json(
        out / "weighted_seal.json",
        {"cases": cases, "threshold": args.threshold, "sha256": sha(pred)},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "index", "model-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--threads", type=int, default=4)
    run(parser.parse_args())
