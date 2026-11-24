"""Rerank query-caption pairs using a local multilingual cross-encoder at unchanged return volume."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import lexical_ranker as lr
import pipeline as p
from boundary_review import read_parent
from lexical_facts import signature
from lexical_index import terms
from weighted_retrieval import sha


def passage(row):
    fields = [
        row.get("summary", ""),
        row.get("abstract") or (row.get("aiDescription") or "")[:100],
        "、".join(terms(row.get("search_term", ""))[:8]),
        " ".join(
            str(row.get(k) or "") for k in ("shootingTime", "city", "place_terms")
        ).strip(),
    ]
    return "。".join(s for s in fields if s)


def select(scores, cross, k, weight):
    order = np.argsort(-scores, kind="stable")
    count = len(cross)
    if not weight or not count:
        return order[:k]
    prefix = order[:count]
    first = scores[prefix]
    second = np.asarray(cross)
    blend = (1 - weight) * (first - first.mean()) / max(first.std(), 0.001) + weight * (
        second - second.mean()
    ) / max(second.std(), 0.001)
    inside = np.argsort(-blend, kind="stable")[: min(k, count)]
    return np.r_[prefix[inside], order[count:k]]


class CrossEncoder:
    def __init__(self, model, threads):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=112)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model / "model_quint8_avx2.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )

    def score(self, query, passages):
        outs = []
        for offset in range(0, len(passages), 16):
            batch = passages[offset : offset + 16]
            tokens = self.tokenizer.encode_batch([(query, text) for text in batch])
            arrays = {
                "input_ids": [v.ids for v in tokens],
                "attention_mask": [v.attention_mask for v in tokens],
                "token_type_ids": [v.type_ids for v in tokens],
            }
            feeds = {
                a.name: np.asarray(arrays[a.name], np.int64)
                for a in self.session.get_inputs()
            }
            res = self.session.run(None, feeds)[0].reshape(-1)
            if len(res) != len(batch) or not np.isfinite(res).all():
                raise ValueError("Expected one finite cross-encoder score per passage")
            outs.extend(res.tolist())
        return outs


def rerank(a):
    parent, proto, sealed = read_parent(a.source)
    if not 0 <= a.weight <= 1:
        raise ValueError("Weight must be between zero and one")
    if (
        sha(a.features / "manifest.json") != proto["features"]
        or sha(a.model / "model.json") != proto["model"]
    ):
        raise ValueError("Use the original features and model")
    manifest, ids = lr.read_features(a.features)
    inventory, meta = lr.check_gallery(a.gallery, manifest["protocol"])
    if meta != proto["gallery_metadata"]:
        raise ValueError("Gallery differs from predictions")
    docs = p.read_json(a.gallery / "weighted_docs.json")
    paths = {v["id"]: v["path"] for v in inventory["images"]}
    config = p.read_json(a.model / "model.json")
    cfg = config["decision"]
    models = []
    for name in (*lr.KINDS, "count"):
        f = a.model / (name + ".txt")
        if sha(f) != config["weights"][name]:
            raise ValueError("Ranker weights changed")
        models.append(lr.lgb.Booster(model_str=f.read_text(encoding="utf8")))
    enc = CrossEncoder(a.encoder, a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    stamps = {
        n: sha(a.encoder / n) for n in ("model_quint8_avx2.onnx", "tokenizer.json")
    }
    stamp = dict(
        proto,
        cross_encoder=dict(parent=sealed, encoder=stamps, weight=a.weight, version=1),
    )
    lock = a.out / "prediction_protocol.json"
    if lock.exists() and p.read_json(lock) != stamp:
        raise ValueError("Use a fresh output for different inputs")
    p.write_json(lock, stamp)
    sets = {}
    with (a.out / "matches.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["case_id", "query", "image_id", "relative_path", "score", "evidence"]
        )
        for q, query in parent["queries"].items():
            doc, x = lr.load_block(a.features, manifest, q)
            both = lr.probabilities(models, x, cfg["iterations"])
            count = models[-1].predict([lr.count_features(x)], num_threads=1)[0]
            scores = lr.relative_score(
                both, cfg["pairwise_weight"], count, cfg["count_adjustment"]
            )
            k = len(parent["ranker"][q])
            if {ids[doc[n]] for n in np.flatnonzero(scores >= cfg["threshold"])} != set(
                parent["ranker"][q]
            ):
                raise ValueError("Source must be the original ranker selection")
            order = np.argsort(-scores, kind="stable")
            head = order[: min(220, max(24, 2 * k + 12))] if k else order[:0]
            pair_scores = enc.score(query, [passage(docs[ids[doc[j]]]) for j in head])
            cross_values = dict(zip(head, pair_scores))
            selected = select(scores, pair_scores, k, a.weight)
            sets[q] = [ids[doc[j]] for j in selected]
            for rank, (j, i) in enumerate(zip(selected, sets[q]), 1):
                details = dict(
                    rank=rank,
                    lexical_semantic_score=round(float(scores[j]), 6),
                    cross_encoder_score=cross_values.get(j),
                )
                writer.writerow((q, query, i, paths[i], 1 / rank, json.dumps(details)))
    p.write_json(a.out / "predictions.json", dict(parent, ranker=sets))
    p.write_json(
        a.out / "seal.json",
        dict(
            predictions=sha(a.out / "predictions.json"),
            matches=sha(a.out / "matches.csv"),
            protocol=signature(stamp),
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("features", "gallery", "model", "source", "encoder", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    cli.add_argument("--weight", type=float, required=True)
    cli.add_argument("--threads", type=int, default=3)
    rerank(cli.parse_args())
