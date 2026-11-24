"""Use a private training-only bank of reviewed queries for a fixed gallery."""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

import pipeline as p
from lexical_facts import family_inventory, signature
from lexical_split import keys
from score_weighted import file_ids, label_sets
from semantic_features import augment
from semantic_index import TextEncoder, model_hashes, save_arrays
from weighted_retrieval import INDEX, sha


def make_bank(root, gallery, plans_path, split_path, model, out, workbook=None):
    plans = p.read_json(plans_path)
    split = p.read_json(split_path)
    train = split["train"]
    if (
        split["plans_sha256"] != sha(plans_path)
        or len(set(train)) != len(train)
        or set(train) & (set(split["test"]) | set(split["calibration"]))
    ):
        raise ValueError("Inconsistent or overlapping training split")
    inv = p.read_json(gallery / "inventory.json")
    meta = p.read_json(gallery / "gallery.json")
    if meta["inventory"] != family_inventory(inv) or meta["index"] != sha(root / INDEX):
        raise ValueError("Training labels and gallery require the same image mapping")
    queries = {q: plans[q]["query"] for q in train}
    labels = label_sets(root, queries, workbook=workbook)
    mapping = file_ids(root, inv)
    ids = sorted(i["id"] for i in inv["images"])
    pos = {img: n for n, img in enumerate(ids)}
    train = [q for q in train if labels[q]]
    masks = [
        sorted({pos[mapping[f]] for f in labels[q] if f in mapping}) for q in train
    ]
    bank = dict(
        ids=ids,
        queries=[plans[q]["query"] for q in train],
        keys=[sorted(keys(plans[q])) for q in train],
        photos=masks,
        gallery=meta["inventory"],
        model=model_hashes(model),
    )
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "bank.json"
    if dest.exists() and p.read_json(dest) != bank:
        raise ValueError("Feedback bank changed; choose a new private directory")
    p.write_json(dest, bank)
    if sha(gallery / "weighted_docs.json") != meta["docs_sha256"]:
        raise ValueError("Gallery evidence changed")
    source = dict(
        bank=sha(dest),
        inventory=meta["inventory"],
        source_index=meta["index"],
        docs=meta["docs_sha256"],
        split=sha(split_path),
    )
    bound = out / "bank_source.json"
    if bound.exists() and p.read_json(bound) != source:
        raise ValueError("Feedback bank has different source evidence")
    p.write_json(bound, source)
    receipt = out / "complete.json"
    path = out / "query-vectors.npz"
    if (
        receipt.exists()
        and path.exists()
        and p.read_json(receipt) == {"bank": sha(dest), "vectors": sha(path)}
    ):
        return
    encoder = TextEncoder(model)
    vectors = encoder.encode(bank["queries"])
    save_arrays(path, vectors=vectors)
    p.write_json(receipt, dict(bank=sha(dest), vectors=sha(path)))


class FeedbackScores:
    def __init__(self, bank_dir, model, ids, gallery, threads):
        bp = bank_dir / "bank.json"
        vp = bank_dir / "query-vectors.npz"
        hashes = {"bank": sha(bp), "vectors": sha(vp)}
        if hashes != p.read_json(bank_dir / "complete.json"):
            raise ValueError("Feedback bank changed")
        bank = p.read_json(bp)
        if (
            bank["ids"] != ids
            or bank["gallery"] != p.read_json(gallery / "gallery.json")["inventory"]
            or bank["model"] != model_hashes(model)
        ):
            raise ValueError("Use the matching private gallery feedback bank")
        self.index_sha = signature(hashes)
        self.model_id = {"feedback": 1, "text": bank["model"]}
        self.encoder = TextEncoder(model, threads)
        self.keys = [set(v) for v in bank["keys"]]
        with np.load(vp, allow_pickle=False) as arr:
            self.vectors = arr["vectors"]
        groups = bank["photos"]
        indptr = np.r_[0, np.cumsum([len(g) for g in groups])]
        indices = np.asarray([i for g in groups for i in g], dtype=np.int32)
        self.matrix = csr_matrix(
            (np.ones(len(indices), np.float32), indices, indptr),
            shape=(len(groups), len(ids)),
        )
        self.sizes = np.asarray(self.matrix.sum(axis=1)).ravel()

    def features(self, plan, doc, x, columns):
        target_keys = keys(plan)
        allowed = np.asarray(
            [not target_keys.intersection(group) for group in self.keys]
        )
        vec = self.encoder.encode([plan["query"]])[0]
        similarity = self.vectors @ vec
        permitted = np.flatnonzero(allowed)
        order = permitted[np.argsort(-similarity[permitted], kind="stable")]
        votes = []
        names = []
        for k in (1, 5, 20):
            best = order[:k]
            weight = np.exp(18 * (similarity[best] - 1))
            vote = np.asarray(self.matrix[best][:, doc].T @ weight).ravel()
            names.append(f"feedback_query_{k}")
            votes.append(vote)
        # Seed support is supplied by the new query itself. Remove the candidate's
        # own contribution to every two-hop vote to avoid treating it as corroboration.
        m = self.matrix[allowed][:, doc]
        size = np.maximum(self.sizes[allowed], 1)
        popularity = np.asarray(m.sum(axis=0)).ravel()
        names.append("feedback_judged_frequency")
        votes.append(np.log1p(popularity))
        for col in (
            "pixel_query",
            "semantic_query_abstract",
            "object_prominent_fraction",
        ):
            scores = x[:, columns.index(col)]
            seeds = np.zeros(len(doc), np.float32)
            if len(scores):
                # Keep ties together. Binary lexical evidence must not privilege low image IDs.
                cutoff = np.partition(scores, max(0, len(scores) - 15))[
                    max(0, len(scores) - 15)
                ]
                seeds[(scores >= cutoff) & (scores > 0)] = 1
            weight = np.asarray(m @ seeds).ravel() / np.sqrt(size)
            self_weight = np.asarray(m.T @ (1 / np.sqrt(size))).ravel()
            cross = np.maximum(
                0, np.asarray(m.T @ weight).ravel() - seeds * self_weight
            )
            votes.append(
                cross / np.sqrt(np.maximum(1, popularity) * max(1, seeds.sum()))
            )
            names.append("feedback_neighbors_" + col)
        return names, np.column_stack(votes)


def build(source, gallery, plans, bank, text_model, out, threads=4):
    data = FeedbackScores(
        bank, text_model, p.read_json(source / "ids.json"), gallery, threads
    )
    augment(source, gallery, plans, out, data, "feedback")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="stage", required=True)
    for stage, args in (
        ("bank", ("root", "gallery", "plans", "split", "model", "out")),
        ("augment", ("source", "gallery", "plans", "bank", "model", "out")),
    ):
        c = sub.add_parser(stage)
        for key in args:
            c.add_argument("--" + key, type=Path, required=True)
        if stage == "bank":
            c.add_argument("--reference-workbook", type=Path)
    a = cli.parse_args()
    if a.stage == "bank":
        make_bank(
            a.root, a.gallery, a.plans, a.split, a.model, a.out, a.reference_workbook
        )
    else:
        build(a.source, a.gallery, a.plans, a.bank, a.model, a.out)
