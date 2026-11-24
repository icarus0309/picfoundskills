"""Predict a gallery description prototype from a query using training relevance pairs."""

import argparse
from pathlib import Path

import numpy as np

import pipeline as p
from lexical_facts import signature
from semantic_features import augment, relative
from semantic_index import TextEncoder, model_hashes, save_arrays
from weighted_retrieval import sha


def grouped_folds(groups, folds=5):
    parent = list(range(len(groups)))

    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    seen = {}
    for n, keys in enumerate(groups):
        for k in keys:
            if k in seen:
                parent[root(n)] = root(seen[k])
            else:
                seen[k] = n
    components = {}
    for n in range(len(groups)):
        components.setdefault(root(n), []).append(n)
    loads = [0] * folds
    assignments = np.zeros(len(groups), np.int32)
    for component in sorted(components.values(), key=lambda c: (-len(c), min(c))):
        target = min(range(folds), key=lambda j: loads[j])
        loads[target] += len(component)
        assignments[component] = target
    return assignments


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def ridge(x, y):
    center = x.mean(axis=0)
    z = unit(x - center)
    # Both views use normalized, corpus-centered vectors. Ridge shrinks unsupported directions.
    lhs = z.T @ z + 3 * np.eye(z.shape[1], dtype=np.float32)
    weights = np.linalg.solve(lhs, z.T @ y).astype(np.float32)
    return {"center": center, "weights": weights}


def train(index, bank_dir, out):
    bank = p.read_json(bank_dir / "bank.json")
    text_source = p.read_json(index / "text_protocol.json")
    bank_source = p.read_json(bank_dir / "bank_source.json")
    if (
        any(
            text_source.get(k) != bank_source[k]
            for k in ("inventory", "source_index", "docs")
        )
        or bank["gallery"] != bank_source["inventory"]
        or sha(bank_dir / "bank.json") != bank_source["bank"]
    ):
        raise ValueError("Projection requires feedback and text from the same gallery")
    model_id = text_source["model"]
    receipt = p.read_json(bank_dir / "complete.json")
    if (
        bank["model"] != model_id
        or sha(bank_dir / "bank.json") != receipt["bank"]
        or sha(bank_dir / "query-vectors.npz") != receipt["vectors"]
        or sha(index / "text.npz")
        != p.read_json(index / "text_complete.json")["sha256"]
    ):
        raise ValueError("Source index or feedback bank changed")
    out.mkdir(parents=True, exist_ok=True)
    stamp = dict(bank=receipt, gallery_index=sha(index / "text.npz"), version=1)
    if (out / "fit_inputs.json").exists() and p.read_json(
        out / "fit_inputs.json"
    ) != stamp:
        raise ValueError("Use a fresh projection output")
    p.write_json(out / "fit_inputs.json", stamp)
    with np.load(bank_dir / "query-vectors.npz", allow_pickle=False) as a:
        x = a["vectors"]
    with np.load(index / "text.npz", allow_pickle=False) as a:
        if bank["ids"] != a["ids"].tolist():
            raise ValueError("Bank belongs to a different gallery")
        vec = a["vectors"]
        views = [vec[a[c]] for c in ("title", "abstract")]
    centered = [unit(v - v.mean(axis=0)) for v in views]
    use = [i for i, items in enumerate(bank["photos"]) if items]
    if len(use) < 6:
        raise ValueError("Training requires at least six queries with mapped positives")
    targets = np.column_stack(
        [
            unit(np.vstack([v[bank["photos"][i]].mean(axis=0) for i in use]))
            for v in centered
        ]
    )
    folds = grouped_folds([bank["keys"][i] for i in use])
    x = x[use]
    private = out / "training-folds"
    private.mkdir(exist_ok=True)
    assignment = {bank["queries"][i]: int(folds[j]) for j, i in enumerate(use)}
    p.write_json(private / "assignment.json", assignment)
    p.write_json(
        private / "source.json",
        {"split": bank_source["split"], "bank": bank_source["bank"]},
    )
    save_arrays(out / "projection.npz", **ridge(x, targets))
    hashes = {}
    for k in sorted(set(folds.tolist())):
        keep = folds != k
        if not keep.any():
            raise ValueError("Training queries form a single overlapping topic")
        file = private / f"{k}.npz"
        save_arrays(file, **ridge(x[keep], targets[keep]))
        hashes[str(k)] = sha(file)
    p.write_json(
        private / "seal.json",
        dict(folds=hashes, assignment=sha(private / "assignment.json")),
    )
    p.write_json(
        out / "projection.json",
        dict(model=model_id, weights=sha(out / "projection.npz"), version=1),
    )


class ProjectedScores:
    def __init__(self, index, model, project, ids, gallery, threads, training):
        spec = p.read_json(project / "projection.json")
        proto = p.read_json(index / "text_protocol.json")
        checksum = sha(index / "text.npz")
        if (
            spec["model"] != model_hashes(model)
            or proto["model"] != spec["model"]
            or proto["docs"] != p.read_json(gallery / "gallery.json")["docs_sha256"]
            or checksum != p.read_json(index / "text_complete.json")["sha256"]
            or sha(project / "projection.npz") != spec["weights"]
        ):
            raise ValueError("Incompatible learned projection or text index")
        with np.load(index / "text.npz", allow_pickle=False) as a:
            if a["ids"].tolist() != ids:
                raise ValueError("Text features use different image IDs")
            vec = a["vectors"]
            self.views = []
            for k in ("title", "abstract"):
                v = vec[a[k]]
                active = a[k] != 0
                self.views.append((unit(v - v.mean(axis=0)), active))
        self.encoder = TextEncoder(model, threads)
        self.model_id = spec
        self.index_sha = signature([checksum, spec])
        self.assignment = {}

        def weights(path):
            with np.load(path, allow_pickle=False) as arr:
                return arr["center"], arr["weights"]

        self.weights = {-1: weights(project / "projection.npz")}
        if training:
            private = project / "training-folds"
            seal = p.read_json(private / "seal.json")
            if sha(private / "assignment.json") != seal["assignment"]:
                raise ValueError("Cross-fitting assignments changed")
            self.assignment = p.read_json(private / "assignment.json")
            for k, digest in seal["folds"].items():
                path = private / (k + ".npz")
                if sha(path) != digest:
                    raise ValueError("Cross-fitting weights changed")
                self.weights[int(k)] = weights(path)
            self.index_sha = signature([self.index_sha, seal])

    def features(self, plan, doc, x, columns):
        q = self.encoder.encode([plan["query"]])[0]
        center, w = self.weights[self.assignment.get(plan["query"], -1)]
        pred = np.split(unit(q - center) @ w, 2)
        names = []
        values = []
        for k, (vectors, mask), query in zip(("title", "abstract"), self.views, pred):
            val = np.where(mask[doc], vectors[doc] @ unit(query), 0)
            names.extend(["projected_" + k, "projected_relative_" + k])
            values.extend([val, relative(val)])
        return names, np.column_stack(values)


def build(
    source, gallery, plans, index, model, project, out, training=False, threads=3
):
    ev = ProjectedScores(
        index,
        model,
        project,
        p.read_json(source / "ids.json"),
        gallery,
        threads,
        training,
    )
    augment(source, gallery, plans, out, ev, "projection")
    record = dict(manifest=sha(out / "manifest.json"), cross_fit=training)
    if training:
        record.update(p.read_json(project / "training-folds" / "source.json"))
    p.write_json(out / "projection_mode.json", record)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    subs = cli.add_subparsers(dest="stage", required=True)
    for stage, names in (
        ("fit", ("index", "bank", "out")),
        (
            "augment",
            ("source", "gallery", "plans", "index", "model", "projection", "out"),
        ),
    ):
        sub = subs.add_parser(stage)
        for k in names:
            sub.add_argument("--" + k, type=Path, required=True)
        if stage == "augment":
            sub.add_argument("--cross-fit", action="store_true")
    a = cli.parse_args()
    if a.stage == "fit":
        train(a.index, a.bank, a.out)
    else:
        build(
            a.source,
            a.gallery,
            a.plans,
            a.index,
            a.model,
            a.projection,
            a.out,
            a.cross_fit,
        )
