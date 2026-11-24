"""Add collection-level evidence to existing candidates, without selecting or excluding folders."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
import numpy as np
import pipeline as p
from lexical_facts import family_inventory, signature
from lexical_features import block_name, load_block
from weighted_retrieval import sha

CHANNELS = (
    "literal_exact_search",
    "head_exact_search",
    "head_title",
    "head_prominent",
    "alias_exact_search",
    "instance_title",
    "typical_title",
    "equivalent_title",
)


def augment(x, doc, columns, groups, group_size):
    local = groups[doc]
    names = []
    data = [x]
    n = int(group_size.sum())
    for key in CHANNELS:
        if key == "head_prominent":
            key = "object_prominent_fraction"
        if key not in columns:
            continue
        support = x[:, columns.index(key)]
        hits = np.bincount(local, weights=support, minlength=len(group_size))
        prior = (hits.sum() + 1) / (n + 2)
        rate = (hits + 10 * prior) / (group_size + 10)
        for suffix, values in (
            ("rate", rate),
            ("share", hits / max(1, hits.sum())),
            ("relative", rate / max(1e-6, rate.max())),
        ):
            names.append("album_" + key + "_" + suffix)
            data.append(values[local, None])
    return np.column_stack(data).astype(np.float32), columns + names


def build(source, gallery, out):
    m = p.read_json(source / "manifest.json")
    inv = p.read_json(gallery / "inventory.json")
    ids = p.read_json(source / "ids.json")
    if (
        sha(source / "ids.json") != m["ids_hash"]
        or family_inventory(inv) != m["protocol"]["inventory"]
    ):
        raise ValueError("Gallery identity mismatch")
    folders = {i["id"]: i["directory"] for i in inv["images"]}
    _, groups = np.unique([folders[i] for i in ids], return_inverse=True)
    sizes = np.bincount(groups)
    out.mkdir(parents=True, exist_ok=True)
    stamp = dict(m["protocol"], album_source=sha(source / "manifest.json"))
    stamp["encoder"] = signature(
        [
            stamp["encoder"],
            hashlib.sha256(
                Path(__file__).read_text(encoding="utf-8-sig").encode()
            ).hexdigest(),
        ]
    )
    protocol = out / "feature_protocol.json"
    if protocol.exists() and p.read_json(protocol) != stamp:
        raise ValueError("Changed album inputs")
    p.write_json(protocol, stamp)
    p.write_json(out / "ids.json", ids)
    prior = (
        p.read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
    )
    hashes = {}
    names = prior.get("columns")
    for q in m["blocks"]:
        path = out / block_name(q)
        if path.exists() and prior.get("blocks", {}).get(q) == sha(path):
            hashes[q] = prior["blocks"][q]
            continue
        doc, x = load_block(source, m, q)
        features, names = augment(x, doc, m["columns"], groups, sizes)
        with tempfile.NamedTemporaryFile("wb", dir=out, delete=False) as f:
            tmp = Path(f.name)
            np.savez_compressed(f, ids=doc, x=features)
        os.replace(tmp, path)
        hashes[q] = sha(path)
    p.write_json(
        out / "manifest.json",
        dict(
            protocol=stamp,
            columns=names,
            blocks=hashes,
            ids_hash=sha(out / "ids.json"),
            signature=signature(stamp),
        ),
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("features", "gallery", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    a = cli.parse_args()
    build(a.features, a.gallery, a.out)
