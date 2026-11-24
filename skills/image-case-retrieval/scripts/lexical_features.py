"""Build resumable, label-blind feature blocks for a batch of natural-language queries."""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
import numpy as np
import pipeline as p
import lexical_index
from lexical_facts import signature, family_inventory
from lexical_plans import valid
from weighted_retrieval import sha


def block_name(q):
    return hashlib.sha256(q.encode()).hexdigest() + ".npz"


def build(gallery, plans_file, out, facts_file=None, contrast_file=None):
    out.mkdir(parents=True, exist_ok=True)
    plans = p.read_json(plans_file)
    if not plans or any(
        not valid(r) or not isinstance(r.get("query"), str) for r in plans.values()
    ):
        raise ValueError("Invalid query plans")
    docs = gallery / "weighted_docs.json"
    meta = p.read_json(gallery / "gallery.json")
    if meta["docs_sha256"] != sha(docs) or meta["inventory"] != family_inventory(
        p.read_json(gallery / "inventory.json")
    ):
        raise ValueError("Gallery inventory or documents changed")
    if facts_file:
        manifest = p.read_json(facts_file.parent / "facts_manifest.json")
        if manifest["gallery"] != meta or manifest["facts_sha256"] != sha(facts_file):
            raise ValueError("Visual facts belong to a different gallery")
    profiles = None
    if contrast_file:
        contrast = p.read_json(contrast_file)
        if contrast["queries"] != {q: v["query"] for q, v in plans.items()} or set(
            contrast["profiles"]
        ) != set(plans):
            raise ValueError("Contrast vocabulary is for different queries")
        profiles = contrast["profiles"]
    code = Path(lexical_index.__file__).read_text(encoding="utf-8-sig")
    stamp = {
        "gallery": sha(docs),
        "inventory": meta["inventory"],
        "source_index": meta["index"],
        "plans": sha(plans_file),
        "contrast": sha(contrast_file) if contrast_file else None,
        "facts": sha(facts_file) if facts_file else None,
        "encoder": hashlib.sha256(code.encode()).hexdigest(),
        "builder_version": 2,
    }
    protocol = out / "feature_protocol.json"
    if protocol.exists() and p.read_json(protocol) != stamp:
        raise ValueError("Feature inputs changed; use another output")
    p.write_json(protocol, stamp)
    index = lexical_index.LexicalIndex(
        p.read_json(docs), p.read_json(facts_file) if facts_file else None
    )
    p.write_json(out / "ids.json", index.ids)
    previous = (
        p.read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
    )
    receipts = out / "receipts"
    receipts.mkdir(exist_ok=True)
    names = previous.get("columns")
    hashes = {}
    for num, (q, plan) in enumerate(plans.items(), 1):
        path = out / block_name(q)
        receipt = receipts / (path.stem + ".json")
        expected = previous.get("blocks", {}).get(q)
        if expected is None and receipt.exists():
            expected = p.read_json(receipt)["sha256"]
        if names is None or not path.exists() or not expected or sha(path) != expected:
            if profiles is not None:
                plan = {**plan, "contrast": profiles[q]}
            ids, x, names = index.features(plan)
            with tempfile.NamedTemporaryFile("wb", dir=out, delete=False) as f:
                tmp = Path(f.name)
                np.savez_compressed(f, ids=ids, x=x)
            os.replace(tmp, path)
            expected = sha(path)
            p.write_json(receipt, {"sha256": expected})
        hashes[q] = expected
        if num % 200 == 0:
            print(f"Features {num}/{len(plans)}", flush=True)
    p.write_json(
        out / "manifest.json",
        {
            "protocol": stamp,
            "columns": names,
            "blocks": hashes,
            "ids_hash": sha(out / "ids.json"),
            "signature": signature(stamp),
        },
    )


def load_block(out, manifest, q):
    path = out / block_name(q)
    if sha(path) != manifest["blocks"][q]:
        raise ValueError("Feature block changed")
    with np.load(path, allow_pickle=False) as data:
        return data["ids"], data["x"]


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("gallery", "plans", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    cli.add_argument("--facts", type=Path)
    cli.add_argument("--contrast", type=Path)
    a = cli.parse_args()
    build(a.gallery, a.plans, a.out, a.facts, a.contrast)
