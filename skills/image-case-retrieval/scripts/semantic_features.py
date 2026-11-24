"""Attach query-conditioned text or pixel evidence to a frozen lexical candidate set."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import pipeline as p
from clip_index import Encoder
from lexical_facts import signature, family_inventory
from lexical_features import block_name, load_block
from semantic_index import TextEncoder, model_hashes, save_arrays
from weighted_retrieval import sha


def prompts(plan):
    subjects = [group[0] for group in plan["heads"] if group]
    details = [group[0] for key in ("facets", "places") for group in plan[key] if group]
    return [plan["query"], "、".join(subjects), "、".join(details)]


def relative(values):
    # A cosine of 0.55 can be exceptional for one query and common for another.
    if not len(values):
        return values
    return (values - np.median(values)) / max(float(np.std(values)), 0.01)


class FieldScores:
    def __init__(self, index, model, ids, gallery, threads):
        protocol = p.read_json(index / "text_protocol.json")
        if (protocol["docs"] != p.read_json(gallery / "gallery.json")["docs_sha256"]
                or protocol["model"] != model_hashes(model)):
            raise ValueError("Text index and model or gallery do not match")
        path = index / "text.npz"
        self.index_sha = sha(path)
        if self.index_sha != p.read_json(index / "text_complete.json")["sha256"]:
            raise ValueError("Incomplete text index")
        with np.load(path, allow_pickle=False) as data:
            if data["ids"].tolist() != ids:
                raise ValueError("Gallery image order changed")
            self.vec, self.title, self.abstract, self.tags = (
                data[k] for k in ("vectors", "title", "abstract", "tags"))
        self.encoder = TextEncoder(model, threads)
        self.model_id = protocol["model"]

    def features(self, plan, doc, x, columns):
        words = prompts(plan)
        embedded = self.encoder.encode(["为这个句子生成表示以用于检索相关文章："+t if t else "" for t in words])
        sim = self.vec @ embedded.T
        values = []
        names = []
        for j, role in enumerate(("query", "head", "detail")):
            score = sim[:, j]
            tags = score[self.tags[doc]]
            fields = {"title": score[self.title[doc]], "abstract": score[self.abstract[doc]],
                      "strongest_tag": tags.max(axis=1), "first_tags": tags[:, :3].max(axis=1)}
            for field, vals in fields.items():
                names.append("semantic_" + role + "_" + field)
                values.append(vals)
                if role == "query":
                    names.append("semantic_relative_" + field)
                    values.append(relative(vals))
        return names, np.column_stack(values)


def graph(photos, valid, k=6):
    """Nearest *other* photos, in blocks to bound RAM. Invalid vectors never cast a vote."""
    width = min(k, max(0, int(valid.sum()) - 1))
    if not width:
        return np.empty((len(photos), 0), np.int32)
    rows = []
    for off in range(0, len(photos), 128):
        scores = photos[off:off + 128] @ photos.T
        scores[:, ~valid] = -np.inf
        scores[np.arange(len(scores)), np.arange(off, off + len(scores))] = -np.inf
        selected = np.argpartition(-scores, width - 1, axis=1)[:, :width]
        ordered = np.argsort(-np.take_along_axis(scores, selected, axis=1), axis=1)
        rows.append(np.take_along_axis(selected, ordered, axis=1))
    return np.vstack(rows).astype(np.int32)


class PixelScores:
    def __init__(self, index, model, ids, gallery, threads):
        protocol = p.read_json(index / "clip_protocol.json")
        inv = p.read_json(gallery / "inventory.json")["images"]
        inventory_hash = hashlib.sha256(json.dumps(inv, sort_keys=True).encode()).hexdigest()
        self.model_id = {k: sha(model / k) for k in protocol["model"]}
        done = p.read_json(index / "clip_completion.json")
        if (inventory_hash != protocol["images"] or protocol["model"] != self.model_id
                or not done["complete"] or done["priority_only"] or done["requested"] != len(ids)):
            raise ValueError("A completed full-gallery Chinese-CLIP index and matching encoder are required")
        paths = [index / "vectors" / (i + ".npy") for i in ids]
        digests = [sha(path) if path.exists() else None for path in paths]
        if sum(v is not None for v in digests) != done["encoded"]:
            raise ValueError("Incomplete pixel vectors")
        self.index_sha = signature([ids, digests, self.model_id])
        combined = index / "rerank-graph.npz"
        receipt = index / "rerank-graph.json"
        if (not receipt.exists() or not combined.exists()
                or p.read_json(receipt) != dict(source=self.index_sha, data=sha(combined), version=1)):
            valid = np.asarray([v is not None for v in digests])
            if not valid.any():
                raise ValueError("The corpus has no decodable photos")
            first = np.load(paths[np.flatnonzero(valid)[0]], allow_pickle=False)
            photos = np.zeros((len(ids), len(first)), np.float32)
            for row in np.flatnonzero(valid):
                photos[row] = np.load(paths[row], allow_pickle=False)
            save_arrays(combined, photos=photos, valid=valid, neighbors=graph(photos, valid))
            p.write_json(receipt, dict(source=self.index_sha, data=sha(combined), version=1))
        with np.load(combined, allow_pickle=False) as data:
            self.photos, self.valid, self.neighbors = (data[k] for k in ("photos","valid","neighbors"))
        self.encoder = Encoder(model, threads)

    def features(self, plan, doc, x, columns):
        words = prompts(plan)
        vectors = self.encoder.texts([t or "照片" for t in words])
        sims = self.photos @ vectors.T
        for j, t in enumerate(words):
            if not t:
                sims[:, j] = 0
        values = [self.valid[doc].astype(np.float32)]
        names = ["pixel_available"]
        for j, role in enumerate(("query", "head", "detail")):
            vals = sims[doc,j]
            values.extend([vals,relative(vals)])
            names.extend(["pixel_"+role, "pixel_relative_"+role])
        near = self.neighbors[doc]
        for name in ("object_prominent_fraction","head_exact_search","semantic_query_abstract"):
            support = np.zeros(len(self.photos), np.float32)
            support[doc] = x[:,columns.index(name)]
            vals = support[near] if near.shape[1] else np.zeros((len(doc), 1), np.float32)
            values.extend([np.where(self.valid[doc], vals.max(axis=1), 0),
                           np.where(self.valid[doc], vals.mean(axis=1), 0)])
            names.extend(["pixel_neighbor_max_"+name,"pixel_neighbor_mean_"+name])
        return names,np.column_stack(values)


def augment(source, gallery, plans_path, out, evidence, kind):
    manifest = p.read_json(source / "manifest.json")
    ids = p.read_json(source / "ids.json")
    plans = p.read_json(plans_path)
    old = manifest["protocol"]
    meta = p.read_json(gallery / "gallery.json")
    if (old["gallery"] != meta["docs_sha256"] or old["source_index"] != meta["index"]
            or sha(gallery / "weighted_docs.json") != old["gallery"]):
        raise ValueError("Lexical and semantic documents or source index differ")
    if (sha(plans_path) != old["plans"] or set(plans) != set(manifest["blocks"])
            or family_inventory(p.read_json(gallery / "inventory.json")) != old["inventory"]
            or sha(source / "ids.json") != manifest["ids_hash"]):
        raise ValueError("Feature block inputs changed")
    impl = hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()
    encoder = signature([old["encoder"], kind, impl, evidence.model_id])
    stamp = dict(old, encoder=encoder, **{kind + "_source": sha(source / "manifest.json"),
                                        kind + "_index": evidence.index_sha})
    out.mkdir(parents=True, exist_ok=True)
    proto = out / "feature_protocol.json"
    if proto.exists() and p.read_json(proto) != stamp:
        raise ValueError("Changed feature inputs; choose a new directory")
    p.write_json(proto, stamp)
    p.write_json(out / "ids.json",ids)
    receipts = out / "receipts"
    receipts.mkdir(exist_ok=True)
    hashes = {}
    cols = None
    for num,(q,plan) in enumerate(plans.items(),1):
        path = out / block_name(q)
        receipt = receipts / (path.stem+".json")
        if receipt.exists() and path.exists():
            entry = p.read_json(receipt)
            if sha(path) == entry["sha256"]:
                hashes[q] = entry["sha256"]
                cols = entry["columns"]
                continue
        doc,x = load_block(source,manifest,q)
        names, values = evidence.features(plan, doc, x, manifest["columns"])
        cols = manifest["columns"] + names
        with tempfile.NamedTemporaryFile(dir=out, delete=False) as f:
            tmp=Path(f.name)
            np.savez_compressed(f, ids=doc, x=np.column_stack((x,values)).astype(np.float32))
        os.replace(tmp,path)
        hashes[q]=sha(path)
        p.write_json(receipt,dict(sha256=hashes[q],columns=cols))
        if num%100==0:
            print(f"{kind} {num}/{len(plans)}",flush=True)
    p.write_json(out / "manifest.json",dict(protocol=stamp,columns=cols,blocks=hashes,
                                             ids_hash=manifest["ids_hash"],signature=signature(stamp)))


def build(kind, source, gallery, plans, index, model, out, threads=4):
    cls = {"text":FieldScores, "pixel":PixelScores}[kind]
    evidence = cls(index, model, p.read_json(source / "ids.json"), gallery, threads)
    augment(source, gallery, plans, out, evidence, kind)


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("kind",choices=["text","pixel"])
    for key in ("source","gallery","plans","index","model","out"):
        cli.add_argument("--"+key,type=Path,required=True)
    cli.add_argument("--threads",type=int,default=4)
    a=cli.parse_args()
    build(a.kind,a.source,a.gallery,a.plans,a.index,a.model,a.out,a.threads)

