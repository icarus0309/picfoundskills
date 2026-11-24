"""Build a label-free BGE text index for the prepared photo gallery."""

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

import pipeline as p
from lexical_index import terms
from lexical_facts import family_inventory
from weighted_retrieval import sha


def save_arrays(path, **arrays):
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temp = Path(f.name)
        try:
            np.savez(f, **arrays)
        except BaseException:
            f.close()
            temp.unlink(missing_ok=True)
            raise
    os.replace(temp, path)


def model_hashes(directory):
    return {
        name: sha(directory / name)
        for name in ("model_quantized.onnx", "tokenizer.json")
    }


class TextEncoder:
    def __init__(self, directory, threads=4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=128)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(directory / "model_quantized.onnx"),
            opts,
            providers=["CPUExecutionProvider"],
        )

    def encode(self, texts):
        output = []
        for start in range(0, len(texts), 32):
            chunk = texts[start : start + 32]
            tokens = self.tokenizer.encode_batch(chunk)
            arrays = {
                "input_ids": [t.ids for t in tokens],
                "attention_mask": [t.attention_mask for t in tokens],
                "token_type_ids": [t.type_ids for t in tokens],
            }
            feed = {
                v.name: np.asarray(arrays[v.name], dtype=np.int64)
                for v in self.session.get_inputs()
            }
            hidden = self.session.run(None, feed)[0][:, 0]
            vec = hidden / np.maximum(
                np.linalg.norm(hidden, axis=1, keepdims=True), 1e-8
            )
            vec[[not t for t in chunk]] = 0
            output.append(vec)
        return np.concatenate(output, axis=0)


def build(gallery, model, out, threads):
    out.mkdir(parents=True, exist_ok=True)
    docs_path = gallery / "weighted_docs.json"
    digest = sha(docs_path)
    meta = p.read_json(gallery / "gallery.json")
    if meta["inventory"] != family_inventory(p.read_json(gallery / "inventory.json")):
        raise ValueError("Gallery inventory changed")
    if meta["docs_sha256"] != digest:
        raise ValueError("Gallery documents changed")
    stamp = dict(
        docs=digest,
        inventory=meta["inventory"],
        source_index=meta["index"],
        model=model_hashes(model),
        format=2,
    )
    protocol = out / "text_protocol.json"
    if protocol.exists() and p.read_json(protocol) != stamp:
        raise ValueError("Existing text index has different inputs")
    p.write_json(protocol, stamp)
    path = out / "text.npz"
    receipt = out / "text_complete.json"
    if (
        receipt.exists()
        and path.is_file()
        and p.read_json(receipt)["sha256"] == sha(path)
    ):
        return
    docs = p.read_json(docs_path)
    ids = sorted(docs)
    titles = [str(docs[i].get("summary") or "") for i in ids]
    abstracts = [
        str(docs[i].get("abstract") or docs[i].get("aiDescription") or "") for i in ids
    ]
    image_terms = [
        list(
            dict.fromkeys(
                terms(docs[i].get("keyword", ""))
                + terms(docs[i].get("search_term", ""))
                + terms(docs[i].get("tagIds", ""))
            )
        )[:30]
        for i in ids
    ]
    vocab = sorted(set(t for group in image_terms for t in group))
    # Deterministic batches of similar lengths reduce padding for variable-length Chinese captions.
    all_text = sorted(set(titles + abstracts + vocab + [""]), key=lambda t: (len(t), t))
    lookup = {t: n for n, t in enumerate(all_text)}
    tag_ids = np.zeros((len(ids), max(1, max(map(len, image_terms)))), np.int32)
    for row, tags in zip(tag_ids, image_terms):
        row[: len(tags)] = [lookup[t] for t in tags]
    assert lookup[""] == 0
    print(
        f"Encoding {len(all_text)} distinct titles, descriptions and tags", flush=True
    )
    vectors = TextEncoder(model, threads).encode(all_text)
    save_arrays(
        path,
        ids=np.asarray(ids),
        vectors=vectors,
        tags=tag_ids,
        title=np.array([lookup[t] for t in titles]),
        abstract=np.array([lookup[t] for t in abstracts]),
    )
    p.write_json(
        receipt, dict(sha256=sha(path), images=len(ids), strings=len(all_text))
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for arg in ("gallery", "model", "out"):
        cli.add_argument("--" + arg, type=Path, required=True)
    cli.add_argument("--threads", type=int, default=4)
    a = cli.parse_args()
    build(a.gallery, a.model, a.out, a.threads)
