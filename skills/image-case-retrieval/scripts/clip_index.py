"""Build a resumable Chinese-CLIP photo index locally, using an ONNX model directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pipeline as p


class Encoder:
    def __init__(self, model_dir, threads):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cfg = p.read_json(model_dir / "preprocessor_config.json")
        self.mean = np.asarray(cfg["image_mean"], dtype=np.float32)
        self.std = np.asarray(cfg["image_std"], dtype=np.float32)
        self.size = (cfg["size"]["width"], cfg["size"]["height"])
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=52)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_dir / "model_quantized.onnx"),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def pixels(self, image):
        rgb = image.convert("RGB").resize(self.size, Image.Resampling.BICUBIC)
        val = (np.asarray(rgb, dtype=np.float32) / 255 - self.mean) / self.std
        return val.transpose(2, 0, 1)

    def encode(self, name, pixels, strings):
        tokens = self.tokenizer.encode_batch(strings)
        feed = {
            "pixel_values": np.stack(pixels),
            "input_ids": np.array([v.ids for v in tokens], dtype=np.int64),
            "attention_mask": np.array(
                [v.attention_mask for v in tokens], dtype=np.int64
            ),
        }
        v = self.session.run([name], feed)[0]
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)

    def images(self, pixels):
        return self.encode("image_embeds", pixels, ["照片"])

    def texts(self, strings):
        return self.encode(
            "text_embeds",
            [np.zeros((3, self.size[1], self.size[0]), dtype=np.float32)],
            strings,
        )


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(1024 * 1024):
            h.update(data)
    return h.hexdigest()


def build(args):
    inventory = p.read_json(args.inventory)["images"]
    encoder = Encoder(args.model_dir, args.threads)
    stamp = {
        "images": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest(),
        "model": {
            name: digest(args.model_dir / name)
            for name in (
                "model_quantized.onnx",
                "preprocessor_config.json",
                "tokenizer.json",
            )
        },
        "root": str(args.root.resolve()),
        "format": 1,
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "clip_protocol.json"
    if manifest.exists() and p.read_json(manifest) != stamp:
        raise ValueError(
            "Photo inventory or encoder changed; use a new index directory"
        )
    p.write_json(manifest, stamp)
    vectors = out / "vectors"
    vectors.mkdir(exist_ok=True)
    failed = out / "decode_errors.json"
    errors = p.read_json(failed) if failed.exists() else {}
    priority = set()
    for file in args.prioritize:
        for pairs in p.read_json(file).values():
            priority.update(i for i, n in pairs.items() if n is not None and n >= 1)
    order = sorted(inventory, key=lambda r: (r["id"] not in priority, r["id"]))
    if args.only_priority:
        order = [r for r in order if r["id"] in priority]
    if args.only_priority and not args.prioritize:
        raise ValueError("--only-priority requires at least one prediction file")
    completion = out / "clip_completion.json"
    p.write_json(completion, {"complete": False})
    jobs = [r for r in order if not (vectors / (r["id"] + ".npy")).exists()]
    since = time.monotonic()
    for offset in range(0, len(jobs), args.batch):
        photos = []
        ids = []
        for row in jobs[offset : offset + args.batch]:
            i = row["id"]
            try:
                image = p.open_photo(args.root / "图片" / row["path"])
                with image:
                    photos.append(encoder.pixels(image))
                ids.append(i)
            except (OSError, ValueError, RuntimeError, ImportError) as exc:
                errors[i] = type(exc).__name__
        if photos:
            for i, v in zip(ids, encoder.images(photos), strict=True):
                path = vectors / (i + ".npy")
                with path.with_suffix(".tmp").open("wb") as f:
                    np.save(f, v.astype(np.float32), allow_pickle=False)
                os.replace(path.with_suffix(".tmp"), path)
                errors.pop(i, None)
        if offset % (args.batch * 16) == 0:
            p.write_json(failed, errors)
            print(
                f"{min(offset + args.batch, len(jobs))}/{len(jobs)}, {time.monotonic() - since:.1f}s",
                flush=True,
            )
    p.write_json(failed, errors)
    p.write_json(
        completion,
        {
            "complete": True,
            "requested": len(order),
            "encoded": sum((vectors / (r["id"] + ".npy")).exists() for r in order),
            "errors": sum(r["id"] in errors for r in order),
            "priority_only": args.only_priority,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inventory", "model-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prioritize", type=Path, nargs="*", default=[])
    parser.add_argument("--only-priority", action="store_true")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--batch", type=int, default=8)
    build(parser.parse_args())
