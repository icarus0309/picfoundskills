"""Build a gallery from the source workbooks, optionally caching reusable visual facts per photo."""

import argparse
import concurrent.futures as futures
import hashlib
import json
from pathlib import Path

import pipeline as p
from provider_guard import BlockRegistry
from providers import (
    CompatibleAPI,
    ProviderContentBlocked,
    ProviderMediaUnavailable,
    ProviderOutputInvalid,
    load_vision_config,
)
from weighted_retrieval import collect, sha, INDEX

FACT_KEYS = ("subjects", "background", "setting", "actions", "composition", "words")
PROMPT = """单独观察这张照片，为相册建立可复用的中文事实索引。只描述可见内容。返回JSON，下面每项为1至8个简短、便于检索的短语，缺乏证据则空数组：subjects主要物体/动物的具体种类和俗称；background仅在背景、图案或拼贴小格里出现的对象；setting场景及可见支持证据；actions谁对谁做什么、主体与环境的空间关系；composition景别、主体大小和表现方式（活体、雕像、图案、菜肴等）；words能清楚读出的标牌原文。无法从视觉确定的具体场馆和拍摄地点不得猜测。仅输出对象，键为subjects,background,setting,actions,composition,words。"""


def signature(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def family_inventory(inv):
    return signature(
        [[r["id"], r["path"], r["size"], r["mtime_ns"]] for r in inv["images"]]
    )


def prepare(root, out):
    out.mkdir(parents=True, exist_ok=True)
    # Build the fresh inventory in scratch space before replacing existing outputs.
    import tempfile

    with tempfile.TemporaryDirectory(dir=out) as tmp:
        inv = p.inventory(root, Path(tmp), include_cases=False)
    ident = {"inventory": family_inventory(inv), "index": sha(root / INDEX)}
    meta = out / "gallery.json"
    if meta.exists():
        old = p.read_json(meta)
        if any(old.get(k) != v for k, v in ident.items()):
            raise ValueError("Gallery has changed; use a new directory")
        if family_inventory(p.read_json(out / "inventory.json")) != old["inventory"]:
            raise ValueError("Stored inventory changed")
        if sha(out / "weighted_docs.json") != old["docs_sha256"]:
            raise ValueError("Gallery documents changed")
        return
    p.write_json(out / "inventory.json", inv)
    p.write_json(out / "weighted_docs.json", collect(root, inv))
    p.write_json(meta, {**ident, "docs_sha256": sha(out / "weighted_docs.json")})


def extract(args):
    out = args.out
    inv = p.read_json(out / "inventory.json")
    docs = p.read_json(out / "weighted_docs.json")
    config, key = load_vision_config(args.vlm_config)
    api = CompatibleAPI(
        out / "fact_cache",
        key,
        config["base_url"],
        config["model"],
        provider="qwen",
        max_calls=args.max_calls,
    )
    registry = BlockRegistry(out)
    inherited = set()
    for file in args.block_registry:
        if family_inventory(
            p.read_json(file.parent / "inventory.json")
        ) != family_inventory(inv):
            raise ValueError("Block registry uses a different gallery")
        inherited.update(p.read_json(file)["blocked_image_ids"])
    store = out / "fact_rows"
    store.mkdir(exist_ok=True)
    spec = signature(
        [
            family_inventory(inv),
            sha(out / "weighted_docs.json"),
            config["base_url"],
            config["model"],
            PROMPT,
        ]
    )
    stamp = out / "facts_protocol.json"
    if stamp.exists() and p.read_json(stamp)["signature"] != spec:
        raise ValueError("Changed fact extractor; use a fresh gallery")
    p.write_json(
        stamp, {"signature": spec, "focus": args.focus, "model": config["model"]}
    )

    def wanted(row):
        d = docs[row["id"]]
        brief = " ".join(d.get(k, "") for k in ("summary", "abstract"))
        return (
            not d.get("summary")
            or not d.get("aiDescription")
            or any(term in brief for term in args.focus)
            or any(term in brief for term in ("拼贴", "九宫格", "组图"))
        )

    def one(row):
        i = row["id"]
        file = store / (i + ".json")
        if i in inherited or registry.contains(i):
            p.write_json(file, {"status": "blocked"})
            return
        if file.exists():
            return
        try:
            photo = p.thumbnail(args.root, out, row)
        except (ValueError, OSError, RuntimeError, ImportError):
            p.write_json(file, {"status": "decode_failed"})
            return
        try:
            for attempt in range(2):
                result = api.call(
                    "catalog-facts" if not attempt else "catalog-facts-repair",
                    PROMPT,
                    [photo],
                )
                if set(result) == set(FACT_KEYS) and all(
                    isinstance(v, list) and all(isinstance(t, str) for t in v)
                    for v in result.values()
                ):
                    p.write_json(file, {"status": "ok", "fields": result})
                    return
            p.write_json(file, {"status": "schema_abstention"})
        except ProviderContentBlocked as exc:
            registry.register([i], "catalog-facts", exc.media)
            p.write_json(file, {"status": "blocked"})
        except (ProviderMediaUnavailable, ProviderOutputInvalid):
            p.write_json(file, {"status": "provider_abstention"})

    jobs = [r for r in inv["images"] if wanted(r)]
    print("Selected for offline indexing:", len(jobs), flush=True)
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(one, jobs):
            pass
    facts = {r["id"]: p.read_json(store / (r["id"] + ".json")) for r in jobs}
    p.write_json(out / "facts.json", facts)
    p.write_json(
        out / "facts_manifest.json",
        {
            "gallery": p.read_json(out / "gallery.json"),
            "facts_sha256": sha(out / "facts.json"),
            "extractor": spec,
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--vlm-config", type=Path)
    cli.add_argument("--focus", nargs="*", default=[])
    cli.add_argument("--block-registry", nargs="*", type=Path, default=[])
    cli.add_argument("--workers", type=int, default=12)
    cli.add_argument("--max-calls", type=int, default=6000)
    args = cli.parse_args()
    prepare(args.root, args.out)
    if args.vlm_config:
        extract(args)
