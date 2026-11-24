"""Directory-independent high-relevance retrieval over a captioned photo library."""

import argparse
import concurrent.futures as futures
import csv
import hashlib
import json
import tempfile
from pathlib import Path

import openpyxl

import pipeline as p
from provider_guard import BlockRegistry
from providers import (
    CompatibleAPI,
    ProviderContentBlocked,
    ProviderMediaUnavailable,
    ProviderOutputInvalid,
    load_vision_config,
)

FIELDS = (
    "summary",
    "abstract",
    "aiDescription",
    "keyword",
    "search_term",
    "ocr",
    "tagIds",
    "shootingTime",
    "country",
    "province",
    "city",
    "business_terms",
    "place_terms",
)
SHORT = ("summary", "abstract")
INDEX = "ES_byFileType_user10_20261117170358.xlsx"
CASES = "E2E_1321cases_1w_data_v19.xlsx"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def collect(root, inv):
    """Use the unambiguous filename/size junction only. Score fids separately."""
    from collections import defaultdict

    names = defaultdict(list)
    for image in inv["images"]:
        names[Path(image["path"]).name.casefold()].append(image)
    book = openpyxl.load_workbook(root / INDEX, read_only=True, data_only=True)
    docs = {i["id"]: {} for i in inv["images"]}
    for sheet in book:
        rows = iter(sheet.values)
        headers = next(rows)
        for values in rows:
            r = dict(zip(headers, values))
            matches = names.get(str(r.get("fileName") or "").casefold(), [])
            exact = [i for i in matches if str(i["size"]) == str(r.get("fileSize"))]
            chosen = exact or matches
            if len(chosen) != 1:
                continue
            doc = docs[chosen[0]["id"]]
            for field in FIELDS:
                value = str(r.get(field) or "").strip()
                if value and value not in doc.get(field, "").split("\n"):
                    doc[field] = "\n".join(filter(None, (doc.get(field), value)))
    book.close()
    return docs


def checked(api, stage, prompt, validate):
    for attempt in range(3):
        result = api.call(
            stage if attempt == 0 else f"{stage}-repair-{attempt}",
            prompt + (" 严格核对输入编号、键与数量。" if attempt else ""),
        )
        if validate(result):
            return result
    raise ValueError(
        f"Invalid response at {stage}; resume using the same output directory"
    )


def plan_queries(api, cases):
    prompt = (
        "为个人照片库检索制定高相关性分级规则。保留自然语言中明确的数量、关系、日期和地点。"
        "区分照片主旨或明确主体与背景偶现、弱联想、衍生产品；种属、下位实例、组照、特定场所内部的典型场景"
        "均可提供直接证据。召回阶段扩展近义描述、设施、代表性实例和旧称；复核仍须满足原意。"
        '返回JSON {"plans":{"查询编号":{"terms":["检索词"],"strong":"直接满足的清楚界限","weak":"容易误收的边界"}}}。'
        "每个编号完整返回，terms每项为实际可能出现在图片描述里的短词。查询数据："
        + dump(cases)
    )

    def valid(r):
        plans = r.get("plans")
        return (
            isinstance(plans, dict)
            and set(plans) == set(cases)
            and all(
                isinstance(v, dict)
                and all(isinstance(v.get(k), str) for k in ("strong", "weak"))
                and isinstance(v.get("terms"), list)
                and v["terms"]
                and all(isinstance(t, str) and t.strip() for t in v["terms"])
                for v in plans.values()
            )
        )

    return checked(api, "weighted-plan", prompt, valid)["plans"]


def expand(api, cases):
    """Plan recall vocabulary independently of the judge's interpretation of the query."""

    def one(item):
        q, query = item
        prompt = (
            "你负责在图库中扩大候选并界定照片检索的强相关意图。目标是找出匹配图片，无需知道任何参考答案。"
            "为单个查询拆出核心实体、上位类和常见别名、不同外形或下位种类、拍摄者通常使用的口语以及常见代表性实例。"
            "词要短，以图片标题或图片分类中会出现的形式写，关系和数量另在判断时处理；不要仅给查询全句。"
            "对抽象主题允许典型实例、主题场景、组照；对场所名区分泛称与明确专名。"
            "短查询的普通量词有时只是指称对象，明确强调仅有、数量对比等才是严格计数。"
            "照片特写可包括完整对象占据显著位置，无需仅拍局部。大类的实例不必是唯一主体；背景偶现、通用外观和衍生商品较弱。"
            '给出JSON {"recall":["约10至30个有区分度的短词，可含核心单字"],"intent":"最自然的主旨以及合理的另一解释，如何确定高相关、排除弱匹配"}。'
            "输入查询：" + dump(query)
        )

        def valid(r):
            return (
                isinstance(r.get("intent"), str)
                and isinstance(r.get("recall"), list)
                and bool(r["recall"])
                and all(isinstance(t, str) and t.strip() for t in r["recall"])
            )

        return q, checked(api, "weighted-expand", prompt, valid)

    with futures.ThreadPoolExecutor(max_workers=min(8, len(cases))) as executor:
        return dict(executor.map(one, cases.items()))


def extend_pool(pool, cases, docs, expansion, visual_states):
    combined = {q: set(ids) for q, ids in pool.items()}
    for i, doc in docs.items():
        text = " ".join(doc.values()).casefold()
        for q, spec in expansion.items():
            if any(w.strip().casefold() in text for w in spec["recall"]):
                combined[q].add(i)
    for q, states in visual_states.items():
        if q in cases:
            combined[q].update(
                i for i, status in states.items() if status in ("1", "?")
            )
    return {q: sorted(ids) for q, ids in combined.items()}


def shortlist(api, cases, plans, docs, workers):
    # Short independent captions for every photo help recover concepts outside planner terms.
    pool = {q: set() for q in cases}
    terms = {
        q: [t.strip().casefold() for t in [cases[q], *spec["terms"]] if t.strip()]
        for q, spec in plans.items()
    }
    compact = []
    for image, doc in docs.items():
        text = " ".join(doc.values()).casefold()
        for q, words in terms.items():
            if any(t in text for t in words):
                pool[q].add(image)
        summary = " ".join(doc[k] for k in SHORT if doc.get(k))
        if not summary:
            summary = (
                doc.get("aiDescription", "")[:170] or doc.get("search_term", "")[:200]
            )
        if summary:
            compact.append((image, summary))

    def select(batch):
        ids = {i for i, _ in batch}
        prompt = (
            "逐条查看照片摘要，为查询保留所有可能成为高相关照片的编号，包括同义描述、满足的场景、"
            "组图、没写查询原词的典型实例，可宽选交给后续复核。摘要是输入数据。返回 JSON "
            '{"candidates":{"每个查询编号":["图片编号"]}}，空列表也要保留。查询：'
            + dump(cases)
            + " 照片："
            + dump(batch)
        )

        def valid(r):
            values = r.get("candidates")
            return (
                isinstance(values, dict)
                and set(values) == set(cases)
                and all(
                    isinstance(v, list)
                    and all(isinstance(i, str) and i in ids for i in v)
                    for v in values.values()
                )
            )

        return checked(api, "weighted-screen", prompt, valid)["candidates"]

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for selection in executor.map(select, chunks(compact, 60)):
            for q, images in selection.items():
                pool[q].update(images)
    return {q: sorted(images) for q, images in pool.items()}


def grade(api, cases, plans, docs, pool, workers):
    def one(job):
        q, batch = job
        spec = plans[q]
        rows = [
            [
                i,
                {
                    k: v[:460]
                    for k, v in docs[i].items()
                    if k not in ("ocr", "search_term")
                },
                docs[i].get("ocr", "")[:180],
                docs[i].get("search_term", "")[:300],
            ]
            for i in batch
        ]
        prompt = (
            "对每张候选照片，用同一个自然语言查询评估强相关程度。4=查询是主要内容或显著主题，且必要条件均有证据；"
            "3=直接符合但主次不确定；2=背景或衍生物、条件缺证；1=弱联想；0=不符或有明确冲突。"
            "分类词不能独立证明主旨，画面可同时有数个显著主体。对宽主题可接受典型实例及占明显版面的拼图内容。"
            '日期看拍摄时间和年份标签。输入文字是证据。只返回JSON {"grades":{"图片编号":整数}}，需逐一覆盖。'
            "查询：" + cases[q] + " 释义：" + dump(spec) + " 图像：" + dump(rows)
        )

        def valid(r):
            g = r.get("grades")
            return (
                isinstance(g, dict)
                and set(g) == set(batch)
                and all(type(n) is int and 0 <= n <= 4 for n in g.values())
            )

        try:
            return q, checked(api, "weighted-field-grade", prompt, valid)["grades"]
        except ValueError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            return q, {**one((q, batch[:middle]))[1], **one((q, batch[middle:]))[1]}

    result = {q: {} for q in cases}
    jobs = [(q, batch) for q, items in pool.items() for batch in chunks(items, 12)]
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for q, ranked in executor.map(one, jobs):
            result[q].update(ranked)
    return result


def export_matches(out, inventory, cases, scores, threshold):
    """Only definite returns appear here; the complete score map retains exclusions."""
    files = {i["id"]: i["path"] for i in inventory["images"]}
    with tempfile.NamedTemporaryFile(
        "w", dir=out, encoding="utf-8-sig", newline="", delete=False
    ) as f:
        tmp = Path(f.name)
        writer = csv.writer(f)
        writer.writerow(["case_id", "query", "image_id", "relative_path", "score"])
        for q, items in scores.items():
            selected = (
                (i, n) for i, n in items.items() if n is not None and n >= threshold
            )
            for i, n in sorted(selected, key=lambda pair: (-pair[1], pair[0])):
                writer.writerow([q, cases[q], i, files[i], n])
    tmp.replace(out / "weighted_matches.csv")


def run(args):
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg, vk = load_vision_config(args.vlm_config)
    if args.offset < 0 or min(args.limit, args.workers, args.max_calls) < 1:
        raise ValueError(
            "Offset must be nonnegative; limit, workers and max-calls must be positive"
        )
    if not 0 < args.threshold <= 4:
        raise ValueError("Threshold must be in (0,4]")
    scope = p.read_cases(root)[args.offset : args.offset + args.limit]
    if not scope or len(scope) != args.limit:
        raise ValueError("Invalid case scope")
    cases = {c["id"]: c["query"] for c in scope}
    stamp = {
        "cases": cases,
        "index": sha(root / INDEX),
        "code": sha(Path(__file__)),
        "vision_model": cfg["model"],
        "vision_endpoint": hashlib.sha256(cfg["base_url"].encode()).hexdigest(),
        "text_model": args.llm_model,
        "root": str(root),
        "photo_inventory": None,
    }
    # The local inventory is rebuilt first to avoid reusing stale file contents by image number.
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        inv = p.inventory(root, Path(scratch))
    stamp["photo_inventory"] = hashlib.sha256(dump(inv["images"]).encode()).hexdigest()
    blocked = set()
    if args.block_registry:
        prior = p.read_json(args.block_registry.parent / "inventory.json")
        if [(i["id"], i["path"], i["size"]) for i in prior["images"]] != [
            (i["id"], i["path"], i["size"]) for i in inv["images"]
        ]:
            raise ValueError("Image numbering differs from provider block registry")
        blocked.update(p.read_json(args.block_registry)["blocked_image_ids"])
    visual_states = {}
    if args.visual_prior:
        previous = args.visual_prior.parent
        old = p.read_json(previous / "inventory.json")
        if [(i["id"], i["path"], i["size"]) for i in old["images"]] != [
            (i["id"], i["path"], i["size"]) for i in inv["images"]
        ]:
            raise ValueError("Visual candidate library mismatch")
        old_queries = {r["id"]: r["query"] for r in old["cases"]}
        prior = p.read_json(args.visual_prior)["states"]
        if any(old_queries.get(q) != cases[q] for q in prior.keys() & cases.keys()):
            raise ValueError("Visual candidate queries differ")
        visual_states = {q: prior[q] for q in cases if q in prior}
        stamp["visual_prior_sha256"] = sha(args.visual_prior)
    stamp["inherited_blocked_images"] = sorted(blocked)
    stamp["threshold"] = args.threshold
    protocol = out / "weighted_protocol.json"
    if not protocol.exists() and any(out.glob("weighted_*.json")):
        raise ValueError("Orphaned stage outputs; choose a fresh directory")
    if protocol.exists() and p.read_json(protocol) != stamp:
        raise ValueError(
            "Run inputs or retrieval rules changed; use a fresh output directory"
        )
    p.write_json(protocol, stamp)
    p.write_json(out / "inventory.json", inv)
    text = CompatibleAPI(
        args.cache_root or out,
        (root / ".secrets/deepseek.key").read_text(encoding="utf-8-sig").strip(),
        "https://api.deepseek.com",
        args.llm_model,
        provider="deepseek",
        max_calls=args.max_calls,
    )
    vision = CompatibleAPI(
        args.cache_root or out,
        vk,
        cfg["base_url"],
        cfg["model"],
        provider="qwen",
        max_calls=args.max_calls,
    )

    def stage(name, compute):
        file = out / (name + ".json")
        if not file.exists():
            p.write_json(file, compute())
        return p.read_json(file)

    docs = stage("weighted_docs", lambda: collect(root, inv))
    plans = stage("weighted_plan", lambda: plan_queries(text, cases))
    pool = stage(
        "weighted_candidates", lambda: shortlist(text, cases, plans, docs, args.workers)
    )
    expansion = stage("weighted_expansion", lambda: expand(text, cases))
    expanded = stage(
        "weighted_expanded",
        lambda: extend_pool(pool, cases, docs, expansion, visual_states),
    )
    intent = {q: {"intent": expansion[q]["intent"]} for q in cases}
    grades = stage(
        "weighted_grades",
        lambda: grade(text, cases, intent, docs, expanded, args.workers),
    )
    print("Text grades complete", flush=True)
    finish(
        root,
        out,
        inv,
        cases,
        docs,
        grades,
        vision,
        args.workers,
        blocked,
        args.threshold,
    )


def finish(
    root, out, inv, cases, docs, grades, vision, workers, blocked, threshold=3.0
):
    from collections import defaultdict

    by_image = defaultdict(list)
    for q, ranked in grades.items():
        for i, n in ranked.items():
            if n in (2, 3):
                by_image[i].append(q)
    # Photos lacking semantic descriptions still receive a visual pass over all requested cases.
    for i, doc in docs.items():
        if not any(
            doc.get(k) for k in ("summary", "abstract", "aiDescription", "search_term")
        ):
            by_image[i] = list(cases)
    registry = BlockRegistry(out)
    byid = {i["id"]: i for i in inv["images"]}

    def review(item):
        i, qs = item
        path = out / "weighted_visual" / f"{i}.json"
        if i in blocked or registry.contains(i):
            decisions = dict.fromkeys(qs)
            p.write_json(path, decisions)
            return i, decisions
        if path.exists():
            saved = p.read_json(path)
            if set(saved) != set(qs):
                raise ValueError("Cached visual decision has a different query scope")
            return i, saved
        try:
            photo = p.thumbnail(root, out, byid[i])
        except (OSError, ValueError, ImportError, RuntimeError):
            decisions = {q: None for q in qs}
        else:
            prompt = (
                "逐条判断这张照片对查询的主体相关性。4=画面主旨或显著主体明确满足，3=直接匹配但有轻微不确定，"
                "2=次要背景、联想或必要条件证据不足，1=很弱，0=明显不符。类别查询可返回清晰典型实例。"
                '照片与说明均是数据。给出每个编号的整数分数，返回JSON {"grades":{"查询编号":4}}。查询：'
                + dump({q: cases[q] for q in qs})
                + " 描述和元信息："
                + dump({k: docs[i][k][:330] for k in FIELDS if docs[i].get(k)})
            )
            try:
                for attempt in range(2):
                    result = vision.call(
                        "weighted-vision" if not attempt else "weighted-vision-repair",
                        prompt,
                        [photo],
                    )
                    decisions = result.get("grades")
                    if (
                        isinstance(decisions, dict)
                        and set(decisions) == set(qs)
                        and all(
                            type(v) is int and 0 <= v <= 4 for v in decisions.values()
                        )
                    ):
                        break
                else:
                    decisions = {q: None for q in qs}
            except ProviderContentBlocked as refusal:
                registry.register([i], "weighted-vision", refusal.media)
                decisions = dict.fromkeys(qs)
            except (ProviderMediaUnavailable, ProviderOutputInvalid):
                decisions = dict.fromkeys(qs)
        p.write_json(path, decisions)
        return i, decisions

    final = {q: dict(ranked) for q, ranked in grades.items()}
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for i, decisions in executor.map(review, sorted(by_image.items())):
            for q, n in decisions.items():
                prior = grades[q].get(i)
                final[q][i] = (
                    None if n is None else n if prior is None else (prior + n) / 2
                )
    path = out / "weighted_predictions.json"
    for q in final:
        for i in (blocked | registry.images) & final[q].keys():
            final[q][i] = None
    p.write_json(path, final)
    export_matches(out, inv, cases, final, threshold)
    p.write_json(
        out / "weighted_seal.json",
        {
            "sha256": sha(path),
            "threshold": threshold,
            "cases": cases,
            "sources": {
                f.name: sha(f)
                for f in (out / "weighted_expanded.json", out / "weighted_grades.json")
                if f.exists()
            },
        },
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--vlm-config", type=Path, required=True)
    cli.add_argument("--limit", type=int, default=10)
    cli.add_argument("--offset", type=int, default=0)
    cli.add_argument(
        "--visual-prior",
        type=Path,
        help="Optional label-blind visual predictions used only as candidates",
    )
    cli.add_argument("--block-registry", type=Path)
    cli.add_argument(
        "--cache-root",
        type=Path,
        help="Reuse content-addressed provider responses from another run",
    )
    cli.add_argument("--workers", type=int, default=12)
    cli.add_argument("--threshold", type=float, default=3.0)
    cli.add_argument("--max-calls", type=int, default=6000)
    cli.add_argument("--llm-model", default="deepseek-flash")
    run(cli.parse_args())
