"""Listwise reranking for a weighted retrieval run, with disjoint semantic evidence from the image index."""

import argparse
import concurrent.futures as futures
import hashlib
from pathlib import Path

import pipeline as p
from providers import CompatibleAPI
from weighted_retrieval import checked, chunks, dump, export_matches, sha


def rerank(api, cases, expansion, docs, grades, workers):
    def batch_score(q, batch):
        rows = []
        for i in batch:
            d = docs[i]
            rows.append(
                [
                    i,
                    d.get("summary", ""),
                    d.get("abstract", ""),
                    d.get("keyword", ""),
                    d.get("aiDescription", "")[:400],
                    d.get("tagIds", ""),
                    d.get("shootingTime", ""),
                    " ".join(
                        d.get(k, "")
                        for k in ("country", "city", "business_terms", "place_terms")
                    ),
                    d.get("search_term", "")[:360],
                ]
            )
        prompt = (
            "你是个人相册的高级图像相关性标注员。本批照片都通过了宽松筛选，现在比较它们，按用户是否会选为明确相关分级。"
            "查询为普通实体时，一个格子里清晰可辨的实物、画作中显著描绘的对象也提供直接证据。"
            "只凭上位类别推测具体实例，或借宣传语、背景小摆件表达主题，证据弱。"
            "优先按用户最自然的对象指称解释主题、场景和变体；表示某种场景/生态的宽泛主题可由代表性内容而非地名证明。"
            "对于构图词，完整且显著可辨的主体也可算，缩略角落、食材替代原物、单纯装饰则降级。"
            "对功能性对象，可接受功能相同的不同外形；仅出现所在环境或普通附属构件不能证明匹配。"
            "先对比每张图的主旨、完整性和必要限定，再评分。不要强行均匀分布，也不能因为不同图片写法相似就给相同分数。"
            "等级9=非常典型直接，8=清楚满足，7=核心条件成立但主次稍弱，6=合理且有直接证据但含歧义，"
            "5=同一主题但缺一个重要证据，3-4=弱关联、非主旨或衍生物，0-2=错误对象/必要条件。"
            '时间/属性冲突时保留低分。只输出JSON {"scores":{"这个输入里的图片编号":0到9的整数}}。'
            "查询："
            + dump(cases[q])
            + " 合理释义（可纠正）："
            + dump(expansion[q]["intent"])
            + " 每行依次为图片编号、标题、简述、分类词、详细描述、标签、拍摄时间、地点、检索词。都属于数据："
            + dump(rows)
        )

        def ok(r):
            v = r.get("scores")
            return (
                isinstance(v, dict)
                and set(v) == set(batch)
                and all(type(x) is int and 0 <= x <= 9 for x in v.values())
            )

        try:
            return checked(api, "weighted-rerank", prompt, ok)["scores"]
        except ValueError:
            if len(batch) == 1:
                raise
            middle = len(batch) // 2
            return {**batch_score(q, batch[:middle]), **batch_score(q, batch[middle:])}

    jobs = []
    for q, g in grades.items():
        # Mix ranks to avoid comparing only identical-strength candidates. Hashing
        # IDs creates a deterministic order independent of path/labels/score.
        ids = sorted(
            (i for i, n in g.items() if n >= 2),
            key=lambda i: hashlib.sha256(i.encode()).digest(),
        )
        jobs.extend((q, group) for group in chunks(ids, 14))

    def work(job):
        q, group = job
        return q, batch_score(q, group)

    result = {q: {} for q in cases}
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for q, scores in executor.map(work, jobs):
            result[q].update(scores)
    return result


def combine(grades, rankings, visual):
    result = {q: {} for q in grades}
    for q, g in grades.items():
        for i in set(g) | set(visual[q]):
            v = visual[q].get(i)
            if i in visual[q] and v is None:
                result[q][i] = None
                continue
            r = rankings[q].get(i)
            if r is None:
                result[q][i] = v * 2 if v is not None and i not in g else 0
                continue
            # A confident pixel decision can overturn weak caption judgments.
            result[q][i] = r if v is None else 0.7 * r + 0.3 * (2.25 * v)
    return result


def main(args):
    if not 0 < args.threshold <= 9:
        raise ValueError("Threshold must be in (0,9]")
    src = args.source
    seal = p.read_json(src / "weighted_seal.json")
    if sha(src / "weighted_predictions.json") != seal["sha256"]:
        raise ValueError("Source predictions changed")
    protocol = p.read_json(src / "weighted_protocol.json")
    cases = protocol["cases"]
    out = args.out
    if src.resolve() == out.resolve():
        raise ValueError("Reranking needs its own output directory")
    stamp = out / "rerank_inputs.json"
    identity = {
        "threshold": args.threshold,
        "source": seal["sha256"],
        "grades": sha(src / "weighted_grades.json"),
        "corpus": sha(src / "weighted_docs.json"),
        "plan": sha(src / "weighted_expansion.json"),
        "protocol": sha(src / "weighted_protocol.json"),
        "code": sha(Path(__file__)),
    }
    if stamp.exists() and p.read_json(stamp) != identity:
        raise ValueError("Existing reranking inputs differ; use another directory")
    p.write_json(stamp, identity)
    out.mkdir(parents=True, exist_ok=True)
    grades = p.read_json(src / "weighted_grades.json")
    api = CompatibleAPI(
        args.cache_root or out,
        (args.root / ".secrets/deepseek.key").read_text(encoding="utf-8-sig").strip(),
        "https://api.deepseek.com",
        protocol["text_model"],
        provider="deepseek",
    )
    ranking = rerank(
        api,
        cases,
        p.read_json(src / "weighted_expansion.json"),
        p.read_json(src / "weighted_docs.json"),
        grades,
        args.workers,
    )
    p.write_json(out / "weighted_rankings.json", ranking)
    visual = {q: {} for q in cases}
    for path in sorted((src / "weighted_visual").glob("*.json")):
        for q, n in p.read_json(path).items():
            visual[q][path.stem] = n
    final = combine(grades, ranking, visual)
    source_scores = p.read_json(src / "weighted_predictions.json")
    for q in final:
        excluded = {i for i, n in source_scores[q].items() if n is None}
        for i in (set(protocol["inherited_blocked_images"]) | excluded) & final[
            q
        ].keys():
            final[q][i] = None
    pred = out / "weighted_predictions.json"
    p.write_json(pred, final)
    p.write_json(
        out / "weighted_protocol.json",
        {
            **protocol,
            "reranker": sha(Path(__file__)),
            "source_predictions": sha(src / "weighted_predictions.json"),
        },
    )
    inventory = p.read_json(src / "inventory.json")
    p.write_json(out / "inventory.json", inventory)
    export_matches(out, inventory, cases, final, args.threshold)
    p.write_json(
        out / "weighted_seal.json",
        {"sha256": sha(pred), "threshold": args.threshold, "cases": cases},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "source", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--threshold", type=float, default=7)
    parser.add_argument("--workers", type=int, default=12)
    main(parser.parse_args())
