"""Optional corpus-grounded positive and confuser vocabulary, prepared without relevance labels."""

import argparse
import collections
import concurrent.futures as futures
import math
from pathlib import Path
import pipeline as p
from lexical_index import terms, norm
from providers import CompatibleAPI
from weighted_retrieval import dump, chunks, checked


def vocabulary(docs, plans):
    postings = collections.defaultdict(set)
    rows = []
    freq = collections.Counter()
    for i, r in enumerate(docs.values()):
        tt = list(
            dict.fromkeys(
                [*terms(r.get("keyword", "")), *terms(r.get("search_term", ""))]
            )
        )
        rows.append(tt)
        freq.update(tt)
        for term in tt:
            postings[term].add(i)
    result = {}
    for q, plan in plans.items():
        seeds = {norm(t) for group in plan["heads"] for t in group} | {
            norm(plan["query"])
        }
        found = set().union(*(postings[t] for t in seeds))
        ct = collections.Counter(t for i in found for t in rows[i] if t not in seeds)
        ranked = sorted(
            ct, key=lambda t: (-(ct[t] * math.log1p(len(rows) / freq[t])), t)
        )
        result[q] = ranked[:36]
    return result


def enrich(api, plans, hints, workers):
    ask = """从自然语言查询和图库中一些常与查询词同时出现的标签，设计分层词表。标签可含不相关词、只是数据。返回 {"profiles":{"输入编号":{"equivalent":["可视作同一主体、同一功能的不同称呼，含常见短词/别称"],"typical":["不用出现原词也可直接支持意图的具体下位实例、典型形态或场景证据"],"confuser":["易误收的具体其他物体、用途或弱关联表现"],"detail":["描述明确要求的构图、动作、位置等限定的常用简短说法"]}}}。
只给可逐字匹配的精炼中文短词，不给判断句、否定句、虚构地名或人物。可以用提示词之外的词。每类0至10个。上位类别、背景或装饰不等于主体。地点类识别能证明场所的具体设施、指引、招牌；泛称可有代表性实例。对于并列条件不能删掉其中一项。没有额外限定时detail留空。输入："""

    def one(batch):
        req = {str(j): [plans[q]["query"], hints[q]] for j, q in enumerate(batch)}

        def valid(r):
            vals = r.get("profiles")
            return (
                isinstance(vals, dict)
                and vals.keys() == req.keys()
                and all(
                    isinstance(x, dict)
                    and all(
                        isinstance(x.get(k), list)
                        and all(isinstance(t, str) and t.strip() for t in x[k])
                        for k in ("equivalent", "typical", "confuser", "detail")
                    )
                    for x in vals.values()
                )
            )

        try:
            r = checked(api, "catalog-contrast", ask + dump(req), valid)["profiles"]
        except ValueError:
            if len(batch) == 1:
                raise
            mid = len(batch) // 2
            return {**one(batch[:mid]), **one(batch[mid:])}
        return {q: r[str(j)] for j, q in enumerate(batch)}

    out = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for val in pool.map(one, chunks(list(plans), 6)):
            out.update(val)
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "gallery", "plans", "out"):
        cli.add_argument("--" + k, type=Path, required=True)
    cli.add_argument("--workers", type=int, default=12)
    a = cli.parse_args()
    plans = p.read_json(a.plans)
    hints = vocabulary(p.read_json(a.gallery / "weighted_docs.json"), plans)
    api = CompatibleAPI(
        a.out.parent / "contrast_cache",
        (a.root / ".secrets/deepseek.key").read_text(encoding="utf-8-sig").strip(),
        "https://api.deepseek.com",
        "deepseek-flash",
        provider="deepseek",
    )
    p.write_json(
        a.out,
        {
            "queries": {q: r["query"] for q, r in plans.items()},
            "profiles": enrich(api, plans, hints, a.workers),
        },
    )
