"""Score sealed predictions against the *named* high-weight column. Extra outputs cost one each."""

import argparse
import collections
import re
from pathlib import Path

import openpyxl
import pipeline as p
from weighted_retrieval import CASES, INDEX, sha


def label_sets(root, cases, header="高权重预期fid"):
    book = openpyxl.load_workbook(root / CASES, read_only=True, data_only=True)
    try:
        rows = iter(book.worksheets[0].values)
        headers = [str(x or "").strip().casefold() for x in next(rows)]
        if headers.count(header.casefold()) != 1:
            raise ValueError(f"Missing or duplicate label column {header}")
        col = headers.index(header.casefold())
        result = {}
        for r in rows:
            q = str(r[0])
            if q not in cases:
                continue
            if q in result or str(r[1]) != cases[q]:
                raise ValueError("Case ID/query mismatch in reference workbook")
            result[q] = {
                t.strip()
                for t in re.split(r"[,，;；、\s]+", str(r[col] or ""))
                if t.strip()
            }
        if set(result) != set(cases):
            raise ValueError("Missing reference rows")
        return result
    finally:
        book.close()


def file_ids(root, inv):
    names = collections.defaultdict(list)
    for i in inv["images"]:
        names[Path(i["path"]).name.casefold()].append(i)
    result = collections.defaultdict(set)
    book = openpyxl.load_workbook(root / INDEX, read_only=True, data_only=True)
    for sheet in book:
        rows = iter(sheet.values)
        heads = next(rows)
        for values in rows:
            r = dict(zip(heads, values))
            fid = str(r.get("fid") or "").strip()
            if not fid:
                continue
            matches = names.get(str(r.get("fileName") or "").casefold(), [])
            exact = [i for i in matches if str(i["size"]) == str(r.get("fileSize"))]
            if len(exact or matches) == 1:
                result[fid].add((exact or matches)[0]["id"])
    book.close()
    # A local file representing several fids is still one returned image. Give it
    # at most one hit; unknown, missing and unresolved references stay in the denominator.
    return {f: next(iter(images)) for f, images in result.items() if len(images) == 1}


def metrics(tp, fp, fn):
    rate = lambda a, b: a / b if b else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": rate(tp, tp + fp),
        "coverage": rate(tp, tp + fn),
        "f1": rate(2 * tp, 2 * tp + fp + fn),
    }


def measure(labels, mapping, outputs):
    cases = {}
    for q, refs in labels.items():
        hits = {mapping[f] for f in refs if f in mapping} & outputs[q]
        cases[q] = {
            **metrics(len(hits), len(outputs[q] - hits), len(refs) - len(hits)),
            "unmapped_fids": sum(f not in mapping for f in refs),
            "reference_fids": len(refs),
        }
    # A blank positive-only list is unscorable; keep its returned count visible.
    labeled = [r for r in cases.values() if r["reference_fids"]]
    totals = metrics(*(sum(r[k] for r in labeled) for k in ("tp", "fp", "fn")))
    totals.update(
        labeled_cases=len(labeled),
        unscored_cases=len(cases) - len(labeled),
        hit_cases=sum(r["tp"] > 0 for r in labeled),
    )
    return {"total": totals, "cases": cases}


def score_legacy(root, out, header):
    path = out / "predictions.json"
    seal = p.read_json(out / "predictions_seal.json")
    if sha(path) != seal["sha256"]:
        raise ValueError("Legacy predictions changed")
    inv = p.read_json(out / "inventory.json")
    states = p.read_json(path)["states"]
    cases = {c["id"]: c["query"] for c in inv["cases"] if c["id"] in states}
    if cases.keys() != states.keys():
        raise ValueError("Legacy prediction scope mismatch")
    mapping = file_ids(root, inv)
    labels = label_sets(root, cases, header)
    results = {
        "reference_header": header,
        "prediction_sha256": seal["sha256"],
        "final": measure(
            labels,
            mapping,
            {q: {i for i, v in s.items() if v == "1"} for q, s in states.items()},
        ),
    }
    p.write_json(out / "weighted_scores.json", results)
    print(results["final"]["total"])


def score(root, out, header):
    protocol = p.read_json(out / "weighted_protocol.json")
    if sha(root / INDEX) != protocol["index"]:
        raise ValueError("Image index differs from the frozen run")
    seal = p.read_json(out / "weighted_seal.json")
    path = out / "weighted_predictions.json"
    if sha(path) != seal["sha256"] or seal["cases"] != protocol["cases"]:
        raise ValueError("Sealed predictions changed")
    grades = p.read_json(path)
    if set(grades) != set(seal["cases"]):
        raise ValueError("Incomplete query coverage")
    mapping = file_ids(root, p.read_json(out / "inventory.json"))
    labels = label_sets(root, seal["cases"], header)
    outputs = {
        q: {i for i, n in items.items() if n is not None and n >= seal["threshold"]}
        for q, items in grades.items()
    }
    results = {
        "reference_header": header,
        "threshold": seal["threshold"],
        "prediction_sha256": seal["sha256"],
        "final": measure(labels, mapping, outputs),
        "candidate": measure(
            labels, mapping, {q: set(items) for q, items in grades.items()}
        ),
    }
    p.write_json(out / "weighted_scores.json", results)
    print(results["final"]["total"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--legacy", action="store_true", help="Score a sealed V1/V2 batch directory"
    )
    parser.add_argument("--label-header", default="高权重预期fid")
    args = parser.parse_args()
    (score_legacy if args.legacy else score)(args.root, args.out, args.label_header)
