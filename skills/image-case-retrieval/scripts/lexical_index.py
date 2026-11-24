"""Inverted index and field-evidence features. All rows are derived from gallery data and query plans."""

import collections
import datetime
import functools
import json
import math
import re
import unicodedata


import numpy as np

FIELDS = (
    "title",
    "abstract",
    "lead",
    "description",
    "keyword",
    "tag",
    "search",
    "ocr",
    "place",
    "subject",
    "background",
    "scene",
    "action",
    "composition",
)
EXACT = ("keyword", "tag", "search", "subject")
COLS = FIELDS + tuple("exact_" + f for f in EXACT) + ("first_keyword",)
TI = FIELDS.index("title")
LOC = FIELDS.index("place")
BG = FIELDS.index("background")
PRIMARY = FIELDS.index("subject")
KINDS = ("object", "venue", "scene", "compound", "text", "metadata")


def norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(s)).casefold())


def terms(s):
    out = []
    for line in str(s).splitlines():
        try:
            v = json.loads(line) if line.lstrip().startswith("[") else None
        except ValueError:
            v = None
        out.extend(v if isinstance(v, list) else re.split(r"[,，;；、|]", line))
    return [norm(t) for t in out if isinstance(t, str) and norm(t)]


def grams(s):
    return [s[i : i + 2] for i in range(len(s) - 1) if "\n" not in s[i : i + 2]]


def doc_fields(d, f):
    fact = f.get("fields", {}) if f.get("status") == "ok" else {}
    desc = norm(d.get("aiDescription", ""))
    clauses = re.split(r"[。；！？]", desc)
    lead = clauses[0][:110]
    back = [
        c
        for c in re.split(r"[，。；]", desc)
        if re.search("背景|角落|远处|远景|边缘", c)
    ]
    field_tokens = {
        "keyword": terms(d.get("keyword", "")),
        "tag": terms(d.get("tagIds", "")),
        "search": terms(d.get("search_term", "")),
        "subject": [norm(x) for x in fact.get("subjects", [])],
    }
    fields = dict(
        title=norm(d.get("summary", "")),
        abstract=norm(d.get("abstract", "")),
        lead=lead,
        description=desc,
        ocr=norm(d.get("ocr", "")) + "\n" + "\n".join(map(norm, fact.get("words", []))),
        place="\n".join(
            norm(d.get(k, ""))
            for k in ("country", "province", "city", "business_terms", "place_terms")
        ),
        background="\n".join([*back, *map(norm, fact.get("background", []))]),
        scene="\n".join(map(norm, fact.get("setting", []))),
        action="\n".join(map(norm, fact.get("actions", []))),
        composition="\n".join(map(norm, fact.get("composition", []))),
    )
    fields.update({k: "\n".join(v) for k, v in field_tokens.items()})
    years = set(re.findall(r"(?:19|20)\d{2}(?=年)", fields["tag"]))
    year = re.match(r"(?:19|20)\d{2}", str(d.get("shootingTime", "")))
    if year:
        years.add(year[0])
    return (
        tuple(fields[k] for k in FIELDS),
        tuple(set(field_tokens[k]) for k in EXACT),
        set(field_tokens["keyword"][:2]),
        years,
    )


class LexicalIndex:
    def __init__(self, docs, facts=None):
        facts = facts or {}
        self.ids = sorted(docs)
        self.fields = []
        self.exact = []
        self.first = []
        self.years = []
        lens = []
        bygram = collections.defaultdict(list)
        freq = collections.defaultdict(list)
        bychar = collections.defaultdict(list)
        for i, k in enumerate(self.ids):
            fields, exact, first, years = doc_fields(docs[k], facts.get(k, {}))
            self.fields.append(fields)
            self.exact.append(exact)
            self.first.append(first)
            self.years.append(years)
            text = "\n".join(fields)
            counts = collections.Counter(grams(text))
            lens.append(sum(counts.values()))
            for g, n in counts.items():
                bygram[g].append(i)
                freq[g].append(n)
            for c in set(text) - {"\n"}:
                bychar[c].append(i)
        self.post = {
            g: (np.array(v, dtype=np.int32), np.array(freq[g], dtype=np.float32))
            for g, v in bygram.items()
        }
        self.uni = {g: np.array(v, dtype=np.int32) for g, v in bychar.items()}
        self.length = np.array(lens, dtype=np.float32)
        self.avlen = max(1, self.length.mean())
        # Neighbours share a populated city and their recorded capture times differ by at most 30 minutes.
        events = collections.defaultdict(list)
        for i, k in enumerate(self.ids):
            city = norm(docs[k].get("city", ""))
            try:
                moment = datetime.datetime.fromisoformat(
                    str(docs[k].get("shootingTime", ""))
                ).timestamp()
            except (ValueError, OverflowError, OSError):
                continue
            if city:
                events[city].append((moment, i))
        self.neighbours = np.full((len(self.ids), 4), -1, np.int32)
        for event in events.values():
            event.sort()
            for j, (time, i) in enumerate(event):
                near = sorted(
                    (abs(tm - time), i2)
                    for tm, i2 in event[max(0, j - 4) : j + 5]
                    if i2 != i and abs(tm - time) <= 1800
                )
                for ix, (_, other) in enumerate(near[:4]):
                    self.neighbours[i, ix] = other
        self.tag_positions = collections.defaultdict(list)
        for i, f in enumerate(self.fields):
            for j, tag in enumerate(
                dict.fromkeys(f[FIELDS.index("search")].split("\n"))
            ):
                if tag:
                    self.tag_positions[tag].append((i, 1 / (1 + j)))
        self.match = functools.lru_cache(768)(self._match)

    def _match(self, t):
        if not t:
            return np.empty(0, np.int32), np.empty((0, len(COLS)), np.float32)
        if len(t) == 1:
            pool = self.uni.get(t, ())
        else:
            tokens = set(grams(t))
            if not tokens or any(g not in self.post for g in tokens):
                return np.empty(0, np.int32), np.empty((0, len(COLS)), np.float32)
            rare = min(tokens, key=lambda g: len(self.post[g][0]))
            pool = self.post[rare][0]
        ids = []
        rows = []
        for i in pool:
            f = self.fields[i]
            hits = [t in x for x in f]
            if any(hits):
                ids.append(i)
                rows.append(
                    hits + [t in v for v in self.exact[i]] + [t in self.first[i]]
                )
        return np.asarray(ids, dtype=np.int32), np.asarray(
            rows, dtype=np.float32
        ).reshape(-1, len(COLS))

    def evidence(self, ids, plan):
        output = {}
        for term in dict.fromkeys(
            [t for g in plan["heads"] + plan["facets"] + plan["places"] for t in g]
            + plan["examples"]
            + plan["context"]
        ):
            t = norm(term)
            if t:
                fields = [name for name, s in zip(FIELDS, self.fields[ids]) if t in s]
                if fields:
                    output[term] = fields
        return output

    def features(self, plan):
        n = len(self.ids)
        names = []
        data = []

        def add(name, arr):
            names.append(name)
            data.append(np.broadcast_to(arr, n))

        def union(words):
            a = np.zeros((n, len(COLS)), np.float32)
            for t in set(map(norm, words)):
                ids, rows = self.match(t)
                a[ids] = np.maximum(a[ids], rows)
            return a

        heads = plan["heads"]
        facets = plan["facets"]
        places = plan["places"]
        literal = union([plan["query"]])
        direct = union([g[0] for g in heads])
        aliases = union([t for g in heads for t in g[1:]])
        variants = union(plan["examples"])
        context = union(plan["context"])
        combined = np.maximum(direct, aliases)
        for label, mat in (
            ("literal", literal),
            ("head", direct),
            ("alias", aliases),
            ("instance", variants),
        ):
            for k, col in enumerate(COLS):
                add(label + "_" + col, mat[:, k])
        for prefix, groups in (
            ("object", heads),
            ("facet", facets),
            ("location", places),
        ):
            group_hits = []
            important = []
            metadata = []
            for g in groups:
                m = union(g)
                group_hits.append(m.max(axis=1))
                important.append(m[:, [0, 1, 2, PRIMARY]].max(axis=1))
                metadata.append(m[:, LOC])
            for title, vals in (
                ("anyfield", group_hits),
                ("prominent", important),
                ("placefield", metadata),
            ):
                v = np.stack(vals) if vals else np.zeros((1, n), np.float32)
                add(prefix + "_" + title + "_fraction", v.mean(axis=0))
                add(prefix + "_" + title + "_all", v.min(axis=0))
        add("associated_context", context.max(axis=1))
        add("context_prominent", context[:, [0, 1, PRIMARY, LOC]].max(axis=1))
        add("head_supported_fields", combined[:, : len(FIELDS)].sum(axis=1))
        add(
            "background_only",
            combined[:, BG] * (1 - combined[:, [0, 1, PRIMARY]].max(axis=1)),
        )
        add(
            "has_visual_facts",
            np.array(
                [bool(f[PRIMARY] or f[FIELDS.index("scene")]) for f in self.fields],
                np.float32,
            ),
        )
        add(
            "has_description",
            np.array([bool(f[0] or f[1] or f[3]) for f in self.fields], np.float32),
        )
        needed = set(map(str, plan["years"]))
        year_good = np.array([bool(needed & s) for s in self.years], np.float32)
        add("year_support", year_good)
        add(
            "year_conflict",
            np.array(
                [bool(needed and s and not s <= needed) for s in self.years], np.float32
            ),
        )
        query = norm(plan["query"])
        scores = np.zeros(n, np.float32)
        overlap = np.zeros(n, np.float32)
        tokens = set(grams(query))
        for g in sorted(tokens):
            if g not in self.post:
                continue
            ids, tf = self.post[g]
            idf = math.log1p((n - len(ids) + 0.5) / (len(ids) + 0.5))
            scores[ids] += (
                idf
                * tf
                * 2.2
                / (tf + 1.2 * (0.25 + 0.75 * self.length[ids] / self.avlen))
            )
            overlap[ids] += 1
        ratio = overlap / max(1, len(tokens))
        add("bm25", np.log1p(scores))
        add("query_bigram_coverage", ratio)
        pool = (
            (literal.max(axis=1) > 0)
            | (combined.max(axis=1) > 0)
            | (variants.max(axis=1) > 0)
            | (context.max(axis=1) > 0)
        )
        for g in places:
            pool |= union(g).max(axis=1) > 0
        if needed and not heads and not places:
            pool |= year_good > 0
        pool |= (ratio >= 0.50) & (overlap >= 2)
        if "contrast" in plan:
            expansions = plan["contrast"]
            strong_cols = [
                0,
                1,
                PRIMARY,
                COLS.index("first_keyword"),
                COLS.index("exact_search"),
            ]
            union_strength = np.zeros(n, np.float32)
            for label in ("equivalent", "typical", "confuser", "detail"):
                words = expansions[label]
                mat = union(words)
                for col in (
                    "title",
                    "abstract",
                    "lead",
                    "keyword",
                    "exact_search",
                    "subject",
                    "background",
                    "scene",
                    "place",
                    "action",
                    "composition",
                    "first_keyword",
                ):
                    add(label + "_" + col, mat[:, COLS.index(col)])
                if label in ("equivalent", "typical"):
                    pool |= mat.max(axis=1) > 0
                    union_strength = np.maximum(
                        union_strength, mat[:, strong_cols].mean(axis=1)
                    )
            focus = set(norm(t) for g in heads for t in g)
            extras = set(norm(t) for t in expansions["equivalent"])
            for label, words in (("focus", focus), ("alternative", extras)):
                positions = np.zeros(n, np.float32)
                for word in words:
                    for i, weight in self.tag_positions.get(word, ()):
                        positions[i] = max(positions[i], weight)
                add(label + "_search_position", positions)
            strength = (
                combined[:, strong_cols].sum(axis=1)
                + literal[:, strong_cols].sum(axis=1) * 0.5
                + union_strength
                + ratio * 0.25
            )
            near = np.r_[strength, 0][self.neighbours]
            nearby = near.max(axis=1)
            add("nearby_strength", nearby)
            add("nearby_mean", near.mean(axis=1))
            if plan["kind"] in ("venue", "scene", "metadata"):
                pool |= nearby >= 2.5
            for key, a in (("salience", strength), ("lexical", scores)):
                ordered = np.sort(a)
                add(
                    key + "_rank",
                    np.log1p(n - np.searchsorted(ordered, a, side="right")),
                )
                add(key + "_relative", a / max(0.01, float(a.max())))
            add("candidate_log_size", np.log1p(pool.sum()))
            add(
                "literal_exact_frequency",
                np.log1p(literal[:, COLS.index("exact_search")].sum()),
            )
            add("salient_frequency", np.log1p((strength >= 2).sum()))
        for name, val in (
            ("head_groups", len(heads)),
            ("facet_groups", len(facets)),
            ("place_groups", len(places)),
            ("requires_year", bool(needed)),
            ("disjunctive", plan["any_head"]),
        ):
            add(name, float(val))
        for kind in KINDS:
            add("kind_" + kind, float(plan["kind"] == kind))
        ids = np.flatnonzero(pool).astype(np.int32)
        return ids, np.column_stack([v[ids] for v in data]).astype(np.float32), names
