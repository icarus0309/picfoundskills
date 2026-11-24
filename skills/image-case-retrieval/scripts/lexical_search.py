"""Retrieve photos for new queries with the packaged ranker and a prepared gallery."""

import argparse
from pathlib import Path
from types import SimpleNamespace
import pipeline as p
import lexical_venue
import lexical_album
import lexical_features
import lexical_ranker
from lexical_contrast import vocabulary, enrich
from lexical_plans import compile_queries
from providers import CompatibleAPI


def search(a):
    cases = p.read_json(a.queries)
    if (
        not isinstance(cases, dict)
        or not cases
        or not all(isinstance(t, str) and t.strip() for t in cases.values())
    ):
        raise ValueError("Queries must map distinct IDs to nonempty strings")
    a.out.mkdir(parents=True, exist_ok=True)
    spec = p.read_json(a.model / "model.json")

    def client():
        if not a.key_file:
            raise ValueError(
                "Provide --key-file or cached --plans and --contrast when required"
            )
        return CompatibleAPI(
            a.out / "text_cache",
            a.key_file.read_text(encoding="utf-8-sig").strip(),
            "https://api.deepseek.com",
            "deepseek-flash",
            provider="deepseek",
        )

    plans = a.plans or a.out / "plans.json"
    if not a.plans and not plans.exists():
        p.write_json(plans, compile_queries(client(), cases, a.workers))
    prepared = p.read_json(plans)
    if {q: r["query"] for q, r in prepared.items()} != cases:
        raise ValueError("Saved plans belong to different queries")
    contrast = None
    if spec.get("contrast"):
        contrast = a.contrast or a.out / "contrast.json"
        if not a.contrast and not contrast.exists():
            hints = vocabulary(p.read_json(a.gallery / "weighted_docs.json"), prepared)
            p.write_json(
                contrast,
                {
                    "queries": cases,
                    "profiles": enrich(client(), prepared, hints, a.workers),
                },
            )
    facts = (a.gallery / "facts.json") if spec["visual_enrichment"] else None
    features = a.out / "features"
    lexical_features.build(a.gallery, plans, features, facts, contrast)
    if any(c.startswith("album_") for c in spec["columns"]):
        augmented = a.out / "album"
        lexical_album.build(features, a.gallery, augmented)
        features = augmented
    lexical_ranker.predict(
        SimpleNamespace(
            features=features,
            model=a.model,
            plans=plans,
            split=None,
            gallery=a.gallery,
            out=a.out / "result",
            contrast=contrast,
            facts=facts,
        )
    )
    result = a.out / "result"
    if a.require_venue_evidence:
        lexical_venue.apply(a.gallery, plans, result, a.out / "venue-result")
        result = a.out / "venue-result"
    print(result / "matches.csv")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("gallery", "queries", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    for key in ("plans", "contrast", "key-file"):
        cli.add_argument("--" + key, type=Path)
    cli.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets" / "lexical-ranker",
    )
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--require-venue-evidence", action="store_true")
    search(cli.parse_args())
