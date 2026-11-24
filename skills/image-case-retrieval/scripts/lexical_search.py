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
from weighted_retrieval import sha


def search(a):
    cases = p.read_json(a.queries)
    if (
        not isinstance(cases, dict)
        or not cases
        or not all(isinstance(t, str) and t.strip() for t in cases.values())
    ):
        raise ValueError("Queries must map distinct IDs to nonempty strings")
    a.out.mkdir(parents=True, exist_ok=True)
    assets = Path(__file__).resolve().parents[1] / "assets"
    model = a.model
    if model is None:
        model = assets / (
            "semantic-ranker" if getattr(a, "text_model", None) else "lexical-ranker"
        )
        if (
            not getattr(a, "clip_model", None)
            and (assets / "text-ranker" / "model.json").is_file()
        ):
            if "pixel_query" in p.read_json(model / "model.json")["columns"]:
                model = assets / "text-ranker"
    spec = p.read_json(model / "model.json")
    recipe_path = model / "review_recipe.json"
    recipe = p.read_json(recipe_path) if recipe_path.exists() else {}
    if getattr(a, "review_config", None) and not recipe:
        raise ValueError("This ranker lacks a calibrated visual review recipe")
    columns = set(spec["columns"])
    uses_text = "semantic_query_title" in columns
    uses_pixel = "pixel_query" in columns
    uses_feedback = "feedback_query_1" in columns
    if uses_text and not getattr(a, "text_model", None):
        raise ValueError("This ranker requires --text-model")
    if uses_pixel and not getattr(a, "clip_model", None):
        raise ValueError("This ranker requires --clip-model and a complete pixel index")
    if "projected_title" in columns and not getattr(a, "projection", None):
        if not (model / "projection" / "projection.json").is_file():
            raise ValueError("This ranker requires --projection")
        a.projection = model / "projection"
    if uses_feedback and not getattr(a, "feedback_bank", None):
        raise ValueError("This ranker requires its matching private --feedback-bank")

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
    if uses_text:
        import semantic_features
        import semantic_index

        index = getattr(a, "text_index", None) or a.gallery / "text-index"
        semantic_index.build(a.gallery, a.text_model, index, 4)
        destination = a.out / "text-features"
        semantic_features.build(
            "text", features, a.gallery, plans, index, a.text_model, destination
        )
        features = destination
    if "projected_title" in columns:
        import semantic_projection

        destination = a.out / "projected-features"
        semantic_projection.build(
            features, a.gallery, plans, index, a.text_model, a.projection, destination
        )
        features = destination
    if uses_pixel:
        index = getattr(a, "clip_index", None) or a.gallery / "clip-index"
        destination = a.out / "pixel-features"
        semantic_features.build(
            "pixel", features, a.gallery, plans, index, a.clip_model, destination
        )
        features = destination
    if uses_feedback:
        import semantic_feedback

        destination = a.out / "feedback-features"
        semantic_feedback.build(
            features, a.gallery, plans, a.feedback_bank, a.text_model, destination
        )
        features = destination
    prediction = dict(
        features=features,
        model=model,
        plans=plans,
        split=None,
        gallery=a.gallery,
        out=a.out / "result",
        contrast=contrast,
        facts=facts,
    )
    lexical_ranker.predict(SimpleNamespace(**prediction))
    result = prediction["out"]
    if getattr(a, "review_config", None):
        if not getattr(a, "photo_root", None):
            raise ValueError(
                "--review-config requires --photo-root with the original gallery"
            )
        from boundary_review import review, apply
        from transfer_review import transfer

        scout_model = (
            getattr(a, "scout_model", None) or model / recipe["scout_directory"]
        )
        if recipe["target"] != sha(model / "model.json") or recipe["scout"] != sha(
            scout_model / "model.json"
        ):
            raise ValueError(
                "The calibrated review recipe belongs to different rankers"
            )
        scout = result
        proposed = a.out / "text-features"
        if recipe["scout"] != recipe["target"]:
            scout = a.out / "scout-result"
            lexical_ranker.predict(
                SimpleNamespace(
                    **dict(prediction, features=proposed, model=scout_model, out=scout)
                )
            )
        reviewed = a.out / "reviews"
        review(
            a.photo_root,
            a.gallery,
            proposed,
            scout_model,
            scout,
            a.review_config,
            reviewed,
            recipe["review_depth"],
            a.workers,
            getattr(a, "block_registry", []),
        )
        depth, drop, add = recipe["depth"], recipe["drop"], recipe["add"]
        if scout == result:
            final = a.out / "reviewed"
            apply(a.gallery, result, reviewed, final, drop, add, depth)
            result = final
        else:
            final = a.out / "reviewed"
            transfer(a.gallery, scout, result, reviewed, final, depth, drop, add)
            result = final / "result"
    require_venue = getattr(a, "require_venue_evidence", None)
    if require_venue is None:
        require_venue = recipe.get("require_venue_evidence", False)
    if require_venue:
        lexical_venue.apply(a.gallery, plans, result, a.out / "venue-result")
        result = a.out / "venue-result"
    print(result / "matches.csv")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("gallery", "queries", "out"):
        cli.add_argument("--" + key, type=Path, required=True)
    for key in ("plans", "contrast", "key-file"):
        cli.add_argument("--" + key, type=Path)
    cli.add_argument(
        "--model",
        type=Path,
        help="Model directory. With --text-model the default is the packaged semantic ranker.",
    )
    for key in (
        "text-model",
        "text-index",
        "clip-model",
        "clip-index",
        "photo-root",
        "review-config",
        "scout-model",
        "feedback-bank",
        "projection",
    ):
        cli.add_argument("--" + key, type=Path)
    cli.add_argument("--block-registry", nargs="*", type=Path, default=[])
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument(
        "--require-venue-evidence", action=argparse.BooleanOptionalAction, default=None
    )
    search(cli.parse_args())
