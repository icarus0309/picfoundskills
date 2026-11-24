"""Sample ranked errors on named cases and render unlabeled contact sheets for a local blind audit."""

import argparse
import random
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "skills/image-case-retrieval/scripts")
)
import pipeline as p
from score_weighted import label_sets, file_ids
from weighted_retrieval import sha


def sample(a):
    d = a.run
    g = d / "gallery"
    mode = p.read_json(d / "choice.json")["chosen"]
    result = d / ("pred-" + mode)
    if (
        not (result / "scores.json").exists()
        or sha(result / "predictions.json")
        != p.read_json(result / "seal.json")["predictions"]
    ):
        raise ValueError("A sealed, scored run is required")
    pred = p.read_json(result / "predictions.json")
    qs = {q: pred["queries"][q] for q in a.cases}
    inv = p.read_json(g / "inventory.json")
    mapping = file_ids(a.root, inv)
    refs = label_sets(a.root, qs)
    images = {r["id"]: r for r in inv["images"]}
    rng = random.Random(a.seed)
    rows = []
    for q, text in qs.items():
        expected = {mapping[f] for f in refs[q] if f in mapping}
        found = set(pred["ranker"][q])
        for status, pool in (("extra", found - expected), ("missed", expected - found)):
            rows.extend(
                (text, i, status)
                for i in rng.sample(sorted(pool), min(a.per_group, len(pool)))
            )
    rng.shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(str(a.font), 21)
    for page, start in enumerate(range(0, len(rows), 8), 1):
        sheet = Image.new(
            "RGB", (1200, ((min(8, len(rows) - start) + 3) // 4) * 278), "white"
        )
        pen = ImageDraw.Draw(sheet)
        for n, (query, i, _) in enumerate(rows[start : start + 8], start):
            x = (n - start) % 4 * 300
            y = (n - start) // 4 * 278
            pen.text((x + 3, y + 2), f"R{n + 1:02d}  {query}", font=font, fill="black")
            try:
                with Image.open(p.thumbnail(a.root, g, images[i])) as original:
                    picture = original.convert("RGB")
                    picture.thumbnail((292, 240))
                    sheet.paste(
                        picture,
                        (
                            x + (300 - picture.width) // 2,
                            y + 33 + (240 - picture.height) // 2,
                        ),
                    )
            except (ImportError, RuntimeError, ValueError, OSError):
                pen.text((x, y + 110), "无法解码", font=font, fill="black")
        sheet.save(a.out / f"blind-{page}.jpg", quality=90)
    p.write_json(
        a.out / "visual-review-key.json",
        [
            {"ticket": f"R{j:02d}", "query": t, "image_id": i, "category": s}
            for j, (t, i, s) in enumerate(rows, 1)
        ],
    )
    print(len(rows), "review rows")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for k in ("root", "run", "out", "font"):
        cli.add_argument("--" + k, type=Path, required=True)
    cli.add_argument(
        "--cases", nargs="+", default=["TC0005", "TC0006", "TC0010", "TC0020"]
    )
    cli.add_argument("--seed", type=int, default=7821)
    cli.add_argument("--per-group", type=int, default=2)
    sample(cli.parse_args())
