"""Search Kaggle for candidate external-validation datasets.

External validation needs an *independent* source, not our own data resplit. This
lists candidates with their licences so one can be chosen on evidence rather than
convenience.
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

TERMS = [
    "nail disease",
    "nail",
    "onychomycosis",
    "nail fungus",
    "subungual melanoma",
    "clubbing finger",
    "dermatology clinical images",
]

EXCLUDE = {
    "nikhilgurav21/nail-disease-detection-dataset",
    "josephrasanjana/nail-disease-image-classification-dataset",
}


def main() -> int:
    creds = json.loads((Path.home() / ".kaggle" / "kaggle.json").read_text())
    auth = (creds["username"], creds["key"])

    seen: dict[str, dict] = {}
    for term in TERMS:
        r = requests.get(
            "https://www.kaggle.com/api/v1/datasets/list",
            params={"search": term},
            auth=auth,
            timeout=30,
        )
        if r.status_code != 200:
            print(f"[warn] '{term}' -> HTTP {r.status_code}")
            continue
        for d in r.json():
            ref = d.get("ref")
            if not ref or ref in seen or ref in EXCLUDE:
                continue
            seen[ref] = d

    rows = []
    for ref, d in seen.items():
        mb = (d.get("totalBytes") or 0) / 1e6
        if mb < 3:
            continue
        rows.append(
            {
                "ref": ref,
                "mb": mb,
                "licence": str(d.get("licenseName")),
                "title": str(d.get("title"))[:48],
                "downloads": d.get("downloadCount") or 0,
                "usability": d.get("usabilityRating") or 0,
            }
        )

    rows.sort(key=lambda r: -r["mb"])
    print(f"{'SIZE':>9}  {'LICENCE':22s}  {'DL':>6}  REF")
    print("-" * 100)
    for r in rows[:40]:
        print(
            f"{r['mb']:8.0f}MB  {r['licence'][:22]:22s}  {r['downloads']:6d}  "
            f"{r['ref']}"
        )
        print(f"{'':9}  {'':22s}  {'':6s}  {r['title']}")
    print(f"\n{len(rows)} candidates >=3MB, excluding the two already used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
