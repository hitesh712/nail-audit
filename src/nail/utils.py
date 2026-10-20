"""Small shared helpers."""

from __future__ import annotations

import pandas as pd


def df_to_markdown(df: pd.DataFrame, index_name: str | None = None) -> str:
    """Render a DataFrame as a markdown table without requiring ``tabulate``.

    pandas' own ``to_markdown`` raises if tabulate is absent, which is a silly
    reason for a reporting script to die three hours into a run.
    """
    if df.empty:
        return "_(empty)_"

    idx_header = index_name or (df.index.name or "")
    headers = [str(idx_header)] + [str(c) for c in df.columns]

    rows: list[list[str]] = []
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if v % 1 else f"{v:.0f}")
            else:
                cells.append(str(v))
        rows.append([str(idx)] + cells)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    def line(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"

    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)
