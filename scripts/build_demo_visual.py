#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def main() -> int:
    root = Path.cwd()
    table_dir = root / "docs" / "assets" / "tables"
    html_dir = root / "docs" / "assets" / "interactive"
    table_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, 61):
        maturity = i / 2.0
        fitted_yield = 3.2 + 1.1 * (1 - math.exp(-maturity / 6.0)) + 0.20 * math.sin(maturity / 2.5)
        observed_yield = fitted_yield + 0.15 * math.sin(i / 3.0)
        residual = observed_yield - fitted_yield
        rows.append(
            {
                "maturity_years": round(maturity, 3),
                "fitted_yield_pct": round(fitted_yield, 5),
                "observed_yield_pct": round(observed_yield, 5),
                "residual_yield_pct": round(residual, 5),
            }
        )

    csv_path = table_dir / "synthetic_curve_demo.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "ok": True,
        "source": "synthetic public demo only",
        "rows": len(rows),
        "note": "No WRDS/vendor data used.",
    }
    (table_dir / "synthetic_curve_demo_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    html_rows = "\n".join(
        f"<tr><td>{r['maturity_years']}</td><td>{r['fitted_yield_pct']}</td><td>{r['observed_yield_pct']}</td><td>{r['residual_yield_pct']}</td></tr>"
        for r in rows
    )
    html = f"""<!doctype html>
<html>
<head><meta charset=\"utf-8\"><title>Synthetic curve demo</title></head>
<body>
<h1>Synthetic issuer-curve demo</h1>
<p>This public smoke-test asset uses synthetic numbers only. It contains no WRDS/vendor data.</p>
<table border=\"1\" cellpadding=\"4\" cellspacing=\"0\">
<thead><tr><th>Maturity</th><th>Fitted yield</th><th>Observed yield</th><th>Residual</th></tr></thead>
<tbody>
{html_rows}
</tbody>
</table>
</body>
</html>
"""
    (html_dir / "synthetic_curve_demo.html").write_text(html, encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
