# -*- coding: utf-8 -*-
"""
Top 100 熱點的地形起伏因子（巡查優先級第四因子），改用本機全臺 20 m DTM 計算（免 token、離線）。

定義沿用舊版（scripts/ardswc_terrain_relief.py，Cesium World Terrain）：以熱點為中心、邊長 span_km
（預設 0.3 km）窗口內的 relief_m = 高程 95th − 5th 百分位（不用 max−min，避免單一雜訊像素灌爆）。
窗口內有效高程（>0，海面/無資料為 0）不足一半時回 null＋error，不當作 0（0 會被誤讀成平地）。

輸出：data/ardswc_hotspots/terrain_relief.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dtm20  # noqa: E402

DATA = HERE.parent / "data" / "ardswc_hotspots"


def main(span_km=0.3, n=31):
    src = dtm20.Dtm20Source()
    hotspots = json.loads((DATA / "top100_consolidated.json").read_text(encoding="utf-8"))
    half = span_km * 1000 / 2
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    out = []
    for h in hotspots:
        dlat = half / 110_574.0
        dlon = half / (111_320.0 * np.cos(np.radians(h["lat"])))
        LO, LA = np.meshgrid(np.linspace(h["lon"] - dlon, h["lon"] + dlon, n),
                             np.linspace(h["lat"] - dlat, h["lat"] + dlat, n))
        z = src.sample_points(LO, LA)
        valid = z[z > 0]
        row = {"rank": h["rank"], "span_km": span_km, "lat": h["lat"], "lon": h["lon"], "computed_at": now,
               "source": "MOI 20 m DTM (2025)"}
        if valid.size < z.size / 2:
            row.update(relief_m=None, error="窗口內有效高程不足一半（海域或 DTM 範圍外）")
        else:
            row.update(relief_m=round(float(np.percentile(valid, 95) - np.percentile(valid, 5)), 1),
                       elev_m=round(float(np.median(valid)), 1), error=None)
        out.append(row)
    (DATA / "terrain_relief.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    ok = [r["relief_m"] for r in out if r["relief_m"] is not None]
    print(f"relief computed {len(ok)}/{len(out)}; median {np.median(ok):.1f} m, max {max(ok):.1f} m")


if __name__ == "__main__":
    main()
