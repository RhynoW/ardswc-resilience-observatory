# -*- coding: utf-8 -*-
"""
公開向量圖層 → warning_trainer 的 --extra-csv 欄位（離線、可重現；Big GIS 圖層下載後同法加入）。

目前接入：土石流潛勢溪流（農業部水保署，農業資料開放平臺 data.moa.gov.tw 資料集 H83，110 年度 1726 條，SHP/TWD97）
  debris_dist_km        樣本點到最近潛勢溪流（線）的距離（km）
  debris_within_500m    500 m 內有潛勢溪流（0/1）
  debris_high_1km       1 km 內有「高風險」潛勢溪流（0/1）
未取得：山崩與地滑地質敏感區（地調所，其網站自本機連不上；請由 data.gov.tw 資料集 100220／地調所地質法專區下載 SHP 後，
  以 --poly <shp> --poly-name geosens 加入，輸出欄位 geosens_in（點在區內 0/1）、geosens_dist_km）。
注意：110 年版潛勢溪流是「事後劃設的圖資」，若災點（2019–2025）本身曾被納入劃設依據，命中率會偏高（資訊洩漏）；
  解讀時需與災點發生年份對照。

用法：python scripts/layer_features.py --samples data/warning_trainer/samples.csv \
        --lines <debrisstream.shp> [--poly <shp> --poly-name geosens] --out data/warning_trainer/layers.csv
"""
import argparse
import csv

import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--lines")
    ap.add_argument("--risk-col", default="Risk")
    ap.add_argument("--high-values", default="高")
    ap.add_argument("--poly")
    ap.add_argument("--poly-name", default="poly")
    ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.samples, encoding="utf-8")))
    pts = gpd.GeoDataFrame({"id": [r["id"] for r in rows]},
                           geometry=[Point(float(r["lon"]), float(r["lat"])) for r in rows], crs=4326).to_crs(3826)
    out = {r["id"]: {"id": r["id"]} for r in rows}
    if a.lines:
        ln = gpd.read_file(a.lines, encoding=a.encoding).to_crs(3826)
        print("risk values:", ln[a.risk_col].value_counts().to_dict())
        hi = ln[ln[a.risk_col].astype(str).str.contains(a.high_values)]
        for name, g in (("all", ln), ("high", hi)):
            u = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
            d = pts.geometry.distance(u).to_numpy() / 1000
            for r, v in zip(rows, d):
                if name == "all":
                    out[r["id"]]["debris_dist_km"] = round(float(v), 3)
                    out[r["id"]]["debris_within_500m"] = int(v <= 0.5)
                else:
                    out[r["id"]]["debris_high_1km"] = int(v <= 1.0)
    if a.poly:
        pg = gpd.read_file(a.poly).to_crs(3826)
        u = pg.geometry.union_all() if hasattr(pg.geometry, "union_all") else pg.geometry.unary_union
        d = pts.geometry.distance(u).to_numpy() / 1000
        for r, v in zip(rows, d):
            out[r["id"]][f"{a.poly_name}_in"] = int(v == 0)
            out[r["id"]][f"{a.poly_name}_dist_km"] = round(float(v), 3)
    cols = list(next(iter(out.values())))
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out.values())
    print("→", a.out, cols)


if __name__ == "__main__":
    main()
