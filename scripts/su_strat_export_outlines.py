"""匯出 24 個分層坡面的有效坡面與原始 SU 輪廓向量檔（GeoJSON，TWD97 TM2 與 WGS84 各一份），供在 QGIS 等軟體疊影像檢視或編輯。
欄位：sv（判讀表編號）、su_uid（版本化 ID）、kind（effective＝≥15° 有效坡面／original＝原始 Slope Unit）、area_ha。層別不輸出（維持盲測）。
輸出 poc/su_strat/outlines_3826.geojson、outlines_4326.geojson。用法：python scripts/su_strat_export_outlines.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features
from shapely.geometry import shape

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from landslide_incremental import grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_strat"


def main():
    tf, _ = grid()
    key = json.load(open(OUT / "key.json", encoding="utf-8"))
    eff = np.load(POC / "su_lab10_eff.npy")
    org = np.load(POC / "su_lab10.npy")
    te = {t["su_id"]: t["i"] for t in json.load(open(POC / "su_table_eff.json", encoding="utf-8"))}
    to = {t["su_id"]: t["i"] for t in json.load(open(POC / "su_table.json", encoding="utf-8"))}
    rows = []
    for sv, k in sorted(key.items()):
        for kind, lab, idx in (("effective", eff, te[k["su_id"]]), ("original", org, to[k["su_id"]])):
            m = (lab == idx).astype(np.uint8)
            for g, v in features.shapes(m, mask=m.astype(bool), transform=tf):
                rows.append({"sv": sv, "su_uid": k["su_uid"], "kind": kind, "geometry": shape(g)})
    gdf = gpd.GeoDataFrame(rows, crs=3826).dissolve(by=["sv", "su_uid", "kind"], as_index=False)
    gdf["area_ha"] = (gdf.geometry.area / 1e4).round(2)
    gdf.to_file(OUT / "outlines_3826.geojson", driver="GeoJSON")
    gdf.to_crs(4326).to_file(OUT / "outlines_4326.geojson", driver="GeoJSON")
    print(len(gdf), "個多邊形；", gdf.groupby("kind").area_ha.sum().round(0).to_dict())


if __name__ == "__main__":
    main()
