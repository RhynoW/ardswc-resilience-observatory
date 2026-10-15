"""輸出 Slope Unit 圖資（依 2026-10-08 陳振宇委員來信：請提供 20 m DEM 製作的原始 Slope Unit，以及切除坡度 <15° 之後的 Slope Unit）。
輸出：data/biggis_interp/poc/su_export/slope_units_SUv1.gpkg（EPSG:3826）
  圖層 su_original   ：20 m DTM 切出的原始 Slope Unit（半坡面；只含 POC 範圍內的單元）
  圖層 su_effective  ：同一單元內坡度 ≥15° 的範圍（20 m 格網坡度 np.gradient；多部件 MultiPolygon，未做任何平滑或去碎屑）
  另輸出 su_export/README.txt（來源、參數、演算法、已知限制）與 fragmentation.json（有效坡面破碎程度統計）。
用法：python scripts/su_export_gpkg.py
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import shape

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_export"
V1 = POC / "su_v1" if (POC / "su_v1" / "su_labels.npz").exists() else POC   # 切換到 v2 後，v1 原檔在 su_v1/
SLOPE_MIN = 15.0


def poly_layer(arr, tf, ids_meta, valid_fn=None):
    """整數標籤格網 → 以標籤合併的多部件多邊形（每個 su 一筆）。"""
    rows = []
    for geom, v in features.shapes(arr.astype(np.int32), mask=arr >= 0, transform=tf, connectivity=4):
        rows.append((int(v), shape(geom)))
    g = gpd.GeoDataFrame({"key": [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=3826)
    return g.dissolve(by="key", as_index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = np.load(V1 / "su_labels.npz")
    key20, x0, y0 = L["key20"], float(L["x0"]), float(L["y0"])
    tf = from_origin(x0, y0, 20.0, 20.0)
    with rasterio.open(POC / "su_work" / "dem.tif") as r:
        assert abs(r.transform.c - x0) < 1e-6 and abs(r.transform.f - y0) < 1e-6, (r.transform, x0, y0)
        dem = r.read(1).astype(np.float32)
        dem = np.where(dem == r.nodata, np.nan, dem)
    gy, gx = np.gradient(dem, 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    eff_mask = (slope >= SLOPE_MIN) & np.isfinite(slope)
    key_eff = np.where(eff_mask & (key20 >= 0), key20, -1)

    meta = json.load(open(V1 / "su_meta.json", encoding="utf-8"))
    meta_e = json.load(open(V1 / "su_meta_eff.json", encoding="utf-8"))
    mu = {u["key"]: u for u in meta["units"]}
    me = {u["su_id"]: u for u in meta_e["units"]}
    T = json.load(open(V1 / "su_table.json", encoding="utf-8"))
    in_poc = {t["su_id"] for t in T}
    ver = meta_e["params"]["su_version"]

    go = poly_layer(key20, tf, mu)
    go["su_id"] = go["key"].map(lambda k: mu[k]["su_id"])
    go["su_uid"] = go["su_id"] + "@" + ver
    go["area_ha"] = go["key"].map(lambda k: mu[k]["area_ha"])
    go["mean_slope_deg"] = go["key"].map(lambda k: mu[k]["mean_slope_deg"])
    go["in_poc"] = go["su_id"].map(lambda s: 1 if s in in_poc else 0)
    go = go[["su_id", "su_uid", "area_ha", "mean_slope_deg", "in_poc", "geometry"]]

    ge = poly_layer(key_eff, tf, mu)
    ge["su_id"] = ge["key"].map(lambda k: mu[k]["su_id"])
    ge["su_uid"] = ge["su_id"] + "@" + ver
    ge["area_ha"] = ge["key"].map(lambda k: mu[k]["area_ha"])
    ge["eff_area_ha"] = ge["su_id"].map(lambda s: me[s]["eff_area_ha"])
    ge["eff_frac"] = ge["su_id"].map(lambda s: me[s]["eff_frac"])
    ge["mean_slope_eff_deg"] = ge["su_id"].map(lambda s: me[s]["mean_slope_deg"])
    ge["n_parts"] = ge.geometry.map(lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1)
    ge["in_poc"] = ge["su_id"].map(lambda s: 1 if s in in_poc else 0)
    ge = ge[["su_id", "su_uid", "area_ha", "eff_area_ha", "eff_frac", "mean_slope_eff_deg", "n_parts", "in_poc", "geometry"]]

    path = OUT / "slope_units_SUv1.gpkg"
    if path.exists():
        path.unlink()
    go[go.in_poc == 1].to_file(path, layer="su_original", driver="GPKG")
    ge[ge.in_poc == 1].to_file(path, layer="su_effective", driver="GPKG")
    # 破碎程度（POC 內）
    lab, nlab = ndi.label(eff_mask & (key20 >= 0), structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    pc = ge[(ge.in_poc == 1) & (ge.eff_area_ha > 0)]
    small_px = 5  # 5 格＝0.2 ha
    # 以單元分部件統計：每個 SU 的連通部件數與最大部件占比
    st = {"n_su_poc": int(len(pc)), "eff_parts_median": float(pc.n_parts.median()), "eff_parts_p90": float(pc.n_parts.quantile(0.9)),
          "eff_parts_max": int(pc.n_parts.max()), "share_su_with_gt5_parts": float((pc.n_parts > 5).mean()),
          "eff_area_ha_poc": float(pc.eff_area_ha.sum()), "orig_area_ha_poc": float(pc.area_ha.sum()),
          "eff_frac_median": float(pc.eff_frac.median())}
    # 單格孤島：與 4 鄰域沒有其他有效格相連的格數占有效格比例
    n4 = ndi.convolve(eff_mask.astype(np.uint8), np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode="constant")
    iso = eff_mask & (n4 == 0)
    st["isolated_single_cell_share_of_eff"] = float(iso[key20 >= 0].sum() / max(eff_mask[key20 >= 0].sum(), 1))
    (OUT / "fragmentation.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    print("original", int((go.in_poc == 1).sum()), "effective", int((ge.in_poc == 1).sum()), "→", path, round(path.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
