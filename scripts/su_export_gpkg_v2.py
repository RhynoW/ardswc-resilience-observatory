"""輸出 Slope Unit v2（集水區重疊法）圖資與 v1→v2 對照表、兩版比較統計。
輸出：poc/su_export/slope_units_SUv2.gpkg（EPSG:3826；圖層 su_original、su_effective，只含 POC 範圍內的單元＝下游流程 su_table.json 的單元；未切換前以質心落在 POC 外包框內）
      poc/su_export/compare_v1_v2.json（兩版形狀與破碎統計，同一 POC 範圍）
      poc/su_crosswalk_v1_v2.json（每個 v1 單元對應重疊面積最大的 v2 單元及重疊比例）
用法：python scripts/su_export_gpkg_v2.py
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import shape
from skimage.measure import label as sklabel

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_export"
BBOX_TM2 = (203852.17, 2538747.37, 244887.70, 2583112.13)
PX_HA = 0.04


def polys(arr, tf, keep):
    rows = [(int(v), shape(g)) for g, v in features.shapes(arr.astype(np.int32), mask=np.isin(arr, keep), transform=tf, connectivity=4)]
    g = gpd.GeoDataFrame({"key": [r[0] for r in rows]}, geometry=[r[1] for r in rows], crs=3826)
    return g.dissolve(by="key", as_index=False)


def shape_stats(lab, keys):
    n = int(lab.max()) + 1
    cnt = np.bincount(lab[lab >= 0], minlength=n)
    per = np.zeros(n)
    for a, b in ((lab[:, :-1], lab[:, 1:]), (lab[:-1, :], lab[1:, :])):
        m = a != b
        for x in (a[m], b[m]):
            per += np.bincount(x[x >= 0], minlength=n)
    keys = np.asarray(sorted(keys))
    A, P = cnt[keys] * 400.0, per[keys] * 20.0
    w = 2 * A / np.maximum(P, 1)
    pp = 4 * np.pi * A / np.maximum(P, 1) ** 2
    # 單元本身的連通部件數（4 連通）
    cc = sklabel(np.where(lab >= 0, lab + 1, 0), background=0, connectivity=1)
    parts = np.bincount(lab[lab >= 0], weights=None, minlength=n) * 0
    first = {}
    lab_f, cc_f = lab[lab >= 0], cc[lab >= 0]
    pair = np.unique(lab_f.astype(np.int64) * (int(cc.max()) + 1) + cc_f)
    parts = np.bincount(pair // (int(cc.max()) + 1), minlength=n)
    return {"n": int(len(keys)), "area_ha_q5_25_50_75_95": np.percentile(cnt[keys] * PX_HA, [5, 25, 50, 75, 95]).round(1).tolist(),
            "width_m_median": round(float(np.median(w))), "share_width_lt80m": round(float((w < 80).mean()), 3),
            "polsby_popper_median": round(float(np.median(pp)), 3), "share_units_not_contiguous": round(float((parts[keys] > 1).mean()), 3),
            "parts_max": int(parts[keys].max())}


def eff_parts(key_eff, keys):
    cc = sklabel(np.where(key_eff >= 0, key_eff + 1, 0), background=0, connectivity=1)
    m = key_eff >= 0
    M = int(cc.max()) + 1
    pair = np.unique(key_eff[m].astype(np.int64) * M + cc[m])
    parts = np.bincount(pair // M, minlength=int(key_eff.max()) + 1)
    p = parts[np.asarray(sorted(keys))]
    p = p[p > 0]
    return {"eff_parts_median": float(np.median(p)), "eff_parts_p90": float(np.percentile(p, 90)), "eff_parts_max": int(p.max()),
            "share_eff_gt5_parts": round(float((p > 5).mean()), 3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    V2 = np.load(POC / "su_labels_v2.npz")
    lab, key_eff, x0, y0 = V2["key20"], V2["key_eff"], float(V2["x0"]), float(V2["y0"])
    tf = from_origin(x0, y0, 20.0, 20.0)
    meta = json.load(open(POC / "su_meta_v2.json", encoding="utf-8"))
    units = meta["units"]
    ver = meta["params"]["su_version"]
    # POC 範圍：單元質心在 POC 外包框內
    rr, cc = np.nonzero(lab >= 0)
    n = len(units)
    cx = np.bincount(lab[rr, cc], weights=x0 + (cc + 0.5) * 20, minlength=n) / np.maximum(np.bincount(lab[rr, cc], minlength=n), 1)
    cy = np.bincount(lab[rr, cc], weights=y0 - (rr + 0.5) * 20, minlength=n) / np.maximum(np.bincount(lab[rr, cc], minlength=n), 1)
    inpoc = (cx >= BBOX_TM2[0]) & (cx <= BBOX_TM2[2]) & (cy >= BBOX_TM2[1]) & (cy <= BBOX_TM2[3])
    t2 = POC / "su_table.json"                      # 切換到 v2 後，以下游流程實際使用的 POC 單元為準（與網站一致）
    m_now = json.load(open(POC / "su_meta.json", encoding="utf-8"))
    if str(m_now["params"].get("su_version", "")).startswith("SUv2"):
        ids = {t["su_id"] for t in json.load(open(t2, encoding="utf-8"))}
        inpoc = np.array([u["su_id"] in ids for u in units])
    keys = np.flatnonzero(inpoc)
    go = polys(lab, tf, keys)
    for f in ("su_id", "su_uid", "area_ha", "mean_slope_deg_full"):
        go[f] = go["key"].map(lambda k, f=f: units[k][f])
    go = go[["su_id", "su_uid", "area_ha", "mean_slope_deg_full", "geometry"]]
    ge = polys(key_eff, tf, keys)
    for f in ("su_id", "su_uid", "area_ha", "eff_area_ha", "eff_frac", "mean_slope_deg"):
        ge[f] = ge["key"].map(lambda k, f=f: units[k][f])
    ge["n_parts"] = ge.geometry.map(lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1)
    ge = ge.rename(columns={"mean_slope_deg": "mean_slope_eff_deg"})[["su_id", "su_uid", "area_ha", "eff_area_ha", "eff_frac", "mean_slope_eff_deg", "n_parts", "geometry"]]
    path = OUT / "slope_units_SUv2.gpkg"
    if path.exists():
        path.unlink()
    go.to_file(path, layer="su_original", driver="GPKG")
    ge.to_file(path, layer="su_effective", driver="GPKG")

    # v1 同範圍統計（v1 POC 單元＝su_table.json）
    V1D = POC / "su_v1" if (POC / "su_v1" / "su_labels.npz").exists() else POC
    V1 = np.load(V1D / "su_labels.npz")
    k1 = V1["key20"]
    m1 = json.load(open(V1D / "su_meta.json", encoding="utf-8"))
    sid2key = {u["su_id"]: u["key"] for u in m1["units"]}
    t1 = {t["su_id"] for t in json.load(open(V1D / "su_table.json", encoding="utf-8"))}
    # v1 的 key 不連續：壓成連續標籤
    u1, inv1 = np.unique(k1, return_inverse=True)
    lab1 = inv1.reshape(k1.shape) - (1 if u1[0] < 0 else 0)
    keymap = {int(k): i for i, k in enumerate(u1[u1 >= 0].tolist())}
    keys1 = [keymap[sid2key[s]] for s in t1]
    e1 = np.load(V1D / "su_labels_eff.npz")["key20"]
    e1c = np.where(e1 >= 0, lab1, -1)
    cmp_ = {"v1": {"version": "SUv1-D20m-S10-M1-E15", **shape_stats(lab1, keys1), **eff_parts(e1c, keys1)},
            "v2": {"version": ver, **shape_stats(lab, keys), **eff_parts(key_eff, keys)},
            "note": "同一份 20 m DEM、同一 POC 範圍；寬度＝2×面積／周長；Polsby-Popper＝4π×面積／周長²（1＝圓，愈小愈細長）；有效坡面部件數以 4 連通計"}
    (OUT / "compare_v1_v2.json").write_text(json.dumps(cmp_, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(cmp_, ensure_ascii=False, indent=1))

    # v1→v2 對照（最大重疊）
    m = (lab1 >= 0) & (lab >= 0)
    M = int(lab.max()) + 1
    code, c = np.unique(lab1[m].astype(np.int64) * M + lab[m], return_counts=True)
    a, b = code // M, code % M
    order = np.lexsort((-c, a))
    a, b, c = a[order], b[order], c[order]
    first = np.r_[True, a[1:] != a[:-1]]
    tot1 = np.bincount(lab1[lab1 >= 0])
    inv_key = {i: k for k, i in keymap.items()}
    key2sid = {u["key"]: u["su_id"] for u in m1["units"]}
    cw = {key2sid[inv_key[int(i)]]: {"v2": units[int(j)]["su_id"], "overlap_frac_of_v1": round(float(cc_ / tot1[i]), 3)}
          for i, j, cc_ in zip(a[first], b[first], c[first]) if key2sid[inv_key[int(i)]] in t1}
    (POC / "su_crosswalk_v1_v2.json").write_text(json.dumps({"v1": "SUv1-D20m-S10-M1-E15", "v2": ver, "map": cw}, ensure_ascii=False), encoding="utf-8")
    fr = np.array([v["overlap_frac_of_v1"] for v in cw.values()])
    print("crosswalk", len(cw), "v1 單元落在單一 v2 單元的比例中位數", round(float(np.median(fr)), 2), "≥0.8 的占比", round(float((fr >= 0.8).mean()), 2))
    print(path, round(path.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
