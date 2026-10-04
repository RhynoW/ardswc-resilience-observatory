"""POC 範圍歷年「全臺崩塌地圖層」套疊：首次出現／擴大／縮減／持續裸露／消失（Incremental 判釋）。

輸入：data/biggis_interp/poc/landslide_<西元年>.gpkg（scripts/fetch_landslide_layers.py 產生）
輸出：data/biggis_interp/poc/incremental.json（統計）、incremental_masks.npz（10 m 柵格，快取）

方法（要點）：
1. 2021 年檔案沒有座標系統定義：先以 2022 年圖層為基準，比較「假設 TWD97（EPSG:3826）」與「假設 TWD67（EPSG:3828）」
   兩種解讀下的柵格重疊（IoU），取較高者，並如實回報。
2. 各年圖層轉成 10 m 柵格（EPSG:3826）。年度圖層記錄「當年度仍裸露的完整面積」，所以逐年比較才有意義。
3. 以兩期聯集的連通區塊為單位分類：只在後期出現＝首次出現；只在前期＝消失（復育或判釋遺漏）；
   兩期都有：後期相對前期面積變化 > +THR 為擴大、< −THR 為縮減，其餘為持續裸露。THR 預設 20%，並附 10%、30% 敏感度。
4. 全期（2021→2024）另算「各連通區塊的首次出現年」。
注意：判釋是衛星影像判釋，影像季節與陰影不同會造成假變化，面積變化不等於實際地表變化。
"""
import json
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterio import features
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import box

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
YEARS = [2021, 2022, 2023, 2024]
RES = 10.0
BBOX_LL = (120.55, 22.95, 120.95, 23.35)
THR = 0.20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def grid():
    g = gpd.GeoSeries([box(*BBOX_LL)], crs=4326).to_crs(3826).total_bounds
    x0, y0, x1, y1 = [float(v) for v in g]
    x0, y1 = np.floor(x0 / RES) * RES, np.ceil(y1 / RES) * RES
    w, h = int(np.ceil((x1 - x0) / RES)), int(np.ceil((y1 - y0) / RES))
    return from_origin(x0, y1, RES, RES), (h, w)


def mask(gdf, tf, shape):
    return features.rasterize(((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty),
                              out_shape=shape, transform=tf, fill=0, dtype="uint8").astype(bool)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def classify(prev, cur, thr):
    """以聯集連通區塊分類，回傳 dict（區塊數與面積 ha）。"""
    lab, n = ndi.label(prev | cur, structure=np.ones((3, 3)))
    if n == 0:
        return {}
    idx = np.arange(1, n + 1)
    a_prev = ndi.sum(prev, lab, idx)
    a_cur = ndi.sum(cur, lab, idx)
    px_ha = RES * RES / 1e4
    cat = np.full(n, "持續裸露", dtype=object)
    cat[(a_prev == 0)] = "首次出現"
    cat[(a_cur == 0)] = "消失"
    both = (a_prev > 0) & (a_cur > 0)
    change = (a_cur - a_prev) / np.where(a_prev > 0, a_prev, 1)
    cat[both & (change > thr)] = "擴大"
    cat[both & (change < -thr)] = "縮減"
    out = {}
    for c in ["首次出現", "擴大", "持續裸露", "縮減", "消失"]:
        m = cat == c
        out[c] = {"n": int(m.sum()), "ha_prev": round(float(a_prev[m].sum() * px_ha), 1), "ha_cur": round(float(a_cur[m].sum() * px_ha), 1)}
    return out


def main():
    tf, shape = grid()
    log("柵格", shape, f"{RES:.0f} m")
    raw = {y: gpd.read_file(POC / f"landslide_{y}.gpkg") for y in YEARS}
    for y, g in raw.items():
        log(y, len(g), "CRS", g.crs.to_epsg() if g.crs else None)
    for y in YEARS[1:]:
        raw[y] = raw[y].set_crs(3826, allow_override=True)
    base = mask(raw[2022], tf, shape)

    # 2021 座標系統判定
    g21 = raw[2021]
    trials = {}
    for epsg in (3826, 3828):
        t = g21.set_crs(epsg, allow_override=True).to_crs(3826)
        trials[epsg] = iou(mask(t, tf, shape), base)
        log(f"2021 假設 EPSG:{epsg} 與 2022 柵格 IoU = {trials[epsg]:.3f}")
    # 對照：相鄰年（2022 vs 2023）IoU 當作「同座標系統」的參考水準
    ref = iou(base, mask(raw[2023], tf, shape))
    log(f"參考：2022 與 2023 IoU = {ref:.3f}")
    best = max(trials, key=trials.get)
    raw[2021] = g21.set_crs(best, allow_override=True).to_crs(3826)
    log("採用 2021 座標系統 EPSG:", best)

    M = {y: mask(raw[y], tf, shape) for y in YEARS}
    np.savez_compressed(POC / "incremental_masks.npz", **{str(y): m for y, m in M.items()})
    res = {"grid": {"res_m": RES, "shape": list(shape), "bbox_lonlat": BBOX_LL},
           "crs2021": {"trials_iou_vs_2022": {str(k): round(v, 4) for k, v in trials.items()}, "adopted": best, "ref_iou_2022_2023": round(ref, 4)},
           "area_ha": {str(y): round(float(M[y].sum() * RES * RES / 1e4), 1) for y in YEARS}, "pairs": {}}
    pairs = [(2021, 2022), (2022, 2023), (2023, 2024), (2021, 2024)]
    for a, b in pairs:
        key = f"{a}-{b}"
        res["pairs"][key] = {}
        for thr in (0.1, THR, 0.3):
            res["pairs"][key][f"thr{int(thr * 100)}"] = classify(M[a], M[b], thr)
        px = {"new_px_ha": round(float((M[b] & ~M[a]).sum() * RES * RES / 1e4), 1),
              "kept_px_ha": round(float((M[b] & M[a]).sum() * RES * RES / 1e4), 1),
              "lost_px_ha": round(float((M[a] & ~M[b]).sum() * RES * RES / 1e4), 1), "iou": round(iou(M[a], M[b]), 4)}
        res["pairs"][key]["pixel"] = px
        log(key, px)

    # 全期首次出現年：以四年聯集的連通區塊，找該區塊最早有裸露的年份
    union = np.zeros(shape, bool)
    for y in YEARS:
        union |= M[y]
    lab, n = ndi.label(union, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    first = np.full(n, 9999)
    present = {}
    for y in YEARS:
        present[y] = ndi.sum(M[y], lab, idx) > 0
    for y in reversed(YEARS):
        first[present[y]] = y
    last = np.zeros(n, int)
    for y in YEARS:
        last[present[y]] = y
    n_years = sum(present[y].astype(int) for y in YEARS)
    ha = ndi.sum(union, lab, idx) * RES * RES / 1e4
    res["lifecycle"] = {
        "n_components": int(n),
        "first_year": {str(y): {"n": int((first == y).sum()), "ha": round(float(ha[first == y].sum()), 1)} for y in YEARS},
        "present_all_4": {"n": int((n_years == 4).sum()), "ha": round(float(ha[n_years == 4].sum()), 1)},
        "present_last_only": {"n": int(((first == 2024) & (n_years == 1)).sum())},
        "disappeared_before_2024": {"n": int((last < 2024).sum()), "ha": round(float(ha[last < 2024].sum()), 1)},
        "gap_then_return": {"n": int(sum(1 for i in range(n) if present[2021][i] and not present[2022][i] and (present[2023][i] or present[2024][i])) +
                                     sum(1 for i in range(n) if present[2022][i] and not present[2023][i] and present[2024][i]))},
    }
    (POC / "incremental.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log("完成 → incremental.json")


if __name__ == "__main__":
    main()
