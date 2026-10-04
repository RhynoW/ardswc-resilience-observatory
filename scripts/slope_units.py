"""以 20 m DTM 切 Slope Unit（半坡面／half-basin）並賦予可重現的 SU_ID。

流程（WhiteboxTools 做水文，numpy 做左右岸切分）：
  DTM 裁切（POC＋緩衝） → 填窪（breach least cost） → D8 流向與累積 → 河網（累積面積門檻）
  → 河段子集水區（Subbasins） → 以流向路徑找出每格匯入的河道格，依河道前進方向左／右切成半坡面。
  過小的半坡面併入同河段另一側。
SU_ID：由「河段出口格」的 TWD97 20 m 格網索引加左右岸組成，例如 SU-0123-4567-L。
  同一份 DTM 與同樣參數下可重現；換 DTM 版本或參數則不保證不變（要保留對照表）。
輸出：data/biggis_interp/poc/su_labels.npz（標籤格網）與 su_meta.json（單元清單與統計）。
POC 範圍須與 landslide_incremental.py 一致；標籤另存為 10 m 柵格（與崩塌遮罩同格網）。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
WORK = POC / "su_work"
DTM = REPO / "data" / "dtm" / "tw_dtm20.tif"
BBOX_LL = (120.55, 22.95, 120.95, 23.35)
# 上列經緯度範圍轉 TWD97（EPSG:3826）的外包框（預先以 pyproj 算好；rasterio 與 pyproj 各帶一份不相容的 PROJ 資料庫，
# 這支腳本避免同時載入兩者）
BBOX_TM2 = (203852.17, 2538747.37, 244887.70, 2583112.13)
BUF_M = 6000.0           # 緩衝，避免邊界集水區被截斷
STREAM_HA = 10.0         # 河網累積面積門檻（公頃）
MIN_SU_HA = 1.0          # 小於此面積的半坡面併入同河段另一側

# D8 指標（WhiteboxTools 預設，2 的次方）→ (drow, dcol)
D8 = {1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1), 64: (-1, 0), 128: (-1, 1)}


# TWD97 / TM2 zone 121（與 EPSG:3826 相同），以 WKT 提供，避免查 PROJ 資料庫
TM2_WKT = ('PROJCS["TWD97 / TM2 zone 121",GEOGCS["TWD97",DATUM["Taiwan_Datum_1997",SPHEROID["GRS 1980",6378137,298.257222101]],'
           'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],'
           'PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",121],PARAMETER["scale_factor",0.9999],'
           'PARAMETER["false_easting",250000],PARAMETER["false_northing",0],UNIT["metre",1]]')


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def clip_dtm():
    x0, y0, x1, y1 = BBOX_TM2
    x0, y0, x1, y1 = x0 - BUF_M, y0 - BUF_M, x1 + BUF_M, y1 + BUF_M
    with rasterio.open(DTM) as src:
        win = from_bounds(x0, y0, x1, y1, src.transform).round_offsets().round_lengths()
        arr = src.read(1, window=win)
        tf = src.window_transform(win)
        nod = src.nodata
    arr = np.where(arr == nod, -9999.0, arr).astype("float32")
    WORK.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", transform=tf, nodata=-9999.0, crs=TM2_WKT)   # WhiteboxTools 要求 GeoTIFF 有 geokeys
    with rasterio.open(WORK / "dem.tif", "w", **prof) as dst:
        dst.write(arr, 1)
    log("DTM 裁切", arr.shape, "origin", tf.c, tf.f)
    return arr, tf


def hydro():
    import whitebox
    w = whitebox.WhiteboxTools()
    w.set_verbose_mode(False)
    d = str(WORK) + os.sep
    w.breach_depressions_least_cost(d + "dem.tif", d + "filled.tif", dist=200, fill=True)
    w.d8_pointer(d + "filled.tif", d + "d8.tif")
    w.d8_flow_accumulation(d + "d8.tif", d + "acc.tif", out_type="cells", pntr=True)
    thr = STREAM_HA * 1e4 / 400.0
    w.extract_streams(d + "acc.tif", d + "streams.tif", threshold=thr)
    w.subbasins(d + "d8.tif", d + "streams.tif", d + "subbasins.tif")
    w.stream_link_identifier(d + "d8.tif", d + "streams.tif", d + "links.tif")
    log("WhiteboxTools 水文完成")


def rd(name):
    with rasterio.open(WORK / name) as s:
        return s.read(1)


def halves(d8, streams, sub):
    """每個非河道格追蹤到匯入的河道格，依河道前進方向（左／右）切半坡面。回傳 side（0/1）與入口格索引。"""
    h, w = d8.shape
    idx = np.arange(h * w).reshape(h, w)
    nxt = idx.copy()
    dr = np.zeros(h * w, dtype=np.int64)
    dc = np.zeros(h * w, dtype=np.int64)
    for code, (r, c) in D8.items():
        m = d8 == code
        rr, cc = np.nonzero(m)
        nr, nc = np.clip(rr + r, 0, h - 1), np.clip(cc + c, 0, w - 1)
        nxt[rr, cc] = idx[nr, nc]
        dr[idx[rr, cc]] = r
        dc[idx[rr, cc]] = c
    isstream = streams > 0
    flat_stream = isstream.ravel()
    nxt = nxt.ravel()
    nxt[flat_stream] = np.arange(h * w)[flat_stream]   # 河道格：停在原地
    for _ in range(14):                                 # 指標倍增法：2^14 步足以走完所有路徑
        nxt = nxt[nxt]
    entry = nxt                                          # 每格最終落在哪個河道格（或停在無出口處）
    er, ec = entry // w, entry % w
    rr, cc = np.divmod(np.arange(h * w), w)
    # 河道在入口格的前進方向（取其 D8 指向）
    sr, sc = dr[entry], dc[entry]
    cross = sc * (rr - er) - sr * (cc - ec)             # 列＝向南為正；符號只用來分左右
    side = (cross > 0).astype(np.int8).reshape(h, w)
    return side, entry.reshape(h, w)


def main():
    arr, tf = clip_dtm()
    hydro()
    d8, streams, sub, links = rd("d8.tif").astype(np.int64), rd("streams.tif"), rd("subbasins.tif").astype(np.int64), rd("links.tif").astype(np.int64)
    dem_valid = arr > -9990
    sub = np.where(dem_valid, sub, 0)
    side, entry = halves(d8, streams, sub)
    h, w = d8.shape
    # 基本單元＝(子集水區, 左右)；子集水區編號 0 表示無效
    key = sub * 2 + side
    key[sub <= 0] = -1
    # 過小併入同子集水區另一側（向量化：先對照表再一次套用，避免逐單元掃描整張格網）
    px_ha = 400.0 / 1e4
    valid = key >= 0
    u, cnt = np.unique(key[valid], return_counts=True)
    area = dict(zip(u.tolist(), (cnt * px_ha).tolist()))
    remap = u.copy()
    merged = 0
    for i, (k, a_) in enumerate(zip(u.tolist(), (cnt * px_ha).tolist())):
        if a_ < MIN_SU_HA and (k ^ 1) in area:
            remap[i] = k ^ 1
            merged += 1
    key[valid] = remap[np.searchsorted(u, key[valid])]
    log(f"半坡面單元 {len(area)} 個；小於 {MIN_SU_HA} ha 併入另一側 {merged} 個")
    # 出口格：每個子集水區中高程最低的河道格（無河道格者取最低格）；排序一次即可
    x0, y0 = tf.c, tf.f
    rr_all, cc_all = np.nonzero(sub > 0)
    sb = sub[rr_all, cc_all]
    order = np.lexsort((arr[rr_all, cc_all], ~(streams[rr_all, cc_all] > 0), sb))
    sb_sorted = sb[order]
    first = np.r_[True, sb_sorted[1:] != sb_sorted[:-1]]
    outlet = {int(b_): (int(rr_all[o]), int(cc_all[o])) for b_, o in zip(sb_sorted[first], order[first])}
    ids = {}
    kk = np.unique(key[key >= 0])
    for k in kk.tolist():
        r0, c0 = outlet[k // 2]
        gx = int((x0 + (c0 + 0.5) * 20.0) // 20)
        gy = int((y0 - (r0 + 0.5) * 20.0) // 20)
        ids[k] = f"SU-{gx}-{gy}-{'LR'[k % 2]}"
    # 單元清單（先壓成連續標籤再做統計）
    units = []
    comp = np.where(key >= 0, np.searchsorted(kk, key), -1)
    n = len(kk)
    lab = comp + 1                                   # 0 = 無效
    idx = np.arange(1, n + 1)
    areas = ndi.sum(np.ones_like(lab), lab, idx) * px_ha
    gy_, gx_ = np.gradient(np.where(dem_valid, arr, np.nan), 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx_, gy_)))
    sl = ndi.mean(np.nan_to_num(slope), lab, idx)
    zmin = ndi.minimum(np.where(dem_valid, arr, 1e9), lab, idx)
    zmax = ndi.maximum(np.where(dem_valid, arr, -1e9), lab, idx)
    for k, a, s_, z0, z1 in zip(kk, areas, sl, zmin, zmax):
        units.append({"su_id": ids[int(k)], "key": int(k), "area_ha": round(float(a), 2), "mean_slope_deg": round(float(s_), 1), "z_min": round(float(z0)), "z_max": round(float(z1))})
    log("單元面積（ha）分位數 5/25/50/75/95：", np.percentile(areas, [5, 25, 50, 75, 95]).round(1).tolist())
    # 輸出標籤（20 m）與 10 m 版（與崩塌遮罩對位：以最近鄰重採樣）
    np.savez_compressed(POC / "su_labels.npz", key20=key.astype(np.int32), x0=x0, y0=y0, res=20.0)
    meta = {"params": {"stream_ha": STREAM_HA, "min_su_ha": MIN_SU_HA, "dtm": "data/dtm/tw_dtm20.tif（20 m）", "buffer_m": BUF_M, "origin_xy": [x0, y0], "shape": [h, w]},
            "n_units": len(units), "units": units}
    (POC / "su_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log("完成 → su_labels.npz, su_meta.json")


if __name__ == "__main__":
    main()
