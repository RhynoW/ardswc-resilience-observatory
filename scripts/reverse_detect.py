"""反向流程第 3 步：以 Wayback 影像（2022-08-26、2023-11-02）做「裸露地」基準偵測，再以官方年度崩塌圖層（2022、2023）當真值評估。

這是「狀態」評估：影像偵測出的裸露範圍 vs 官方各年度「仍裸露」範圍（110–113 年度全臺崩塌地圖層）。
「變化」評估（前後期變化 vs 事件型目錄）是下一步，不在此腳本。

基準（不訓練模型）：
  A  固定物理規則：色度植生指標 ExG=(2g−r−b)/(r+g+b) < T_FIXED 且亮度 ≥ V_MIN 的像素為裸露候選（T_FIXED 事先定死，不看真值）。
  B  同規則，ExG 門檻在「空間訓練區」以真值 F1 最大化選出，在「空間測試區」評估（5 km 棋盤格，偶數格訓練、奇數格測試），避免用同一份真值調參又打分。
後處理：像素遮罩以 ~10 m 盒式平均聚合到 10 m TM2 格網（真值同一格網），≥0.5 判為裸露，開運算，移除 <0.1 ha（10 格）連通塊。
評估範圍（遮罩）：兩期影像皆有效（拍攝範圍內）、DTM 坡度 ≥ SLOPE_MIN（排除河道與平坦地；官方圖層不把河道算崩塌）。
指標：10 m 像素 TP/FP/FN、精確率（⚠ 只是下限：官方有漏報，「偵測有、官方無」不等於偽陽性，需人工複核）、召回率、IoU；
      真值連通塊召回率（與偵測重疊 ≥20% 面積算命中），依面積分層（<0.5、0.5–2、>2 ha）；依訓練／測試分區報告。
限制：影像 z17（約 1.2 m），RGB 只有可見光，無法分辨裸土農地／道路／建物（用坡度遮罩只部分排除）；陰影中的裸露會漏；
      影像拍攝日與圖層日期不同（2022-08-26 vs 2022-06-23；2023-11-02 vs 2023-07-12，相差約 2–4 個月）；單一場景、兩個日期，結果不可外推到其他年份。
輸出：data/biggis_interp/poc/reverse/{bare_<date>.npy, eval.json, valid.npy}
用法：python scripts/reverse_detect.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from landslide_incremental import RES, grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "reverse"
Z = 17
DATES = {"2022": "2022-08-26", "2023": "2023-11-02"}
T_FIXED = 0.08
V_MIN = 70
SLOPE_MIN = 15.0
BLOCK_M = 5000.0
MIN_CELLS = 10


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pixel_bare(img):
    """BGR uint8 → (ExG 色度植生指標, 亮度 V)。"""
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    s = r + g + b + 1e-6
    exg = (2 * g - r - b) / s
    return exg, f.max(axis=2)


def cell_index(H, W, tf):
    """10 m TM2 格網每格中心 → z17 圖磚 (tx,ty) 與圖磚內像素 (px,py)。"""
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    xs = tf.c + (np.arange(W) + 0.5) * RES
    ys = tf.f - (np.arange(H) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    mx, my = to_m.transform(X.ravel(), Y.ravel())
    n = 2 ** Z
    u = (np.asarray(mx) + 20037508.342789244) / (2 * 20037508.342789244) * n
    v = (20037508.342789244 - np.asarray(my)) / (2 * 20037508.342789244) * n
    tx, ty = np.floor(u).astype(np.int32), np.floor(v).astype(np.int32)
    px, py = np.floor((u - tx) * 256).astype(np.int16), np.floor((v - ty) * 256).astype(np.int16)
    return tx, ty, px, py


def footprint_mask_px(fp_rings, tx, ty):
    """某圖磚內被該日拍攝範圍涵蓋的像素（多邊形以經緯度 → 圖磚像素）。"""
    import math
    n = 2 ** Z
    m = np.zeros((256, 256), np.uint8)
    for ring in fp_rings:
        pts = []
        for lon, lat in ring:
            u = (lon + 180) / 360 * n
            v = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
            pts.append([(u - tx) * 256, (v - ty) * 256])
        cv2.fillPoly(m, [np.round(np.array(pts)).astype(np.int32)], 1)
    return m.astype(bool)


def build_features(date, tx, ty, px, py, H, W, fps):
    """回傳每格的 (ExG 平均, 亮度≥V_MIN 的裸露像素比例的依賴量) — 為了可重調門檻，存兩個 10 m 欄位：
    exg_med（盒內 ExG 中位）、bright（盒內亮度≥V_MIN 比例）、valid（落在拍攝範圍內）。"""
    d = POC / "wb_tiles" / date
    fp = [f for f in fps if f["date"] == date]
    exg = np.full(H * W, np.nan, np.float32)
    bright = np.zeros(H * W, np.float32)
    valid = np.zeros(H * W, bool)
    key = tx.astype(np.int64) * 1_000_000 + ty
    order = np.argsort(key, kind="stable")
    ks = key[order]
    bounds = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(ks)]])
    have = 0
    for s, e in zip(starts, ends):
        t = int(ks[s] // 1_000_000), int(ks[s] % 1_000_000)
        p = d / f"{t[0]}_{t[1]}.jpg"
        if not p.exists():
            continue
        img = cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (256, 256):
            continue
        ex, v = pixel_bare(img)
        ex = cv2.blur(ex, (9, 9))                                  # ≈10 m 盒式平均
        br = cv2.blur((v >= V_MIN).astype(np.float32), (9, 9))
        rings = [r["rings"] for r in fp if True]
        fm = np.zeros((256, 256), bool)
        for rr in rings:
            fm |= footprint_mask_px(rr, *t)
        idx = order[s:e]
        pxs, pys = px[idx], py[idx]
        valid[idx] = fm[pys, pxs]
        exg[idx] = ex[pys, pxs]
        bright[idx] = br[pys, pxs]
        have += 1
    log(date, "有圖磚", have, "個")
    return exg.reshape(H, W), bright.reshape(H, W), valid.reshape(H, W)


def slope10(tf, H, W):
    r = rasterio.open(POC / "su_work" / "dem.tif")
    dem = r.read(1).astype(np.float32)
    dem[dem < -100] = np.nan
    gy, gx = np.gradient(dem, 20.0)
    sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    xs = tf.c + (np.arange(W) + 0.5) * RES
    ys = tf.f - (np.arange(H) + 0.5) * RES
    cj = np.clip(((xs - r.bounds.left) / 20.0).astype(int), 0, sl.shape[1] - 1)
    ri = np.clip(((r.bounds.top - ys) / 20.0).astype(int), 0, sl.shape[0] - 1)
    return np.nan_to_num(sl[np.ix_(ri, cj)], nan=0.0)


def post(bare):
    m = cv2.morphologyEx(bare.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= MIN_CELLS
    return keep[lab]


def metrics(det, gt, region):
    d, g = det & region, gt & region
    tp, fp, fn = int((d & g).sum()), int((d & ~g).sum()), int((~d & g).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "precision_lower_bound": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None, "iou": round(tp / (tp + fp + fn), 3) if tp + fp + fn else None,
            "det_ha": round((tp + fp) * 0.01, 1), "gt_ha": round((tp + fn) * 0.01, 1)}


def comp_recall(det, gt, region):
    g = (gt & region).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(g, connectivity=8)
    out = {"<0.5ha": [0, 0], "0.5–2ha": [0, 0], ">2ha": [0, 0]}
    d = det & region
    hit_count = np.bincount(lab[d].ravel(), minlength=n)
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA] * 0.01
        if a < 0.1:
            continue
        k = "<0.5ha" if a < 0.5 else ("0.5–2ha" if a <= 2 else ">2ha")
        out[k][1] += 1
        if hit_count[i] / st[i, cv2.CC_STAT_AREA] >= 0.2:
            out[k][0] += 1
    return {k: {"hit": v[0], "n": v[1], "recall": round(v[0] / v[1], 3) if v[1] else None} for k, v in out.items()}


def main():
    OUT.mkdir(exist_ok=True)
    tf, (H, W) = grid()
    fps = json.load(open(POC / "wayback_footprints.json", encoding="utf-8"))
    log("格網", H, W, "建立格→圖磚索引…")
    tx, ty, px, py = cell_index(H, W, tf)
    feats = {}
    for y, dt in DATES.items():
        f = OUT / f"feat_{dt}.npz"
        if f.exists():
            z = np.load(f); feats[y] = (z["exg"], z["bright"], z["valid"])
        else:
            feats[y] = build_features(dt, tx, ty, px, py, H, W, fps)
            np.savez_compressed(f, exg=feats[y][0], bright=feats[y][1], valid=feats[y][2])
    del tx, ty, px, py
    z = np.load(POC / "incremental_masks.npz")
    gt = {y: z[y] for y in DATES}
    sl = slope10(tf, H, W)
    valid = feats["2022"][2] & feats["2023"][2]
    steep = sl >= SLOPE_MIN
    region = valid & steep
    # 空間分區：5 km 棋盤格
    bi = ((tf.c + (np.arange(W) + 0.5) * RES) // BLOCK_M).astype(int)
    bj = ((tf.f - (np.arange(H) + 0.5) * RES) // BLOCK_M).astype(int)
    train = (((bj[:, None] + bi[None, :]) % 2) == 0)
    log("評估範圍 km²", round(region.sum() * 0.0001, 1), "；有效（兩期影像）km²", round(valid.sum() * 0.0001, 1), "；訓練格比例", round(float(train[region].mean()), 2))
    res = {"params": {"T_FIXED": T_FIXED, "V_MIN": V_MIN, "SLOPE_MIN": SLOPE_MIN, "MIN_CELLS": MIN_CELLS, "dates": DATES},
           "region_km2": round(float(region.sum() * 0.0001), 1), "valid_km2": round(float(valid.sum() * 0.0001), 1),
           "gt_in_valid_km2": {y: round(float((gt[y] & valid).sum() * 0.0001), 2) for y in DATES},
           "gt_in_region_frac_of_gt_in_valid": {y: round(float((gt[y] & region).sum() / max((gt[y] & valid).sum(), 1)), 3) for y in DATES}}

    def det_for(y, T):
        ex, br, va = feats[y]
        b = (np.nan_to_num(ex, nan=1.0) < T) & (br >= 0.5)
        return post(b & va)
    # 基準 A
    res["A_fixed"] = {}
    for y in DATES:
        d = det_for(y, T_FIXED)
        np.save(OUT / f"bare_A_{DATES[y]}.npy", d)
        res["A_fixed"][y] = {"all": metrics(d, gt[y], region), "train": metrics(d, gt[y], region & train), "test": metrics(d, gt[y], region & ~train), "gt_components": comp_recall(d, gt[y], region)}
    # 基準 B：訓練區選 T
    best = None
    for T in np.arange(-0.10, 0.25, 0.02):
        f1s = []
        for y in DATES:
            m = metrics(det_for(y, T), gt[y], region & train)
            f1s.append(2 * m["tp"] / max(2 * m["tp"] + m["fp"] + m["fn"], 1))
        s = float(np.mean(f1s))
        if best is None or s > best[1]:
            best = (round(float(T), 2), s)
    res["B_tuned"] = {"T_selected_on_train": best[0], "train_F1": round(best[1], 3)}
    for y in DATES:
        d = det_for(y, best[0])
        np.save(OUT / f"bare_B_{DATES[y]}.npy", d)
        res["B_tuned"][y] = {"train": metrics(d, gt[y], region & train), "test": metrics(d, gt[y], region & ~train), "gt_components_test": comp_recall(d, gt[y], region & ~train)}
    np.save(OUT / "region.npy", region)
    np.save(OUT / "train.npy", train)
    (OUT / "eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
