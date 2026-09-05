# -*- coding: utf-8 -*-
"""
小範圍（預設 3×3km）坡度與流域分析——提示地形上「可能積水/淹水」的候選區域。

沿用 change_detect_viewer 既有的等高線疊圖引擎（scripts/cesium_terrain.py，Cesium World
Terrain），不重新造輪子；差別是這裡不是畫等高線，而是做基礎地形水文分析：

  1. 坡度（slope，度）：標準梯度法（numpy.gradient），與 GDAL/QGIS `slope` 工具同一套數學。
  2. D8 流向：每格找 8 鄰格中下降坡度最陡的方向（GIS 教科書標準演算法，非本專案發明）。
     沒有下降鄰格的格子＝局部窪地（sink），本身就是最直接的積水候選點。
  3. 流量累積（flow accumulation）：依高程由高到低的拓樸順序逐格累加上游流入量——這是
     計算「哪裡的水會匯集」的標準做法（等同 GRASS GIS r.watershed／ArcGIS Flow Accumulation
     的簡化版）。
  4. 「可能積水」候選＝(a) 局部窪地，或 (b) 坡度平緩（<5°，可調）且流量累積偏高
     （前 N 百分位，可調）——地勢平坦又匯集大量上游來水的地方，水不容易排走。

**治理與限制（務必誠實揭露，不是精確淹水模型）**：
- 這是**地形幾何篩選**，不是水文/水利模型——沒有降雨強度、土壤入滲、排水設施（涵管/
  下水道/滯洪池）、土地利用等資料，無法估計實際淹水機率、深度或延時。
- Cesium World Terrain 免費層級解析度有限（實測常見 15–30m，部分地區更粗），抓不到道路
  側溝、建物、小型排水溝這類決定實際都市積水的細節——這個工具看得到的是「大範圍地形窪地與
  匯流路徑」，看不到「這條路會不會積水」。
- 產出一律為**自動篩選候選、非已驗證事實**，需現地勘查或專業水利/水保單位確認。
"""
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cesium_terrain as CT  # noqa: E402

_D8_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_D8_DIST = [math.sqrt(2), 1.0, math.sqrt(2), 1.0, 1.0, math.sqrt(2), 1.0, math.sqrt(2)]


def compute_slope_degrees(elev, cellsize_m):
    """標準梯度法坡度（度），同 GDAL `gdaldem slope` 的數學。"""
    gy, gx = np.gradient(elev, cellsize_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def compute_hillshade(elev, cellsize_m, azimuth_deg=315.0, altitude_deg=45.0):
    """標準山體陰影（同 GDAL `gdaldem hillshade` 公式），純視覺輔助，非分析結果本身。"""
    gy, gx = np.gradient(elev, cellsize_m)
    slope_rad = np.arctan(np.hypot(gx, gy))
    aspect_rad = np.arctan2(gy, -gx)
    zenith_rad = np.radians(90.0 - altitude_deg)
    azimuth_rad = np.radians(azimuth_deg)
    shade = (np.cos(zenith_rad) * np.cos(slope_rad)
             + np.sin(zenith_rad) * np.sin(slope_rad) * np.cos(azimuth_rad - aspect_rad))
    return np.clip(shade, 0.0, 1.0)


def compute_d8_flow(elev, cellsize_m):
    """D8 流向：回傳 best_dir（0-7 對應 _D8_OFFSETS，-1 表示無下降鄰格＝局部窪地）。"""
    ny, nx = elev.shape
    pad = np.pad(elev, 1, mode="edge")
    best_drop = np.zeros((ny, nx))
    best_dir = np.full((ny, nx), -1, dtype=np.int8)
    for k, (di, dj) in enumerate(_D8_OFFSETS):
        neighbor = pad[1 + di:1 + di + ny, 1 + dj:1 + dj + nx]
        drop = (elev - neighbor) / (_D8_DIST[k] * cellsize_m)
        better = drop > best_drop
        best_drop = np.where(better, drop, best_drop)
        best_dir = np.where(better, k, best_dir)
    return best_dir, best_drop


def compute_accumulation(elev, best_dir):
    """流量累積：依高程降冪的拓樸順序逐格累加（每格至少貢獻自己 1 格）。

    處理順序保證正確性：依高程「由高到低」處理，故每格被處理時，所有可能流入它的上游
    （較高）格子都已經處理完畢、累積值已經加總完成——這正是不需要遞迴就能算出流量累積的
    標準做法（D8 流向天生是一棵森林/DAG，拓樸序=高程降冪序）。"""
    ny, nx = elev.shape
    order = np.argsort(-elev.ravel())  # 高程由高到低的攤平索引序
    rows, cols = np.unravel_index(order, (ny, nx))
    flat_dir = best_dir.ravel()[order]
    accum = np.ones(ny * nx, dtype=np.float64)
    for k in range(len(order)):
        d = flat_dir[k]
        if d < 0:
            continue
        i, j = rows[k], cols[k]
        di, dj = _D8_OFFSETS[d]
        ni, nj = i + di, j + dj
        if 0 <= ni < ny and 0 <= nj < nx:
            accum[ni * nx + nj] += accum[i * nx + j]
    return accum.reshape(ny, nx)


def _to_bgr8(gray01):
    return cv2.cvtColor((np.clip(gray01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def analyze(lat, lon, span_km=3.0, res_m=20.0, slope_flat_deg=5.0, accum_percentile=92.0,
            upscale_px=640, source=None):
    """主流程：取地形 -> 坡度/流向/流量累積 -> 判定積水候選 -> 畫合成圖。

    回傳 dict：png_bytes、stats（含門檻、面積比例）、governance_note。
    任何一步失敗（無地形資料等）直接向上拋例外，呼叫端 fail-closed 處理（不猜測結果）。
    """
    z_grid, lat_axis, lon_axis, info = CT.fetch_terrain(lat, lon, span_km=span_km, res_m=res_m, source=source)
    # fetch_terrain 回傳 lat_axis 由南到北遞增（row 0＝最南），若直接畫成圖會南北顛倒
    # （row 0 理論上該對應影像最上緣＝北）。這裡整批上下翻轉成「row 0＝北」的標準地圖方向，
    # 翻轉不影響任何地形關係（下坡方向、匯流路徑皆隨座標系一併翻轉，結果仍正確），
    # 只是重新定義哪個 row index 對應哪個緯度。
    if lat_axis[0] < lat_axis[-1]:
        z_grid = np.flipud(z_grid)
        lat_axis = lat_axis[::-1]
    ny, nx = z_grid.shape
    cellsize_m = span_km * 1000.0 / nx

    finite = np.isfinite(z_grid)
    if finite.sum() < ny * nx * 0.5:
        raise RuntimeError(f"地形網格有效點過少（{finite.sum()}/{ny*nx}），無法可靠分析")
    fill_val = float(np.nanmedian(z_grid[finite]))
    elev = np.where(finite, z_grid, fill_val)

    slope_deg = compute_slope_degrees(elev, cellsize_m)
    best_dir, _ = compute_d8_flow(elev, cellsize_m)
    accum = compute_accumulation(elev, best_dir)

    # 邊界一圈的「窪地」是網格截斷造成的假訊號（外面本來就沒資料，不是真的沒有下降路徑），
    # 排除在窪地判定外；流量累積的邊界效應影響較小（邊界格本來貢獻就小），不特別處理。
    border = np.zeros((ny, nx), dtype=bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    is_sink = (best_dir < 0) & ~border

    accum_thresh = float(np.percentile(accum, accum_percentile))
    is_flat_highflow = (slope_deg < slope_flat_deg) & (accum >= accum_thresh)
    risk_mask = is_sink | is_flat_highflow

    # ── 合成圖：山體陰影底圖 + 藍色排水路徑 + 紅橙色積水候選疊色 ──
    hillshade = compute_hillshade(elev, cellsize_m)
    base = _to_bgr8(hillshade)

    # 排水路徑（流量累積較高但未達積水門檻的格子，畫細藍線示意匯流路徑）
    channel_thresh = float(np.percentile(accum, 80.0))
    channel_mask = (accum >= channel_thresh) & ~risk_mask
    base[channel_mask] = (0.55 * base[channel_mask] + 0.45 * np.array([200, 120, 30])).astype(np.uint8)  # BGR 淺藍

    # 積水候選（橘紅色，較顯眼）
    base[risk_mask] = (0.4 * base[risk_mask] + 0.6 * np.array([40, 90, 235])).astype(np.uint8)  # BGR 橘紅

    # 目標點（通報座標）在網格中的像素位置——已翻轉成 lat_axis[0]＝北緣（row 0）遞減，
    # np.interp 要求 x 遞增，故傳入反轉後的 lat_axis[::-1] 配對反轉後的 row 序。
    row_of_lat = np.interp(lat, lat_axis[::-1], np.arange(ny)[::-1])
    col_of_lon = np.interp(lon, lon_axis, np.arange(nx))
    px, py = int(round(col_of_lon)), int(round(row_of_lat))

    scale = max(1, upscale_px // max(ny, nx))
    disp = cv2.resize(base, (nx * scale, ny * scale), interpolation=cv2.INTER_NEAREST)
    cx, cy = px * scale + scale // 2, py * scale + scale // 2
    if 0 <= cx < disp.shape[1] and 0 <= cy < disp.shape[0]:
        cv2.circle(disp, (cx, cy), max(6, scale), (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(disp, (cx, cy), max(6, scale), (40, 40, 200), 2, lineType=cv2.LINE_AA)

    ok, buf = cv2.imencode(".png", disp)
    if not ok:
        raise RuntimeError("PNG 編碼失敗")

    n_valid = int(finite.sum())
    stats = {
        "span_km": span_km, "res_m_requested": res_m, "cellsize_m_actual": round(cellsize_m, 1),
        "grid_shape": [ny, nx], "elev_min_m": round(float(elev.min()), 1),
        "elev_max_m": round(float(elev.max()), 1), "elev_missing_frac": round(1 - n_valid / (ny * nx), 4),
        "slope_flat_threshold_deg": slope_flat_deg, "accum_percentile_threshold": accum_percentile,
        "n_sink_cells": int(is_sink.sum()), "n_flat_highflow_cells": int(is_flat_highflow.sum()),
        "risk_area_fraction": round(float(risk_mask.sum()) / (ny * nx), 4),
        "computed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }
    governance_note = ("本圖為地形幾何篩選（坡度+D8流向+流量累積），非水文/水利模型——沒有降雨、"
                        "排水設施、土地利用等資料，無法估計實際淹水機率或深度。橘紅色為積水候選"
                        "（局部窪地或平緩高匯流區），淺藍為推估排水/匯流路徑，皆為自動篩選建議、"
                        "非已驗證事實，需現地或專業水利單位確認。地形資料 © Cesium World Terrain"
                        "（免費層級，非商用），解析度有限，抓不到道路側溝等小型排水設施。")
    return {"png_bytes": buf.tobytes(), "stats": stats, "governance_note": governance_note}
