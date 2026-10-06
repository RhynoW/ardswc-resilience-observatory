"""判讀圖 v2（依實務測試的初步回饋改版，2026-10-07）：
  1. Sentinel-2 浮水印遮蔽區不再留灰塊：改用 Sentinel Hub 的上層圖磚補回（sentinel_assist.fetch_image(fill_watermark=True)，補回處解析度較粗 19–38 m/px）。
  2. 時間涵蓋更多：Wayback 最多 4 個不同年份（最新一幀往前各隔 ≥9 個月）＋ Sentinel-2 2021、2023（各取圖層日期 ±120 天內局部雲量低者）＋ 2024-04-04、2025-03-25，
     依拍攝日由舊到新排成 2 列 × 4 欄；每格標示日期、解析度與來源。
  3. Sentinel-2 加對比增強：亮度（LAB 的 L）0.5–99.5 百分位拉伸＋ CLAHE（clip 1.2）；標題註明「對比增強」，只用於目視，不用於任何偵測或評估。
  4. 可疊官方年度圖層的裸露範圍（紅線＝2024 年圖層〔113 年度，影像約 2025-03～04〕、黃線＝2021 年圖層）與白線＝≥15° 有效坡面——
     白線是「統計單元的外框」（20 m DTM 切出的坡面且只保留坡度 ≥15°），不是裸露地的邊界；裸露地要看紅／黃線與影像。專家盲判表不疊官方圖層（避免被官方判釋牽著走）。
用法：from su_panel_v2 import build_cells, compose
"""
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import sentinel_assist as S2  # noqa: E402
import wayback_assist as WB  # noqa: E402
from su_strat_panels import bar  # noqa: E402
from su_wayback_sample import outline_px  # noqa: E402

Z, HALF_M, PW = 18, 380.0, 380
FIXED_S2 = ("20240404", "20250325")
TARGET_S2 = {2021: date(2021, 7, 1), 2023: date(2023, 7, 12)}


def dd(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def enhance(img):
    """Sentinel-2 對比增強：L 通道 1–99 百分位拉伸＋CLAHE。"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    lo, hi = np.percentile(L, (0.5, 99.5))
    L = np.clip((L - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
    lab[..., 0] = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4)).apply(L)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def s2_fetch(lat, lon, ymd):
    img, bounds = S2.fetch_image(lat, lon, HALF_M, ymd, "TRUE_COLOR", fill_watermark=True)
    if S2.cloud_fraction(img) > 0.08 or S2.blank_fraction(img) > 0.02:
        return None
    return img, bounds


def build_cells(t):
    """t：需有 lat、lon。回傳 [{date, kind, label, img(BGR 原圖，S2 已增強), bounds(3857)}]（依日期排序、最多 8 格）。"""
    lat, lon = t["lat"], t["lon"]
    cells = []
    # Wayback：最新一幀往前各隔 ≥270 天，最多 4 幀
    try:
        fr = [f for f in WB.list_frames(lat, lon, Z) if f["date"] >= "20170101" and (f["res_m"] or 9) <= 0.6]
    except Exception:  # noqa: BLE001
        fr = []
    pick = []
    for f in reversed(fr):
        if not pick or all(abs((dd(f["date"]) - dd(p["date"])).days) >= 270 for p in pick):
            pick.append(f)
        if len(pick) >= 4:
            break
    for f in pick:
        try:
            img, b = WB.fetch_image(lat, lon, HALF_M, f["release"], Z)
            cells.append({"date": f["date"], "kind": "W", "label": f"{f['date']} {f.get('res_m') or '?'}m Wayback", "img": img, "bounds": b})
        except Exception:  # noqa: BLE001
            pass
    # Sentinel-2：2021、2023（圖層日期 ±120 天內局部雲量低者）＋ 固定兩期
    for y, td in TARGET_S2.items():
        try:
            sc = S2.search_scenes(lat, lon, td - timedelta(days=120), td + timedelta(days=120), 40.0, 600.0)
        except Exception:  # noqa: BLE001
            sc = []
        sc = sorted(sc, key=lambda x: (x["cloud"], abs((dd(x["date"]) - td).days)))[:4]
        for s in sc:
            try:
                r = s2_fetch(lat, lon, s["date"])
            except Exception:  # noqa: BLE001
                r = None
            if r:
                cells.append({"date": s["date"], "kind": "S", "label": f"{s['date'][:4]}-{s['date'][4:6]}-{s['date'][6:]} 10m Sentinel-2（對比增強）", "img": enhance(r[0]), "bounds": r[1]})
                break
    for d in FIXED_S2:
        try:
            r = s2_fetch(lat, lon, d)
        except Exception:  # noqa: BLE001
            r = None
        if r:
            cells.append({"date": d, "kind": "S", "label": f"{d[:4]}-{d[4:6]}-{d[6:]} 10m Sentinel-2（對比增強）", "img": enhance(r[0]), "bounds": r[1]})
    cells.sort(key=lambda c: c["date"])
    return cells[:8]


def mask_outline_px(mask10, tf, bounds, w, h, tr, ctr_xy, half_m=HALF_M + 60):
    """10 m TM2 遮罩（官方圖層）在範圍內的外環 → 影像像素座標（只取視窗內）。"""
    cx = int((ctr_xy[0] - tf.c) // 10)
    cy = int((tf.f - ctr_xy[1]) // 10)
    k = int(half_m / 10) + 2
    y0, x0 = max(cy - k, 0), max(cx - k, 0)
    sub = mask10[y0:cy + k, x0:cx + k].astype(np.uint8)
    if sub.sum() == 0:
        return []
    cs, _ = cv2.findContours(np.pad(sub, 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    minx, miny, maxx, maxy = bounds
    out = []
    for c in cs:
        if len(c) < 3:
            continue
        p = c[:, 0, :].astype(float) - 1
        X = tf.c + (p[:, 0] + x0 + 0.5) * 10
        Y = tf.f - (p[:, 1] + y0 + 0.5) * 10
        mx, my = tr.transform(X, Y)
        out.append(np.stack([(np.asarray(mx) - minx) / (maxx - minx) * w, (maxy - np.asarray(my)) / (maxy - miny) * h], 1).astype(np.int32))
    return out


def compose(cells, t, lab10, tf, to_m, official=None):
    """official：{'2021': mask10, '2024': mask10} 或 None。白線＝有效坡面；紅線＝2024 圖層；黃線＝2021 圖層。"""
    tiles = []
    for c in cells:
        img, b = c["img"], c["bounds"]
        h, w = img.shape[:2]
        vis = cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC) if c["kind"] == "S" else img.copy()
        H, W = vis.shape[:2]
        sx = W / w
        if official:
            for key, col in (("2021", (0, 220, 255)), ("2024", (60, 60, 255))):
                if key in official:
                    for poly in mask_outline_px(official[key], tf, b, W, H, to_m, (t["cx"], t["cy"])):
                        cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, col, 2, cv2.LINE_AA)
        for poly in outline_px(lab10, t["i"], tf, b, W, H, to_m):
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (255, 255, 255), 2, cv2.LINE_AA)
        vis = cv2.resize(vis, (PW, int(H * PW / W)), interpolation=cv2.INTER_AREA)
        tiles.append(bar(vis, c["label"]))
    if not tiles:
        return None
    hmax = max(x.shape[0] for x in tiles)
    tiles = [np.vstack([x, np.full((hmax - x.shape[0], x.shape[1], 3), 40, np.uint8)]) if x.shape[0] < hmax else x for x in tiles]
    blank = np.full_like(tiles[0], 40)
    rows = []
    for i in range(0, 8, 4):
        r = tiles[i:i + 4]
        r += [blank] * (4 - len(r))
        rows.append(np.hstack(r))
    return np.vstack(rows) if len(tiles) > 4 else rows[0]
