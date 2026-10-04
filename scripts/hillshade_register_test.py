"""山陰影匹配小規模測試：DTM（20 m）山陰影能否當「絕對參考」量測 Wayback 影像的錯位，並解釋兩期影像的相對錯位？
流程（每個測試點，ref／mov 兩期 z17 鑲嵌，已存在 E:/Temp/reg_t*）：
  1. 把 20 m DTM 重採樣到影像像元網格，用太陽方位／高度（依日期、緯度，假設當地太陽時 10:30 過境）算山陰影；
     影像亮度做高通（差分高斯）以去掉反照率差異。
  2. 逐區塊（128 px、步距 64）：影像區塊中央 96 px 在山陰影的 ±16 px 搜尋窗內做 ZNCC 模板匹配（cv2.matchTemplate），
     取峰值位置為該期影像相對山陰影的位移，峰值 ≥ PEAK_MIN 才採用。
  3. 一致性檢驗：(mov 相對山陰影) − (ref 相對山陰影) 應等於兩期影像直接相位相關得到的相對位移 s_AB；
     報告殘差 RMSE、與 s_AB 自身 RMSE 比較，並看相關係數。若殘差明顯小於 s_AB，表示山陰影位移有物理意義；否則是雜訊。
限制：20 m DTM 對 1.2 m 影像只提供山脊／河谷尺度的特徵；太陽時間為假設；植生與陰影邊界（樹冠）干擾。小規模測試，結果只用來決定要不要繼續投入。
用法：python scripts/hillshade_register_test.py
輸出：data/biggis_interp/areas/hillshade_test.json
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / ".claude" / "skills" / "ortho-preprocess" / "scripts"))
import ortho_register as O  # noqa: E402

CASES = [("1", 23.7741, 121.1569, "20230227", "20170218"), ("2", 23.5558, 121.1929, "20221113", "20180324")]
PEAK_MIN = 0.30
SEARCH = 16
BLK, STRIDE, CORE = 128, 64, 96


def sun(lat, lon, ymd, solar_hour=10.5):
    n = int(__import__("datetime").date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])).strftime("%j"))
    dec = math.radians(23.44 * math.sin(math.radians(360 / 365 * (284 + n))))
    h = math.radians((solar_hour - 12) * 15)
    p = math.radians(lat)
    el = math.asin(math.sin(p) * math.sin(dec) + math.cos(p) * math.cos(dec) * math.cos(h))
    az = math.atan2(-math.sin(h) * math.cos(dec), math.cos(p) * math.sin(dec) - math.sin(p) * math.cos(dec) * math.cos(h))   # 北起順時針
    return math.degrees(az) % 360, math.degrees(el)


def dem_grid(lat, lon, H, W, mpp):
    cx, cy = Transformer.from_crs(4326, 3826, always_xy=True).transform(lon, lat)
    d = rasterio.open(REPO / "data" / "dtm" / "tw_dtm20.tif")
    half = max(H, W) * mpp / 2 + 200
    win = rasterio.windows.from_bounds(cx - half, cy - half, cx + half, cy + half, d.transform).round_offsets().round_lengths()
    arr = np.asarray(d.read(1, window=win), np.float32).copy()
    wtf = d.window_transform(win)
    arr[arr == d.nodata] = np.nan
    arr = np.where(np.isnan(arr), np.nanmean(arr), arr)
    xs = cx + (np.arange(W) - W / 2) * mpp
    ys = cy - (np.arange(H) - H / 2) * mpp
    mapx = ((xs - wtf.c) / 20.0 - 0.5).astype(np.float32)
    mapy = ((wtf.f - ys) / 20.0 - 0.5).astype(np.float32)
    gx, gy = np.meshgrid(mapx, np.zeros(1, np.float32))
    mx = np.tile(mapx, (H, 1))
    my = np.tile(mapy[:, None], (1, W))
    return cv2.remap(arr, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def hillshade(dem, mpp, az, el):
    dem = cv2.GaussianBlur(dem, (0, 0), 6.0)               # 20 m DTM 重採樣後平滑，避免階梯狀
    gy, gx = np.gradient(dem, mpp)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)                           # 坡向（北起順時針，朝下坡方向）
    zen, azr = math.radians(90 - el), math.radians(az)
    hs = math.cos(zen) * np.cos(slope) + math.sin(zen) * np.sin(slope) * np.cos(azr - aspect)
    return hs.astype(np.float32)


def highpass(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 3) - cv2.GaussianBlur(g, (0, 0), 25)


def match_to_hs(img_hp, hs_hp, pts):
    """pts：[(cy, cx)]；回傳 {(cy,cx): (dx, dy, peak)}。"""
    out = {}
    H, W = img_hp.shape
    m = SEARCH
    for cy, cx in pts:
        y0, x0 = cy - CORE // 2, cx - CORE // 2
        if y0 - m < 0 or x0 - m < 0 or y0 + CORE + m > H or x0 + CORE + m > W:
            continue
        tpl = img_hp[y0:y0 + CORE, x0:x0 + CORE]
        if tpl.std() < 1e-3:
            continue
        area = hs_hp[y0 - m:y0 + CORE + m, x0 - m:x0 + CORE + m]
        if area.std() < 1e-4:
            continue
        r = cv2.matchTemplate(area, tpl, cv2.TM_CCOEFF_NORMED)
        _, pk, _, loc = cv2.minMaxLoc(r)
        out[(cy, cx)] = (loc[0] - m, loc[1] - m, float(pk))
    return out


def main():
    res = {"params": {"PEAK_MIN": PEAK_MIN, "SEARCH_px": SEARCH, "solar_hour_assumed": 10.5}, "cases": []}
    for t, lat, lon, dref, dmov in CASES:
        d = Path(f"E:/Temp/reg_t{t}")
        ref, mov = cv2.imread(str(d / "ref.png")), cv2.imread(str(d / "mov_original.png"))
        H, W = ref.shape[:2]
        mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** 17
        dem = dem_grid(lat, lon, H, W, mpp)
        pts_ab = O.block_shifts(ref, mov, BLK, STRIDE)
        s_ab = {(int(p[0]), int(p[1])): (p[2], p[3]) for p in pts_ab}
        hp = {}
        for name, im, ymd in (("ref", ref, dref), ("mov", mov, dmov)):
            az, el = sun(lat, lon, ymd)
            hs = hillshade(dem, mpp, az, el)
            hp[name] = (match_to_hs(highpass(im), highpass(hs) if False else (hs - cv2.GaussianBlur(hs, (0, 0), 25)), list(s_ab.keys()) or []), az, el)
        keys = [k for k in s_ab if k in hp["ref"][0] and k in hp["mov"][0] and hp["ref"][0][k][2] >= PEAK_MIN and hp["mov"][0][k][2] >= PEAK_MIN]
        case = {"case": t, "lat": lat, "lon": lon, "ref": dref, "mov": dmov, "sun_ref": [round(hp["ref"][1]), round(hp["ref"][2])], "sun_mov": [round(hp["mov"][1]), round(hp["mov"][2])],
                "n_blocks_ab": len(s_ab), "n_ref_peak_ok": sum(1 for v in hp["ref"][0].values() if v[2] >= PEAK_MIN), "n_mov_peak_ok": sum(1 for v in hp["mov"][0].values() if v[2] >= PEAK_MIN), "n_all_ok": len(keys),
                "median_peak_ref": round(float(np.median([v[2] for v in hp["ref"][0].values()])), 3) if hp["ref"][0] else None,
                "median_peak_mov": round(float(np.median([v[2] for v in hp["mov"][0].values()])), 3) if hp["mov"][0] else None}
        if len(keys) >= 8:
            r_off = np.array([hp["ref"][0][k][:2] for k in keys]) * mpp
            m_off = np.array([hp["mov"][0][k][:2] for k in keys]) * mpp
            ab = np.array([s_ab[k] for k in keys]) * mpp
            pred = m_off - r_off
            resid = ab - pred
            case.update(ref_vs_hillshade_median_m=round(float(np.median(np.hypot(*r_off.T))), 2), mov_vs_hillshade_median_m=round(float(np.median(np.hypot(*m_off.T))), 2),
                        s_ab_rmse_m=round(float(np.sqrt((np.hypot(*ab.T) ** 2).mean())), 2), resid_rmse_m=round(float(np.sqrt((np.hypot(*resid.T) ** 2).mean())), 2),
                        corr_dx=round(float(np.corrcoef(ab[:, 0], pred[:, 0])[0, 1]), 2), corr_dy=round(float(np.corrcoef(ab[:, 1], pred[:, 1])[0, 1]), 2))
        res["cases"].append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
    out = REPO / "data" / "biggis_interp" / "areas" / "hillshade_test.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
