"""Sentinel-2（WMTS 8 位元 TRUE_COLOR，10 m）裸露偵測與評估：補 Wayback 在 2024–2025 年的缺口。
影像：2024-04-04（凱米颱風前）、2025-03-25（與 113 年度圖層影像期相近）；全 POC 有效像素約 94%（浮水印格視為無效）。
規則同 Wayback 基準：色度植生指標 ExG=(2g−r−b)/(r+g+b) < T 且亮度 V ≥ 70，3×3 開運算、移除 <0.1 ha（10 格）。
  A 固定 T=0.08（沿用 Wayback，不看參照標籤）；B 在空間訓練區（5 km 棋盤格偶數格）對 2024 年圖層 F1 最大化選 T，測試區評估。
  ⚠ WMTS 影像為 8 位元顯示用、與 Wayback 的色彩處理不同，T 不保證可沿用，故兩者並列。
評估範圍：兩期皆有效且 DTM 坡度 ≥15°。
(1) 狀態：2025-03-25 偵測 vs 官方 113 年度圖層（incremental_masks 的 '2024'）；2024-04-04 偵測 vs 2023 年圖層（次要，日期差約 7–9 個月）。
(2) 變化：新增裸露＝晚期偵測有、早期偵測（膨脹 1 格）無；對照窗口（2024-04-04～2025-03-25）內事件型目錄多邊形（含 30 m 緩衝）。
    指標：事件內／外新增比例與倍數、窗口事件面積被偵測為新增的比例、與官方圖層新增（2024 有、2023 無）的像素重疊。
限制：單一場景對、單一規則；影像日期與圖層日期不同；事件判釋與年度圖層同為水保署衛星判釋，非獨立真值；偵測沒有雲／河道遮罩。
輸出：poc/s2/eval.json、det_<date>.npy。用法：python scripts/s2_detect.py
"""
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd  # noqa: F401  (先載入，避免與 rasterio 的 PROJ 衝突順序問題)
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import reverse_detect as R  # noqa: E402
from landslide_incremental import RES, grid, mask  # noqa: E402
from landslide_su_events import load_events  # noqa: E402

S2 = R.POC / "s2"
EARLY, LATE = "20240404", "20250325"
T0, V_MIN = 0.08, 70
WIN = (pd.Timestamp("2024-04-04"), pd.Timestamp("2025-03-25"))


def feats(d):
    rgb = np.load(S2 / f"rgb_{d}.npy").astype(np.float32)
    v = np.load(S2 / f"valid_{d}.npy")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    return exg, rgb.max(axis=2), v


def det(exg, vv, valid, T):
    return R.post((exg < T) & (vv >= V_MIN) & valid)


def main():
    tf, shape = grid()
    H, W = shape
    Z = np.load(R.POC / "incremental_masks.npz")
    fe, fl = feats(EARLY), feats(LATE)
    valid = fe[2] & fl[2]
    sl = R.slope10(tf, H, W)
    region = valid & (sl >= R.SLOPE_MIN)
    bi = ((tf.c + (np.arange(W) + 0.5) * RES) // R.BLOCK_M).astype(int)
    bj = ((tf.f - (np.arange(H) + 0.5) * RES) // R.BLOCK_M).astype(int)
    train = (((bj[:, None] + bi[None, :]) % 2) == 0)
    out = {"params": {"dates": [EARLY, LATE], "T_fixed": T0, "V_MIN": V_MIN},
           "region_km2": round(float(region.sum() * 1e-4), 1), "valid_km2": round(float(valid.sum() * 1e-4), 1),
           "gt2024_ha_in_region": round(float((Z["2024"] & region).sum() * 0.01), 1)}
    # B：訓練區選 T（用晚期影像對 2024 年圖層）
    best = None
    for T in np.arange(-0.10, 0.25, 0.02):
        m = R.metrics(det(*fl, T) if False else det(fl[0], fl[1], fl[2], T), Z["2024"], region & train)
        f1 = 2 * m["tp"] / max(2 * m["tp"] + m["fp"] + m["fn"], 1)
        if best is None or f1 > best[1]:
            best = (round(float(T), 2), f1)
    out["B_T_selected_on_train"], out["B_train_F1"] = best[0], round(best[1], 3)
    res = {}
    for name, T in (("A_fixed", T0), ("B_tuned", best[0])):
        dl, de = det(fl[0], fl[1], fl[2], T), det(fe[0], fe[1], fe[2], T)
        np.save(S2 / f"det_{LATE}_{name}.npy", dl)
        np.save(S2 / f"det_{EARLY}_{name}.npy", de)
        r = {"state_late_vs_2024layer": {"all": R.metrics(dl, Z["2024"], region), "train": R.metrics(dl, Z["2024"], region & train), "test": R.metrics(dl, Z["2024"], region & ~train),
                                         "gt_components_all": R.comp_recall(dl, Z["2024"], region)},
             "state_early_vs_2023layer": {"all": R.metrics(de, Z["2023"], region), "test": R.metrics(de, Z["2023"], region & ~train)}}
        # 變化
        new = R.post(dl & ~ndi.binary_dilation(de, iterations=1)) & region
        ev = {n: v for n, v in load_events().items() if WIN[0] < v[0] <= WIN[1]}
        # load_events 只含到 2025-03-01 前的事件；窗口內只取其子集
        U = np.zeros(shape, bool)
        names = []
        for n, (dt, g) in ev.items():
            m = ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=g.values, crs=3826), tf, shape), iterations=3)
            U |= m
            names.append(f"{n}（{dt.date()}）")
        avail = region & ~de            # 早期未偵測為裸露的地方才可能「新增」
        p_in = float(new[U & avail].sum() / max((U & avail).sum(), 1))
        p_out = float(new[~U & avail].sum() / max((~U & avail).sum(), 1))
        off_new = Z["2024"] & ~Z["2023"] & region
        r["change"] = {"events_in_window": names, "new_ha": round(float(new.sum() * 0.01), 1), "new_ha_in_events": round(float((new & U).sum() * 0.01), 1),
                       "share_new_in_events": round(float((new & U).sum() / max(new.sum(), 1)), 3),
                       "P(new|inside events)": round(p_in, 4), "P(new|outside events)": round(p_out, 4), "ratio": round(p_in / p_out, 2) if p_out else None,
                       "events_area_ha_in_region": round(float((U & region).sum() * 0.01), 1), "events_detected_as_new_frac": round(float((new & U).sum() / max((U & region).sum(), 1)), 3),
                       "official_new_ha": round(float(off_new.sum() * 0.01), 1), "official_new_hit_by_s2_new_frac": round(float((off_new & ndi.binary_dilation(new, iterations=2)).sum() / max(off_new.sum(), 1)), 3),
                       "s2_new_overlapping_official_new_frac": round(float((new & ndi.binary_dilation(off_new, iterations=2)).sum() / max(new.sum(), 1)), 3)}
        res[name] = r
        print(name, "T", T, "late state all", {k: r["state_late_vs_2024layer"]["all"][k] for k in ("recall", "precision_lower_bound", "iou")},
              "test recall", r["state_late_vs_2024layer"]["test"]["recall"], "| change", r["change"], flush=True)
    out.update(res)
    (S2 / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
