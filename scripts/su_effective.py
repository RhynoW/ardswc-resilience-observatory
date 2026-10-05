"""Slope Unit「有效坡面範圍」（依陳振宇委員 2026-10-04 建議）：保留原始 Slope Unit 幾何，另以同一份 20 m DTM 的坡度圖，
把每個 SU 內坡度 <15° 的範圍（緩坡、谷底、大部分河床）裁除，只保留坡度 ≥15° 的有效坡面。不刪除平均坡度低於 15° 的整個 SU，只在分析時改用有效坡面。
輸出（poc/，不覆蓋原檔）：
  su_labels_eff.npz  20 m 標籤（有效坡面；無效格＝-1），欄位與 su_labels.npz 相同
  su_meta_eff.json   版本與參數（DEM、解析度、河網門檻、最小單元、坡度門檻、日期、演算法）＋逐單元的 su_uid、面積、有效面積、有效比例、有效坡度
後續：SU_OUT_SUFFIX=_eff python scripts/landslide_su_events.py 以有效坡面重算履歷（su_table_eff.json、su_events_eff.json）。
版本化 SU_ID：su_uid = <原 su_id>@<su_version>；su_version 記錄 DEM 版本、解析度、STREAM_HA、MIN_SU_HA、坡度門檻；
  未來 DEM 或參數更新時產生新版本，並以空間重疊（crosswalk）對應新舊單元，不宣稱 SU_ID 永久不變。
用法：python scripts/su_effective.py
"""
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
POC = REPO / "data" / "biggis_interp" / "poc"
SLOPE_MIN = 15.0
VERSION = "SUv1-D20m-S10-M1-E15"


def main():
    L = np.load(POC / "su_labels.npz")
    key20, x0, y0 = L["key20"], float(L["x0"]), float(L["y0"])
    meta = json.load(open(POC / "su_meta.json", encoding="utf-8"))
    with rasterio.open(POC / "su_work" / "dem.tif") as r:
        dem = r.read(1).astype(np.float32)
        nod = r.nodata
    dem = np.where(dem == nod, np.nan, dem)
    assert dem.shape == key20.shape, (dem.shape, key20.shape)
    gy, gx = np.gradient(dem, 20.0)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    eff = (slope >= SLOPE_MIN) & np.isfinite(slope) & (key20 >= 0)
    key_eff = np.where(eff, key20, -1).astype(np.int32)
    np.savez_compressed(POC / "su_labels_eff.npz", key20=key_eff, x0=x0, y0=y0, res=20.0)
    px_ha = 400.0 / 1e4
    keys = np.unique(key20[key20 >= 0])
    area = dict(zip(*[a.tolist() for a in np.unique(key20[key20 >= 0], return_counts=True)]))
    area_e = dict(zip(*[a.tolist() for a in np.unique(key_eff[key_eff >= 0], return_counts=True)]))
    sl_sum = np.bincount(np.searchsorted(keys, key_eff[eff]), weights=slope[eff], minlength=len(keys))
    cnt = np.bincount(np.searchsorted(keys, key_eff[eff]), minlength=len(keys))
    sl_mean = sl_sum / np.maximum(cnt, 1)
    kidx = {int(k): i for i, k in enumerate(keys)}
    units = []
    n_none = 0
    for u in meta["units"]:
        k = u["key"]
        a_full = area.get(k, 0) * px_ha
        a_eff = area_e.get(k, 0) * px_ha
        i = kidx.get(k)
        flag = "有效坡面" if a_eff >= 1.0 else ("有效坡面<1 ha" if a_eff > 0 else "無有效坡面")
        if a_eff < 1.0:
            n_none += 1
        units.append({"su_id": u["su_id"], "su_uid": f"{u['su_id']}@{VERSION}", "key": k, "area_ha": round(a_full, 2), "eff_area_ha": round(a_eff, 2),
                      "eff_frac": round(a_eff / a_full, 3) if a_full else 0.0, "mean_slope_deg": round(float(sl_mean[i]), 1) if i is not None and cnt[i] else None,
                      "mean_slope_deg_full": u.get("mean_slope_deg"), "eff_flag": flag})
    out = {"params": {**meta["params"], "slope_min_deg": SLOPE_MIN, "su_version": VERSION, "built": datetime.date.today().isoformat(),
                      "algorithm": "WhiteboxTools D8 流向／河網＋左右半坡面切分（slope_units.py）；有效坡面＝同一 DTM 20 m 坡度 ≥15°（np.gradient）",
                      "crosswalk_note": "新版本以空間重疊對應舊版單元"},
           "n_units": len(units), "n_without_effective": n_none, "units": units}
    (POC / "su_meta_eff.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tot = float((key20 >= 0).sum() * px_ha)
    te = float(eff.sum() * px_ha)
    print(f"版本 {VERSION}；單元 {len(units)}；有效坡面 {te:.0f} ha／全部 {tot:.0f} ha（{te / tot:.1%}）；有效面積 <1 ha 的單元 {n_none}")
    import collections
    print("有效比例分位數(5/25/50/75/95)：", np.percentile([u["eff_frac"] for u in units], [5, 25, 50, 75, 95]).round(2).tolist())
    print(collections.Counter(u["eff_flag"] for u in units))


if __name__ == "__main__":
    main()
