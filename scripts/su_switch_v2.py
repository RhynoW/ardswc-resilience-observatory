"""把下游流程切換到 Slope Unit v2（集水區重疊法；slope_units_v2.py 產製）。
1. 備份 v1 的單元與衍生檔到 poc/su_v1/（只在第一次執行時備份，不覆蓋既有備份）。
2. 以 v1 相同格式寫入 su_labels.npz／su_meta.json（原始單元）與 su_labels_eff.npz／su_meta_eff.json（有效坡面），
   讓 landslide_su_events.py、su_photos.py、su_pages_build.py 等不必改即可重跑。
之後依序：
  python scripts/landslide_su_events.py
  SU_OUT_SUFFIX=_eff python scripts/landslide_su_events.py
  python scripts/su_photos.py ; python scripts/su_photo_meta.py ; python scripts/wayback_poc_inventory.py
  python scripts/su_pages_build.py ; python scripts/export_poc_web.py
已完成的 24 個分層人工驗證（su_strat/）與稽核包維持 v1 版本（判讀當時的單元），以 su_crosswalk_v1_v2.json 對到 v2。
用法：python scripts/su_switch_v2.py
"""
import json
import shutil
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
BK = POC / "su_v1"
BACKUP = ["su_labels.npz", "su_meta.json", "su_labels_eff.npz", "su_meta_eff.json", "su_lab10.npy", "su_lab10_eff.npy", "su_table.json", "su_table_eff.json",
          "su_events.json", "su_events_eff.json", "su_evidence.json", "su_photos.json", "su_photo_meta.json", "wayback_su_depth.json"]


def main():
    BK.mkdir(exist_ok=True)
    for f in BACKUP:
        if (POC / f).exists() and not (BK / f).exists():
            shutil.copy2(POC / f, BK / f)
    V2 = np.load(POC / "su_labels_v2.npz")
    lab, key_eff, x0, y0 = V2["key20"], V2["key_eff"], float(V2["x0"]), float(V2["y0"])
    m2 = json.load(open(POC / "su_meta_v2.json", encoding="utf-8"))
    p2 = m2["params"]
    ver = p2["su_version"]
    with rasterio.open(POC / "su_work" / "dem.tif") as r:
        dem = r.read(1).astype(np.float32)
    dem = np.where(dem > -9990, dem, np.nan)
    n = m2["n_units"]
    zmin = np.full(n, np.inf)
    zmax = np.full(n, -np.inf)
    m = (lab >= 0) & np.isfinite(dem)
    np.minimum.at(zmin, lab[m], dem[m])
    np.maximum.at(zmax, lab[m], dem[m])
    params = {"stream_ha": p2["stream_ha"], "min_su_ha": p2["min_su_ha"], "dtm": p2["dtm"], "buffer_m": 6000.0, "origin_xy": [x0, y0], "shape": list(lab.shape),
              "su_version": ver, "method": p2["method"]}
    units = [{"su_id": u["su_id"], "key": u["key"], "area_ha": u["area_ha"], "mean_slope_deg": u["mean_slope_deg_full"],
              "z_min": round(float(zmin[u["key"]])), "z_max": round(float(zmax[u["key"]]))} for u in m2["units"]]
    np.savez_compressed(POC / "su_labels.npz", key20=lab.astype(np.int32), x0=x0, y0=y0, res=20.0)
    (POC / "su_meta.json").write_text(json.dumps({"params": params, "n_units": len(units), "units": units}, ensure_ascii=False), encoding="utf-8")
    ue = []
    n_none = 0
    for u in m2["units"]:
        flag = "有效坡面" if u["eff_area_ha"] >= 1.0 else ("有效坡面<1 ha" if u["eff_area_ha"] > 0 else "無有效坡面")
        n_none += u["eff_area_ha"] < 1.0
        ue.append({"su_id": u["su_id"], "su_uid": u["su_uid"], "key": u["key"], "area_ha": u["area_ha"], "eff_area_ha": u["eff_area_ha"], "eff_frac": u["eff_frac"],
                   "mean_slope_deg": u["mean_slope_deg"], "mean_slope_deg_full": u["mean_slope_deg_full"], "eff_flag": flag})
    pe = {**params, "slope_min_deg": p2["slope_min_deg"], "slope_smoothing": p2["slope_smoothing"], "eff_min_fragment_ha": p2["eff_min_fragment_ha"],
          "built": p2["built"], "algorithm": "集水區重疊法（slope_units_v2.py）；有效坡面＝同一 DTM 20 m 坡度 3×3 平均後 ≥15°，單元內去除 <0.2 ha 碎塊、補 <0.2 ha 孔洞",
          "crosswalk_note": "v1→v2 對照：su_crosswalk_v1_v2.json；v1 原檔備份在 su_v1/"}
    np.savez_compressed(POC / "su_labels_eff.npz", key20=key_eff.astype(np.int32), x0=x0, y0=y0, res=20.0)
    (POC / "su_meta_eff.json").write_text(json.dumps({"params": pe, "n_units": len(ue), "n_without_effective": int(n_none), "units": ue}, ensure_ascii=False), encoding="utf-8")
    print("已切換到", ver, "單元", len(units), "有效面積 <1 ha", int(n_none), "；v1 備份在", BK)


if __name__ == "__main__":
    main()
