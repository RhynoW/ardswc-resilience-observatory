"""委員建議的「分層人工驗證」坡面抽樣（24 個＝6 層 × 4）：使用 ≥15° 有效坡面的 Slope Unit（SUv1-D20m-S10-M1-E15）。
六層（依陳振宇委員 2026-10-04 回覆；治理資料尚未取得，第 5 層以「持續縮減、無事件、無照片」作代理，需在報告中註明）：
  S1 事件有＋近期影像有變化   事件歸因 ≥0.3（擴大／首次出現）且 Sentinel-2 近期新增 ≥1 ha
  S2 事件有＋近期影像無明顯變化  事件歸因 ≥0.3 且 Sentinel-2 近期新增 <0.1 ha
  S3 事件無、照片描述有崩塌／落石（衛星年度圖層無裸露）  無事件歸因、類別＝無崩塌、崩塌／落石類照片 ≥1、近期新增 <0.3 ha
  S4 影像變化高但靠近河床（hard negative）  Sentinel-2 近期新增 ≥2 ha 且其 ≥50% 落在河道遮罩內
  S5 穩定／復育代理  類別＝縮減、無事件歸因、無災害照片、近期新增 <0.1 ha
  S6 負對照  無崩塌、無照片、無事件、近期新增 <0.1 ha、有效坡度 ≥25°、有效面積 ≥5 ha
近期影像變化＝Sentinel-2（2024-04-04 → 2025-03-25）基準偵測：後期有、前期（膨脹 1 格）無（s2_detect.py 的 det_*_A_fixed.npy）。
多樣性：每層盡量涵蓋四個坡向象限（北／東／南／西）、面積大小不同；樣本間距 ≥1.2 km；有效面積 ≥3 ha。
輸出 poc/su_strat/pool_summary.json（各層候選數）、selected.json（含層別與特徵，判讀者不可見）。
用法：python scripts/su_strat_select.py
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rasterio
from scipy import ndimage as ndi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_strat"
SEED = 20261005
SEP_M = 1200.0
LS = ("坡面崩塌／坍方", "落石／滾石")


def aspect_by_su(lab20, keys, dem_path):
    with rasterio.open(dem_path) as r:
        dem = r.read(1).astype(np.float32)
        nod = r.nodata
    dem = np.where(dem == nod, np.nan, dem)
    gy, gx = np.gradient(dem, 20.0)
    asp = np.arctan2(-gx, gy)                                   # 坡面朝向（北起順時針，弧度）
    ok = (lab20 >= 0) & np.isfinite(asp)
    idx = np.searchsorted(keys, lab20[ok])
    s = np.bincount(idx, weights=np.sin(asp[ok]), minlength=len(keys))
    c = np.bincount(idx, weights=np.cos(asp[ok]), minlength=len(keys))
    return (np.degrees(np.arctan2(s, c)) % 360), np.hypot(s, c) / np.maximum(np.bincount(idx, minlength=len(keys)), 1)


def quad(a):
    return "北" if (a >= 315 or a < 45) else ("東" if a < 135 else ("南" if a < 225 else "西"))


def main():
    OUT.mkdir(exist_ok=True)
    T = json.load(open(POC / "su_table_eff.json", encoding="utf-8"))
    lab10 = np.load(POC / "su_lab10_eff.npy")
    meta = json.load(open(POC / "su_meta_eff.json", encoding="utf-8"))
    mu = {u["su_id"]: u for u in meta["units"]}
    ev = {e["su_id"]: e for e in json.load(open(POC / "su_evidence.json", encoding="utf-8"))}
    # S2 近期新增（有效坡面內）
    late = np.load(POC / "s2" / "det_20250325_A_fixed.npy")
    early = np.load(POC / "s2" / "det_20240404_A_fixed.npy")
    new = late & ~ndi.binary_dilation(early, iterations=1)
    river = np.load(POC / "reverse" / "river.npy")
    n = len(T)
    m = lab10 >= 0
    new_ha = np.bincount(lab10[m & new], minlength=n) * 0.01
    new_riv = np.bincount(lab10[m & new & river], minlength=n) * 0.01
    # 坡向
    L = np.load(POC / "su_labels_eff.npz")
    k20 = L["key20"]
    keys = np.unique(k20[k20 >= 0])
    asp, conc = aspect_by_su(k20, keys, POC / "su_work" / "dem.tif")
    kmap = {}
    for u in meta["units"]:
        kmap[u["su_id"]] = u["key"]
    kidx = {int(k): i for i, k in enumerate(keys)}
    rows = []
    for t in T:
        u = mu[t["su_id"]]
        i = t["i"]
        e = ev.get(t["su_id"], {})
        cl = (e.get("field_photos") or {}).get("classes", {})
        ls = sum(cl.get(k, 0) for k in LS)
        dis = sum((e.get("field_photos") or {}).get(k, {}).get("n", 0) for k in ("災害事件", "媒體報導"))
        a = float(asp[kidx[kmap[t["su_id"]]]]) if kmap.get(t["su_id"]) in kidx else None
        rows.append({"i": i, "su_id": t["su_id"], "su_uid": u["su_uid"], "category": t["category"], "eff_area_ha": u["eff_area_ha"], "eff_slope": u["mean_slope_deg"], "cx": t["cx"], "cy": t["cy"],
                     "explained": t["explained_by_events"], "dominant_event": t["dominant_event"], "bare_2024": t["bare_ha"]["2024"], "new_ha_s2": round(float(new_ha[i]), 2),
                     "riv_frac": round(float(new_riv[i] / new_ha[i]), 2) if new_ha[i] > 0 else 0.0, "ls_photos": ls, "dis_photos": dis, "aspect": round(a, 0) if a is not None else None,
                     "quad": quad(a) if a is not None else None})
    ok = [r for r in rows if r["eff_area_ha"] >= 3.0 and r["aspect"] is not None]
    S = {
        "S1 事件有＋近期影像有變化": lambda r: r["explained"] >= 0.3 and r["new_ha_s2"] >= 1.0,
        "S2 事件有＋近期影像無明顯變化": lambda r: r["explained"] >= 0.3 and r["new_ha_s2"] < 0.1,
        "S3 事件無、照片有崩塌／落石": lambda r: r["explained"] < 0.3 and r["category"] == "無崩塌" and r["ls_photos"] >= 1 and r["new_ha_s2"] < 0.3,
        "S4 影像變化高、靠近河床（hard negative）": lambda r: r["new_ha_s2"] >= 2.0 and r["riv_frac"] >= 0.5,
        "S5 穩定／復育代理（縮減、無事件、無照片）": lambda r: r["category"] == "縮減" and r["explained"] < 0.3 and r["dis_photos"] == 0 and r["new_ha_s2"] < 0.1,
        "S6 負對照（無崩塌、無照片、無事件）": lambda r: r["category"] == "無崩塌" and r["dis_photos"] == 0 and r["explained"] < 0.3 and r["new_ha_s2"] < 0.1 and (r["eff_slope"] or 0) >= 25 and r["eff_area_ha"] >= 5.0,
    }
    rng = random.Random(SEED)
    pool = {k: [r for r in ok if f(r)] for k, f in S.items()}
    summ = {k: {"pool": len(v), "quad": dict(Counter(r["quad"] for r in v))} for k, v in pool.items()}
    sel, used = [], []
    for k, v in pool.items():
        rng.shuffle(v)
        pick = []
        for q in ("北", "東", "南", "西"):
            for r in v:
                if r["quad"] == q and r not in pick and all(((r["cx"] - c["cx"]) ** 2 + (r["cy"] - c["cy"]) ** 2) ** 0.5 >= SEP_M for c in used + pick):
                    pick.append(r)
                    break
        for r in v:                                                 # 某象限空缺時以任意候選補滿 4 個
            if len(pick) >= 4:
                break
            if r not in pick and all(((r["cx"] - c["cx"]) ** 2 + (r["cy"] - c["cy"]) ** 2) ** 0.5 >= SEP_M for c in used + pick):
                pick.append(r)
        for r in pick:
            sel.append(dict(r, stratum=k))
        used += pick
        summ[k]["selected"] = len(pick)
        summ[k]["selected_quad"] = dict(Counter(r["quad"] for r in pick))
    (OUT / "pool_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "selected.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in summ.items():
        print(k, "候選", v["pool"], "選", v["selected"], v["selected_quad"], "候選坡向", v["quad"])
    print("合計", len(sel))


if __name__ == "__main__":
    main()
