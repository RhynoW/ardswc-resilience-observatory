"""坡面詳細頁資料（候選排序＋證據鏈）：把每個 ≥15° 有效坡面單元的證據整理成網頁用 JSON（webapp/change_detect_viewer/static/poc/su/）。
每個單元的證據（分四類，不相加、不給總分）：
  E1 衛星事件目錄／年度圖層：類別、四年裸露面積、新增裸露、窗口內事件（名稱、日期、在此單元的面積 ha）
  E2 現地歷史照片：筆數與年份、描述分類（規則式初分）；重要地景照不計入災害
  E3 近期衛星變化：Sentinel-2（2024-04-04→2025-03-25）新增面積、其落在河道遮罩內的比例
  E4 影像品質與偏移提醒：有效面積與比例、Sentinel-2 浮水印遮蔽比例、Wayback 最新一幀日期與張數、Wayback 對 DTM／Sentinel-2 偏移風險
候選層級（POC 規則 v0，未經驗證；只有三種，不給總分；規則與驗證限制寫在頁面）：
  資料不足：有效面積 <1 ha，或有效坡面占比 <0.2，或 Sentinel-2 兩期視窗被浮水印遮蔽 ≥30%（近期變化無從判斷）
  優先人工判讀（任一）：
    P1 近期現地照片（2021 年起）描述含崩塌／落石
    P2 事件目錄連結的新增或擴大（事件歸因 ≥0.3，類別擴大／首次出現）
    P3 Sentinel-2 近期新增 ≥1 ha 且 ≤50% 落在河道遮罩內，且年度圖層類別非「無崩塌」
  一般判讀：其餘
規則依據：24 個分層坡面的人工驗證顯示，近期變化訊號（事件歸因、Sentinel-2 簡單規則）精確率只有約 1/4 且會漏報——所以它們只用來「請人看」，不用來「判定」。
輸出：su_index.json（所有單元的簡表）、su_records.json（有證據的單元完整記錄＋有效坡面輪廓）、rules.json（規則文字與診斷統計）。
用法：python scripts/su_pages_build.py
"""
import json
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from landslide_incremental import grid, mask  # noqa: E402
from landslide_su_events import load_events  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
WEB = REPO / "webapp" / "change_detect_viewer" / "static" / "poc" / "su"
LS = ("坡面崩塌／坍方", "落石／滾石")
RULES_VERSION = "SUrules-v0"


def jl(p, d=None):
    try:
        return json.load(open(p, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return d


def main():
    WEB.mkdir(parents=True, exist_ok=True)
    tf, shape = grid()
    T = jl(POC / "su_table_eff.json")
    lab = np.load(POC / "su_lab10_eff.npy")
    meta = jl(POC / "su_meta_eff.json")
    mu = {u["su_id"]: u for u in meta["units"]}
    ev = {e["su_id"]: e for e in jl(POC / "su_evidence.json")}
    photos = jl(POC / "su_photos.json")
    pm = jl(POC / "su_photo_meta.json")
    wb = {r["su_id"]: r for r in jl(POC / "wayback_su_depth.json", [])}
    n = len(T)
    m = lab >= 0
    # E3：Sentinel-2 近期新增與河道比例；浮水印遮蔽比例
    late = np.load(POC / "s2" / "det_20250325_A_fixed.npy")
    early = np.load(POC / "s2" / "det_20240404_A_fixed.npy")
    new = late & ~ndi.binary_dilation(early, iterations=1)
    river = np.load(POC / "reverse" / "river.npy")
    v1, v2 = np.load(POC / "s2" / "valid_20240404.npy"), np.load(POC / "s2" / "valid_20250325.npy")
    area_px = np.bincount(lab[m], minlength=n)
    new_ha = np.bincount(lab[m & new], minlength=n) * 0.01
    new_riv = np.bincount(lab[m & new & river], minlength=n) * 0.01
    inv = np.maximum(np.bincount(lab[m & ~v1], minlength=n), np.bincount(lab[m & ~v2], minlength=n)) / np.maximum(area_px, 1)
    # E1：窗口內事件（在每個單元的面積）
    evs = load_events()
    per_ev = defaultdict(list)
    for name, (dt, g) in evs.items():
        em = mask(gpd.GeoDataFrame(geometry=g.values, crs=3826), tf, shape)
        c = np.bincount(lab[m & em], minlength=n)
        for i in np.flatnonzero(c):
            per_ev[int(i)].append((str(dt.date()), name.split("_", 1)[-1][:14], round(float(c[i] * 0.01), 2)))
    # E2：照片（逐筆年份、分類）
    ph_by = defaultdict(list)
    ph_ids = defaultdict(list)
    for p in photos:
        if p["su_id"] and p["id"] in pm:
            ph_by[p["su_id"]].append((str(pm[p["id"]]["year"]), pm[p["id"]]["class"]))
            ph_ids[p["su_id"]].append((str(pm[p["id"]]["year"]), pm[p["id"]]["class"], p["id"], (pm[p["id"]].get("desc") or "")[:40]))
    recs, idx = {}, []
    tiers = Counter()
    reasons = Counter()
    for t in T:
        i = t["i"]
        sid = t["su_id"]
        u = mu[sid]
        pl = ph_by.get(sid, [])
        recent_ls = [y for y, c in pl if c in LS and y >= "2021"]
        events = sorted(per_ev.get(i, []), reverse=True)
        riv_frac = float(new_riv[i] / new_ha[i]) if new_ha[i] > 0 else 0.0
        w = wb.get(sid) or {}
        wb_last = w.get("last")
        why, caution = [], []
        # 資料不足
        insuff = []
        if u["eff_area_ha"] < 1.0:
            insuff.append("有效坡面面積 <1 ha（小於 Sentinel-2 解析度可判斷的尺度）")
        if u["eff_frac"] < 0.2:
            insuff.append("有效坡面僅占單元 %.0f%%（多為緩坡／谷底）" % (u["eff_frac"] * 100))
        if inv[i] >= 0.3:
            insuff.append("Sentinel-2 視窗 %.0f%% 被 Copernicus 浮水印遮蔽，近期變化無從判斷" % (inv[i] * 100))
        # 優先
        if recent_ls:
            why.append("P1 近期現地照片（%s）描述含崩塌／落石 %d 筆" % ("、".join(sorted(set(recent_ls))), len(recent_ls)))
        if t["explained_by_events"] >= 0.3 and t["category"] in ("擴大", "首次出現"):
            why.append("P2 事件目錄連結的新增或擴大（%s，歸因 %.0f%%）" % (t["dominant_event"], t["explained_by_events"] * 100))
        if new_ha[i] >= 1.0 and riv_frac <= 0.5 and t["category"] != "無崩塌":
            why.append("P3 Sentinel-2 近期新增 %.1f ha，河道遮罩內 %.0f%%，年度圖層類別為「%s」" % (new_ha[i], riv_frac * 100, t["category"]))
        tier = "資料不足" if insuff else ("優先人工判讀" if why else "一般判讀")
        if tier == "資料不足" and why:
            caution.append("（有優先指標但資料不足：" + "；".join(insuff) + "）")
        if tier == "資料不足":
            why = insuff + (why if why else [])
        tiers[tier] += 1
        for r_ in why:
            reasons[r_[:2]] += 1
        # 提醒
        if w:
            caution.append("Wayback 在此點 2017 年起 %d 個拍攝日，最新 %s；Wayback 對 DTM／Sentinel-2 常有 10–25 m 偏移，疊圖時輪廓可能錯位。" % (w.get("n_dates", 0), wb_last or "—"))
        else:
            caution.append("無 Wayback 高解析影像紀錄（2017 年起）。")
        if riv_frac > 0.5 and new_ha[i] >= 1.0:
            caution.append("Sentinel-2 近期新增有 %.0f%% 落在河道遮罩內（可能是河床變動；人工驗證中靠河床的樣本 4/4 仍有崩塌，不能直接視為誤報）。" % (riv_frac * 100))
        has_ev = t["category"] != "無崩塌" or pl or events or new_ha[i] >= 0.3
        photo_cls = Counter(c for _, c in pl)
        rec = {"su_id": sid, "uid": u["su_uid"], "tier": tier, "why": why, "caution": caution, "area_ha": u["area_ha"], "eff_ha": u["eff_area_ha"], "eff_frac": u["eff_frac"], "slope": u["mean_slope_deg"],
               "cat": t["category"], "bare": t["bare_ha"], "new_total": t["new_bare_ha_total"], "dom_event": t["dominant_event"], "explained": t["explained_by_events"], "growth_pairs": t["growth_pairs"],
               "events": events[:6], "photos": dict(photo_cls), "photo_years": sorted({y for y, _ in pl}), "s2_new_ha": round(float(new_ha[i]), 2), "s2_riv": round(riv_frac, 2), "s2_invalid": round(float(inv[i]), 2),
               "photo_list": [list(x) for x in sorted(ph_ids.get(sid, []), reverse=True)[:10]], "wb_n": w.get("n_dates"), "wb_last": wb_last, "cx": t["cx"], "cy": t["cy"], "i": i}
        code = ("a" if u["eff_area_ha"] < 1.0 else "") + ("b" if u["eff_frac"] < 0.2 else "") + ("c" if inv[i] >= 0.3 else "")
        idx.append([sid, tier, 1 if has_ev else 0, code])
        if has_ev:
            recs[sid] = rec
    # 輪廓（有紀錄的單元）：逐單元取外環（相鄰單元共邊，不能用全域輪廓）
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    objs = ndi.find_objects(lab + 1, max_label=n)
    geo = {}
    for sid, rec in recs.items():
        i = rec["i"]
        sl = objs[i]
        if sl is None:
            continue
        r0, c0 = sl[0].start, sl[1].start
        sub = np.pad((lab[sl] == i).astype(np.uint8), 1)
        cs, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rings = []
        for c in cs:
            if len(c) < 3:
                continue
            ap = cv2.approxPolyDP(c, 1.6, True)[:, 0, :].astype(float) - 1
            if len(ap) < 3:
                continue
            X, Y = tf.c + (ap[:, 0] + c0 + 0.5) * 10, tf.f - (ap[:, 1] + r0 + 0.5) * 10
            lon, lat = to_ll.transform(X, Y)
            rings.append([[round(float(a_), 5), round(float(b_), 5)] for a_, b_ in zip(lon, lat)])
        geo[sid] = rings
    for sid in recs:
        recs[sid]["ring"] = geo.get(sid, [])
    (WEB / "su_records.json").write_text(json.dumps(recs, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (WEB / "su_index.json").write_text(json.dumps({"rules": RULES_VERSION, "su_version": meta["params"]["su_version"], "n": len(idx), "tiers": dict(tiers), "rows": idx}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    rules = {"version": RULES_VERSION, "tiers": dict(tiers), "n_with_record": len(recs), "priority_reasons": dict(reasons), "doc": __doc__}
    (WEB / "rules.json").write_text(json.dumps(rules, ensure_ascii=False, indent=1), encoding="utf-8")
    print("單元", len(idx), "有完整記錄", len(recs), "層級", dict(tiers), "優先理由計數", dict(reasons))
    for f in ("su_records.json", "su_index.json", "rules.json"):
        print(f, round((WEB / f).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
