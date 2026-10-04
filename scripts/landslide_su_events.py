"""POC：四年崩塌地圖層 × 事件型崩塌目錄 × Slope Unit × 大規模崩塌潛勢區。

前置：landslide_incremental.py（incremental_masks.npz）、slope_units.py（su_labels.npz、su_meta.json）、
      官方判釋 data/biggis_interp/polygons.json、大規模崩塌潛勢區 data/biggis_interp/shp/ls/115_potential。
輸出：data/biggis_interp/poc/su_events.json

1. 事件歸因：每期圖層的影像日期很集中（2021≈7 月、2022≈6 月下旬–7 月、2023≈3–11 月（中位 7 月）、「113 年度」＝2025-03 至 04），
   所以兩期之間的事件窗口以「前後兩期影像日期中位數」界定；落在任一期影像日期分布內的事件標為「日期不確定」。
   新增裸露（後期有、前期無）是否落在窗口內事件的判釋多邊形（含 30 m 緩衝）；並以「其他窗口的事件」當對照。
2. Slope Unit：每個 SU 的逐年裸露面積、全期類別（首次出現／擴大／縮減／持續裸露／消失）、擴大主因事件。
3. 大規模崩塌潛勢區：獨立對照——有潛勢區的 SU 與其他 SU 在裸露、擴大上的差異（控制坡度後也看）。
限制：事件判釋是衛星判釋；年度圖層與事件判釋都來自水保署體系，並非完全獨立。
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landslide_incremental import BBOX_LL, RES, grid, mask  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
YEARS = [2021, 2022, 2023, 2024]
MED = {2021: "2021-07-01", 2022: "2022-06-23", 2023: "2023-07-12", 2024: "2025-03-01"}      # 各期影像日期中位數
SPREAD = {2021: ("2021-07-01", "2021-08-27"), 2022: ("2022-06-23", "2022-07-20"),
          2023: ("2023-03-17", "2023-09-14"), 2024: ("2025-03-01", "2025-04-15")}             # 影像日期 10%–90% 範圍
PX_HA = RES * RES / 1e4
MINH = 0.5
THR = 0.20


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_events():
    d = json.load(open(REPO / "data" / "biggis_interp" / "polygons.json", encoding="utf-8"))
    ev = d["events"]
    by = defaultdict(list)
    meta = {}
    for p in d["polys"]:
        e = ev[p[0]]
        dt = pd.Timestamp(e["date"])
        if not (pd.Timestamp("2021-07-02") <= dt <= pd.Timestamp("2025-03-01")):
            continue
        xy = p[2]
        by[e["event"]].append(Polygon(list(zip(xy[0::2], xy[1::2]))).buffer(0))
        meta[e["event"]] = dt
    out = {}
    for name, polys in by.items():
        g = gpd.GeoSeries(polys, crs=4326).to_crs(3826)
        out[name] = (meta[name], g)
    return out


def window(a, b):
    lo, hi = pd.Timestamp(MED[a]), pd.Timestamp(MED[b])
    return lo, hi


def ambiguous(dt, a, b):
    for y in (a, b):
        s0, s1 = SPREAD[y]
        if pd.Timestamp(s0) <= dt <= pd.Timestamp(s1):
            return True
    return False


def classify_components(prev, cur, thr):
    lab, n = ndi.label(prev | cur, structure=np.ones((3, 3)))
    idx = np.arange(1, n + 1)
    ap = ndi.sum(prev, lab, idx)
    ac = ndi.sum(cur, lab, idx)
    cat = np.full(n, "持續裸露", dtype=object)
    both = (ap > 0) & (ac > 0)
    ch = (ac - ap) / np.where(ap > 0, ap, 1)
    cat[both & (ch > thr)] = "擴大"
    cat[both & (ch < -thr)] = "縮減"
    cat[ap == 0] = "首次出現"
    cat[ac == 0] = "消失"
    return lab, n, cat


def main():
    tf, shape = grid()
    Z = np.load(POC / "incremental_masks.npz")
    M = {y: Z[str(y)] for y in YEARS}
    events = load_events()
    log("事件（2021-07 至 2025-03，依事件名稱合併）:", len(events))
    R = {}
    dil = {}
    for name, (dt, g) in events.items():
        R[name] = mask(gpd.GeoDataFrame(geometry=g.values, crs=3826), tf, shape)
        dil[name] = ndi.binary_dilation(R[name], iterations=3)
    res = {"windows": {}, "events": {n: {"date": str(v[0].date()), "area_ha_in_poc": round(float(R[n].sum() * PX_HA), 1)} for n, v in events.items()}}

    # ── SU 標籤對位到 10 m 格網
    meta = json.load(open(POC / "su_meta.json", encoding="utf-8"))
    L = np.load(POC / "su_labels.npz")
    key20, sx0, sy0 = L["key20"], float(L["x0"]), float(L["y0"])
    h, w = shape
    x = tf.c + (np.arange(w) + 0.5) * RES
    y = tf.f - (np.arange(h) + 0.5) * RES
    cols = np.floor((x - sx0) / 20.0).astype(int)
    rows = np.floor((sy0 - y) / 20.0).astype(int)
    ok_c = (cols >= 0) & (cols < key20.shape[1])
    ok_r = (rows >= 0) & (rows < key20.shape[0])
    suk = np.full(shape, -1, dtype=np.int64)
    rr, cc = np.nonzero(ok_r[:, None] & ok_c[None, :])
    suk[rr, cc] = key20[rows[rr], cols[cc]]
    uniq = np.unique(suk[suk >= 0])
    lab_su = np.where(suk >= 0, np.searchsorted(uniq, suk), -1)
    n_su = len(uniq)
    uinfo = {u["key"]: u for u in meta["units"]}
    log("SU 對位", n_su, "個單元（POC 範圍內）")

    # ── 1. 事件歸因
    pairs = [(2021, 2022), (2022, 2023), (2023, 2024), (2021, 2024)]
    new_by_event = {}
    for a, b in pairs:
        lo, hi = window(a, b)
        win_ev = [n for n, (dt, _) in events.items() if lo < dt <= hi]
        oth_ev = [n for n in events if n not in win_ev]
        new = M[b] & ~M[a]
        avail = ~M[a]
        U = np.zeros(shape, bool)
        for n in win_ev:
            U |= dil[n]
        Uo = np.zeros(shape, bool)
        for n in oth_ev:
            Uo |= dil[n]
        p_in = float(new[U & avail].sum() / max((U & avail).sum(), 1))
        p_out = float(new[~U & avail].sum() / max((~U & avail).sum(), 1))
        p_oth = float(new[Uo & ~U & avail].sum() / max((Uo & ~U & avail).sum(), 1))
        per_ev = {}
        for n in win_ev:
            m = new & dil[n]
            per_ev[n] = {"new_ha_inside": round(float(m.sum() * PX_HA), 1), "date": str(events[n][0].date()), "ambiguous_date": ambiguous(events[n][0], a, b)}
        k = f"{a}-{b}"
        new_by_event[k] = {n: (new & dil[n]) for n in win_ev}
        res["windows"][k] = {
            "from": str(lo.date()), "to": str(hi.date()), "events_in_window": len(win_ev), "new_ha": round(float(new.sum() * PX_HA), 1),
            "new_ha_in_window_events(30m)": round(float((new & U).sum() * PX_HA), 1),
            "share_new_explained": round(float((new & U).sum() / max(new.sum(), 1)), 3),
            "P(new | inside window events)": round(p_in, 4), "P(new | outside any event)": round(p_out, 4),
            "P(new | inside OTHER-window events only)": round(p_oth, 4),
            "ratio_in_vs_out": round(p_in / p_out, 2) if p_out else None,
            "ratio_other_vs_out": round(p_oth / p_out, 2) if p_out else None,
            "per_event": dict(sorted(per_ev.items(), key=lambda kv: -kv[1]["new_ha_inside"])),
        }
        log(k, "窗口事件", len(win_ev), "新增", res["windows"][k]["new_ha"], "ha；事件內", res["windows"][k]["new_ha_in_window_events(30m)"], "ha；勝算比", res["windows"][k]["ratio_in_vs_out"], "對照(其他窗口)", res["windows"][k]["ratio_other_vs_out"])

        # 連通區塊：擴大／首次出現的主因事件
        lab, n, cat = classify_components(M[a], M[b], THR)
        idx = np.arange(1, n + 1)
        tally = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
        newpx_total = ndi.sum(new, lab, idx)
        ev_px = {nme: ndi.sum(new_by_event[k][nme], lab, idx) for nme in win_ev}
        if win_ev:
            mat = np.vstack([ev_px[nme] for nme in win_ev])
            best = mat.argmax(0)
            bestv = mat.max(0)
        for i in range(n):
            c = cat[i]
            if c not in ("擴大", "首次出現"):
                continue
            tot = newpx_total[i]
            if tot <= 0:
                continue
            if win_ev and bestv[i] / tot >= 0.30:
                who = win_ev[best[i]]
            else:
                who = "未歸因（無窗口內事件涵蓋）"
            tally[c][who][0] += 1
            tally[c][who][1] += tot * PX_HA
        res["windows"][k]["components"] = {c: {w_: {"n": v[0], "new_ha": round(v[1], 1)} for w_, v in sorted(d.items(), key=lambda kv: -kv[1][1])} for c, d in tally.items()}

    # ── 2. Slope Unit 履歷
    bare = {}
    for yv in YEARS:
        bare[yv] = np.bincount(lab_su[M[yv] & (lab_su >= 0)], minlength=n_su) * PX_HA
    area_su = np.bincount(lab_su[lab_su >= 0], minlength=n_su) * PX_HA
    chain_new = np.zeros(n_su)
    ev_su = defaultdict(lambda: np.zeros(n_su))
    for (a, b), k in zip(pairs[:3], ["2021-2022", "2022-2023", "2023-2024"]):
        new = M[b] & ~M[a]
        chain_new += np.bincount(lab_su[new & (lab_su >= 0)], minlength=n_su) * PX_HA
        for nme, m in new_by_event[k].items():
            ev_su[nme] += np.bincount(lab_su[m & (lab_su >= 0)], minlength=n_su) * PX_HA
    b21, b24 = bare[2021], bare[2024]
    cat = np.full(n_su, "無崩塌", dtype=object)
    for i in range(n_su):
        x21, x24 = b21[i], b24[i]
        if x21 < MINH and x24 < MINH:
            c = "無崩塌" if max(bare[y][i] for y in YEARS) < MINH else "短暫出現"
        elif x21 < MINH:
            c = "首次出現"
        elif x24 < MINH:
            c = "消失"
        else:
            r = (x24 - x21) / x21
            c = "擴大" if r > THR else ("縮減" if r < -THR else "持續裸露")
        cat[i] = c
    grow = np.zeros(n_su, int)
    for a, b in pairs[:3]:
        d = bare[b] - bare[a]
        grow += ((d >= MINH) & (bare[b] > bare[a] * (1 + THR))).astype(int)
    ev_names = list(ev_su.keys())
    dom = np.array([""] * n_su, dtype=object)
    expl = np.zeros(n_su)
    if ev_names:
        mat = np.vstack([ev_su[n_] for n_ in ev_names])
        tot_ev = mat.sum(0)
        dom = np.where(mat.max(0) > 0, np.array(ev_names, dtype=object)[mat.argmax(0)], "")
        expl = np.where(chain_new > 0, np.minimum(tot_ev / np.maximum(chain_new, 1e-9), 1.0), 0)
    sel = [i for i in range(n_su) if cat[i] != "無崩塌"]
    cnt = Counter(cat[sel].tolist())
    ha = defaultdict(float)
    for i in sel:
        ha[cat[i]] += b24[i] if cat[i] != "消失" else b21[i]
    res["su"] = {"n_units": int(n_su), "n_with_bare": len(sel), "category_counts": dict(cnt), "category_ha(2024; 消失用2021)": {k_: round(v, 1) for k_, v in ha.items()},
                 "area_ha_quantiles(5/25/50/75/95)": np.percentile(area_su[area_su > 0], [5, 25, 50, 75, 95]).round(1).tolist(),
                 "growth_pairs_hist": dict(Counter(grow[sel].tolist()))}
    dom_tally = defaultdict(lambda: defaultdict(int))
    for i in sel:
        if cat[i] in ("擴大", "首次出現"):
            dom_tally[cat[i]][dom[i] if (dom[i] and expl[i] >= 0.3) else "未歸因"] += 1
    res["su"]["dominant_event_of_growth_SUs"] = {c: dict(sorted(d.items(), key=lambda kv: -kv[1])) for c, d in dom_tally.items()}
    top = sorted(sel, key=lambda i: -(chain_new[i]))[:40]
    res["su"]["top_units_by_new_bare_ha"] = [
        {"su_id": uinfo[int(uniq[i])]["su_id"], "area_ha": round(float(area_su[i]), 1), "slope_deg": uinfo[int(uniq[i])]["mean_slope_deg"],
         "bare_ha": {str(y_): round(float(bare[y_][i]), 1) for y_ in YEARS}, "category": cat[i], "growth_pairs": int(grow[i]),
         "new_bare_ha_total": round(float(chain_new[i]), 1), "dominant_event": dom[i] or None, "explained_by_events": round(float(expl[i]), 2)} for i in top]
    log("SU 類別：", dict(cnt))

    # SU 逐單元表與標籤（供 su_wayback_sample.py 抽樣驗證）
    union_any = np.zeros(shape, bool)
    for yv in YEARS:
        union_any |= M[yv]
    xs = tf.c + (np.arange(w) + 0.5) * RES
    ys = tf.f - (np.arange(h) + 0.5) * RES
    vm = lab_su >= 0
    r_, c_ = np.nonzero(union_any & vm)
    cnt_b = np.bincount(lab_su[r_, c_], minlength=n_su)
    cx_b = np.bincount(lab_su[r_, c_], weights=xs[c_], minlength=n_su) / np.maximum(cnt_b, 1)
    cy_b = np.bincount(lab_su[r_, c_], weights=ys[r_], minlength=n_su) / np.maximum(cnt_b, 1)
    r2, c2 = np.nonzero(vm)
    cnt_a = np.bincount(lab_su[r2, c2], minlength=n_su)
    cx_a = np.bincount(lab_su[r2, c2], weights=xs[c2], minlength=n_su) / np.maximum(cnt_a, 1)
    cy_a = np.bincount(lab_su[r2, c2], weights=ys[r2], minlength=n_su) / np.maximum(cnt_a, 1)
    table = []
    for i in range(n_su):
        has = cnt_b[i] > 0
        table.append({"i": i, "su_id": uinfo[int(uniq[i])]["su_id"], "category": cat[i], "area_ha": round(float(area_su[i]), 2),
                      "slope_deg": uinfo[int(uniq[i])]["mean_slope_deg"], "bare_ha": {str(y_): round(float(bare[y_][i]), 2) for y_ in YEARS},
                      "growth_pairs": int(grow[i]), "new_bare_ha_total": round(float(chain_new[i]), 2), "dominant_event": dom[i] or None,
                      "explained_by_events": round(float(expl[i]), 2), "cx": round(float(cx_b[i] if has else cx_a[i]), 1), "cy": round(float(cy_b[i] if has else cy_a[i]), 1)})
    (POC / "su_table.json").write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    np.save(POC / "su_lab10.npy", lab_su.astype(np.int32))
    log("SU 逐單元表 →", len(table), "筆")

    # ── 3. 大規模崩塌潛勢區（獨立對照）
    import glob
    shp = glob.glob(str(REPO / "data/biggis_interp/shp/ls/115_potential/**/*.shp"), recursive=True)[0]
    pz = gpd.read_file(shp, encoding="utf-8")
    from shapely.geometry import box
    poc_box = gpd.GeoSeries([box(*BBOX_LL)], crs=4326).to_crs(3826).iloc[0]
    pz = pz[pz.intersects(poc_box)].copy()
    log("POC 內大規模崩塌潛勢區", len(pz))
    pzm = mask(pz, tf, shape)
    in_pz = np.bincount(lab_su[pzm & (lab_su >= 0)], minlength=n_su) * PX_HA
    has_pz = in_pz >= 1.0                      # SU 內至少 1 ha 位於潛勢區
    slope = np.array([uinfo[int(u)]["mean_slope_deg"] for u in uniq])
    rows_ = []
    for name, m in (("有潛勢區的 SU", has_pz), ("其他 SU", ~has_pz)):
        s = [i for i in range(n_su) if m[i] and area_su[i] >= 3.0]
        cs = Counter(cat[s].tolist())
        rows_.append({"group": name, "n": len(s), "median_slope_deg": round(float(np.median(slope[s])), 1) if s else None,
                      "share_with_bare(≥0.5ha any year)": round(1 - cs.get("無崩塌", 0) / max(len(s), 1), 3),
                      "mean_bare_frac_2024": round(float(np.mean(b24[s] / area_su[s])), 4) if s else None,
                      "share_growth(擴大+首次出現)": round((cs.get("擴大", 0) + cs.get("首次出現", 0)) / max(len(s), 1), 3),
                      "categories": dict(cs)})
    # 控制坡度：只比較平均坡度 > 25° 的 SU
    steep = slope > 25
    ctl = []
    for name, m in (("有潛勢區（坡度>25°）", has_pz & steep), ("其他（坡度>25°）", ~has_pz & steep)):
        s = [i for i in range(n_su) if m[i] and area_su[i] >= 3.0]
        cs = Counter(cat[s].tolist())
        ctl.append({"group": name, "n": len(s), "share_with_bare": round(1 - cs.get("無崩塌", 0) / max(len(s), 1), 3),
                    "mean_bare_frac_2024": round(float(np.mean(b24[s] / area_su[s])), 4) if s else None,
                    "share_growth": round((cs.get("擴大", 0) + cs.get("首次出現", 0)) / max(len(s), 1), 3)})
    zones = []
    for _, z in pz.iterrows():
        zm = mask(gpd.GeoDataFrame(geometry=[z.geometry], crs=3826), tf, shape)
        zones.append({"lslno": str(z.get("lslno")), "name": str(z.get("Name")), "town": f"{z.get('County01')}{z.get('Town01')}", "risk": str(z.get("Risk")),
                      "households": int(z.get("Dw_count") or 0) if str(z.get("Dw_count")).replace('.', '').isdigit() else None,
                      "area_ha": round(float(zm.sum() * PX_HA), 1),
                      "bare_ha": {str(y_): round(float((M[y_] & zm).sum() * PX_HA), 1) for y_ in YEARS},
                      "new_ha_2021_2024_chain": round(float(sum(((M[b_] & ~M[a_]) & zm).sum() for a_, b_ in pairs[:3]) * PX_HA), 1)})
    res["potential_zones"] = {"n_in_poc": len(pz), "su_groups": rows_, "steep_control": ctl, "zones": zones,
                              "bare_share_inside_zones_2024": round(float((M[2024] & pzm).sum() / max(pzm.sum(), 1)), 4),
                              "bare_share_poc_2024": round(float(M[2024].sum() / (h * w)), 4)}
    (POC / "su_events.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    log("完成 → su_events.json")


if __name__ == "__main__":
    main()
