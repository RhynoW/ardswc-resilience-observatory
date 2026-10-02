# -*- coding: utf-8 -*-
"""
data/analytics.json 的空間分析 → data/spatial_analysis.json

輸入
  analytics.json 的 grid（0.05° ≈ 5 km 網格，災害事件＋媒體報導的影像筆數）與 event_points（前 40 大災害事件的座標），
  以及 data/ardswc_hotspots/top100_consolidated.json（反覆熱點 top100，用於交叉比對）。
方法（純 numpy／scipy／sklearn，不依賴 esda）
  1. 全域 Moran's I：對 log1p(筆數)，只用有紀錄的網格，距離帶 ≤ 8 km（含斜向鄰格）的二元權重、列標準化；999 次置換檢定。
  2. 局部：Getis-Ord Gi*（熱點／冷點 z 分數）與 LISA 局部 Moran（HH／LL／HL／LH，條件置換 p<0.05）。
  3. 事件空間形態：每個大事件的標準距離（分散度）與標準差橢圓（長軸方位、扁率）；全部事件點的 DBSCAN（haversine, eps 2 km）。
  4. 與 top100 反覆熱點交叉：熱點落在哪個格子、該格 Gi* z、是否落在 HH 群；與「隨機取有紀錄格」的基準比較。
限制（重要）：筆數是「拍了幾張」，受道路可達性、調查人力、災害規模影響，不等於災害發生率；
  Moran／Gi* 反映的是「紀錄密度的空間集中」，不是致災風險。網格以經緯度度數劃分（高緯格較窄，距離以 cos φ 校正）。
輸出另存一份到 webapp/.../static/spatial_analysis.json 供 3D 頁讀取；另輸出 2 km 細格 data/spatial_grid2km.json（見 build_fine_grid）。
用法：python scripts/spatial_analysis.py
"""
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "spatial_analysis.json"
STATIC = REPO / "webapp" / "change_detect_viewer" / "static" / "spatial_analysis.json"      # 3D 頁（/static/spatial3d.html）直接讀這份
RNG = np.random.default_rng(42)
BAND_KM = 8.0
NPERM = 999
MIN_YEAR_N = 300


def km_matrix(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    dlat = la[:, None] - la[None, :]
    dlon = lo[:, None] - lo[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def morans_i(x, W):
    z = x - x.mean()
    return float(len(x) / W.sum() * (z @ W @ z) / (z @ z))


def build_fine_grid(deg=0.02, label="2km"):
    """較細的網格（預設 0.02° ≈ 2 km）：同樣的災害事件＋媒體報導點，只在「陸地格」上算 Getis-Ord Gi*（3×3 鄰格含自身）。
    陸地格＝DTM 高程 > 1 m 或該格有紀錄；海面格不納入，否則零值會讓全島 Gi* 偏高。
    輸出 data/spatial_grid<label>.json（有紀錄的格：[lat, lon, 張數, Gi* z, 格內最低高程, 最高高程]）與網頁用靜態副本。
    注意：陸地格絕大多數沒有紀錄（x=0、變異小），所以有紀錄的格 Gi* 容易偏高；看相對大小與連續性，不要只看「是否 > 1.96」。"""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import dtm20
    from scipy.ndimage import uniform_filter
    ev = json.loads((REPO / "data" / "ardswc_hotspots" / "events_trimmed.json").read_text(encoding="utf-8"))
    P = np.array([(r["lat"], r["lon"]) for r in ev if str(r["photo_type"]) in ("0", "8")])
    P = P[(P[:, 0] > 21.5) & (P[:, 0] < 25.6) & (P[:, 1] > 119.0) & (P[:, 1] < 122.5)]
    iy = np.round(P[:, 0] / deg).astype(int); ix = np.round(P[:, 1] / deg).astype(int)
    y0, y1, x0, x1 = iy.min() - 2, iy.max() + 2, ix.min() - 2, ix.max() + 2
    H, W = y1 - y0 + 1, x1 - x0 + 1
    cnt = np.zeros((H, W)); np.add.at(cnt, (iy - y0, ix - x0), 1)
    gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    dtm = dtm20.Dtm20Source()
    z = dtm.sample_points((gx * deg).ravel(), (gy * deg).ravel()).reshape(H, W)
    land = (z > 1.0) | (cnt > 0)
    x = np.where(land, np.log1p(cnt), 0.0)
    m = land.astype(float)
    swx = uniform_filter(x, 3, mode="constant") * 9.0      # Σ 3×3 的 x（海面格 x=0 不貢獻）
    sw = uniform_filter(m, 3, mode="constant") * 9.0       # 3×3 內的陸地格數（含自身）
    Nn = m.sum(); xbar = x[land].mean(); S = x[land].std()
    with np.errstate(invalid="ignore", divide="ignore"):
        gi = (swx - xbar * sw) / (S * np.sqrt((Nn * sw - sw ** 2) / (Nn - 1)))
    gi = np.where(land, np.nan_to_num(gi), 0.0)
    off = np.array([-0.4, 0.0, 0.4]) * deg
    cells = []
    for r, c in np.argwhere(cnt > 0):
        la, lo = (r + y0) * deg, (c + x0) * deg
        zz = dtm.sample_points((lo + off)[None, :].repeat(3, 0).ravel(), (la + off)[:, None].repeat(3, 1).ravel())
        cells.append([round(la, 3), round(lo, 3), int(cnt[r, c]), round(float(gi[r, c]), 2), int(zz.min()), int(zz.max())])
    out = {"cell_deg": deg, "label": label, "kernel": "3×3（鄰格含自身，約 6 km）", "n_land_cells": int(Nn), "n_cells": len(cells), "n_points": int(cnt.sum()),
           "hot95": int(((gi > 1.96) & (cnt > 0)).sum()), "hot99": int(((gi > 2.576) & (cnt > 0)).sum()),
           "max_n": int(cnt.max()), "fields": ["lat", "lon", "n", "gi_z", "zmin", "zmax"], "cells": cells,
           "note": "2 km 格的 Gi*：陸地格（DTM>1 m 或有紀錄）為母體，3×3 鄰格。筆數仍是拍攝張數，道路可達處偏多；陸地格多為空格，有紀錄的格 Gi* 容易偏高。"}
    txt = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    (REPO / "data" / f"spatial_grid{label}.json").write_text(txt, encoding="utf-8")
    (REPO / "webapp" / "change_detect_viewer" / "static" / f"spatial_grid{label}.json").write_text(txt, encoding="utf-8")
    print(f"{label} 格：有紀錄 {len(cells)} 格（陸地格 {int(Nn)}），Gi* 95% 熱格 {out['hot95']}（99% {out['hot99']}），單格最多 {out['max_n']} 張 → {len(txt) // 1024} KB")


def main():
    A = json.loads((REPO / "data" / "analytics.json").read_text(encoding="utf-8"))
    g = np.array(A["grid"], float)                          # lat, lon, n
    lat, lon, cnt = g[:, 0], g[:, 1], g[:, 2]
    n = len(g)
    x = np.log1p(cnt)
    D = km_matrix(lat, lon)
    Wb = ((D <= BAND_KM) & (D > 0)).astype(float)           # 不含自身
    deg = Wb.sum(1)
    iso = int((deg == 0).sum())
    Wr = Wb / np.where(deg == 0, 1, deg)[:, None]            # 列標準化（孤立格全 0）

    # 1. 全域 Moran's I（置換檢定）
    I = morans_i(x, Wr)
    perm = np.array([morans_i(RNG.permutation(x), Wr) for _ in range(NPERM)])
    EI = -1.0 / (n - 1)
    p_global = float((np.sum(perm >= I) + 1) / (NPERM + 1))
    glob = {"I": round(I, 4), "expected_I": round(EI, 4), "perm_mean": round(float(perm.mean()), 4), "perm_sd": round(float(perm.std()), 4),
            "z_perm": round((I - perm.mean()) / perm.std(), 2), "p_one_sided": round(p_global, 4), "n_cells": n, "isolated_cells": iso,
            "mean_neighbors": round(float(deg.mean()), 2), "band_km": BAND_KM, "variable": "log1p(筆數)"}

    # 2a. Getis-Ord Gi*（含自身的二元權重）
    Ws = Wb + np.eye(n)
    xbar, S = x.mean(), x.std()
    sw = Ws.sum(1); sw2 = (Ws ** 2).sum(1)
    gi = (Ws @ x - xbar * sw) / (S * np.sqrt((n * sw2 - sw ** 2) / (n - 1)))
    # 2b. LISA（條件置換）
    z = (x - xbar) / S
    lag = Wr @ z
    Ii = z * lag
    p_loc = np.ones(n)
    for i in range(n):
        k = int(deg[i])
        if k == 0:
            continue
        others = np.delete(np.arange(n), i)
        samp = z[RNG.choice(others, size=(NPERM, k))]
        sim = z[i] * samp.mean(1)
        p_loc[i] = (np.sum(np.abs(sim) >= abs(Ii[i])) + 1) / (NPERM + 1)
    quad = np.where(z >= 0, np.where(lag >= 0, "HH", "HL"), np.where(lag >= 0, "LH", "LL"))
    lisa = np.where(p_loc < 0.05, quad, "ns")
    hot99, hot95 = gi > 2.576, gi > 1.96

    def cell_rows(mask, top=15):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-cnt[idx])][:top]
        return [{"lat": float(lat[i]), "lon": float(lon[i]), "n": int(cnt[i]), "gi_z": round(float(gi[i]), 2)} for i in idx]

    # 時間分析：逐年／累積的 Gi*、逐年 Moran's I、熱點趨勢分類（資料：events_trimmed.json 的年份欄，災害事件＋媒體報導）
    def gistar(v):
        return (Ws @ v - v.mean() * sw) / (v.std() * np.sqrt((n * sw2 - sw ** 2) / (n - 1)))
    ev = json.loads((REPO / "data" / "ardswc_hotspots" / "events_trimmed.json").read_text(encoding="utf-8"))
    ckey = {(round(float(a), 2), round(float(b), 2)): i for i, (a, b) in enumerate(zip(lat, lon))}
    by_year_all = {}
    pts_year = []
    for r in ev:
        if str(r["photo_type"]) not in ("0", "8"):
            continue
        i = ckey.get((round(round(r["lat"] / 0.05) * 0.05, 2), round(round(r["lon"] / 0.05) * 0.05, 2)))
        if i is None:
            continue
        y = int(r["year"]); by_year_all[y] = by_year_all.get(y, 0) + 1
        pts_year.append((i, y))
    YEARS = sorted(y for y, c in by_year_all.items() if c >= MIN_YEAR_N)         # 筆數太少的年份 Gi*／Moran 不穩定，不納入
    yi = {y: k for k, y in enumerate(YEARS)}
    Y = np.zeros((n, len(YEARS)))
    for i, y in pts_year:
        if y in yi:
            Y[i, yi[y]] += 1
    t_total, t_hot, t_cold, t_moran, t_mp = [], [], [], [], []
    G = np.zeros_like(Y); GC = np.zeros_like(Y)
    cum = np.cumsum(Y, axis=1)
    for k in range(len(YEARS)):
        xk = np.log1p(Y[:, k])
        G[:, k] = gistar(xk); GC[:, k] = gistar(np.log1p(cum[:, k]))
        Ik = morans_i(xk, Wr)
        pk = np.array([morans_i(RNG.permutation(xk), Wr) for _ in range(199)])
        t_total.append(int(Y[:, k].sum())); t_hot.append(int((G[:, k] > 1.96).sum())); t_cold.append(int((G[:, k] < -1.96).sum()))
        t_moran.append(round(Ik, 3)); t_mp.append(round(float((np.sum(pk >= Ik) + 1) / 200), 3))
    hot_mat = G > 1.96
    n_hot_years = hot_mat.sum(1)
    half = len(YEARS) // 2
    recent = hot_mat[:, -3:].sum(1); early = hot_mat[:, :half].sum(1)
    trend = np.full(n, "none", dtype=object)
    trend[(n_hot_years >= max(3, 0.4 * len(YEARS))) & (recent >= 2)] = "persistent"      # 長期且近年仍是熱點
    trend[(recent >= 2) & (early == 0)] = "emerging"                                      # 近 3 年多數為熱點、前半期從未是
    trend[(early >= 2) & (recent == 0)] = "declining"                                     # 前半期曾是熱點、近 3 年不再
    temporal = {"years": YEARS, "min_year_points": MIN_YEAR_N, "total": t_total, "hot95": t_hot, "cold95": t_cold, "moran_I": t_moran, "moran_p": t_mp,
                "trend_counts": {k: int((trend == k).sum()) for k in ("persistent", "emerging", "declining", "none")},
                "note": "單年 Gi*／Moran 以當年 log1p(筆數) 計算；2026 為不完整年份（到抓取日）；趨勢分類為描述性規則，非統計檢定。"}

    # 每格地形高程範圍（5×5 取樣的最低／最高，m）：3D 頁用絕對高度畫柱，避免依賴 Cesium 的 RELATIVE_TO_GROUND（自訂地形上會讓畫面失效）
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import dtm20
    off = np.linspace(-0.0235, 0.0235, 5)
    gl, go = np.meshgrid(off, off)
    qlat = (lat[:, None] + gl.ravel()[None, :]); qlon = (lon[:, None] + go.ravel()[None, :])
    zq = dtm20.Dtm20Source().sample_points(qlon.ravel(), qlat.ravel()).reshape(n, 25)
    cells = [{"lat": round(float(lat[i]), 2), "lon": round(float(lon[i]), 2), "n": int(cnt[i]), "gi_z": round(float(gi[i]), 2), "lisa": str(lisa[i]),
              "zmin": int(zq[i].min()), "zmax": int(zq[i].max()),
              "y": [int(v) for v in Y[i]], "g": [round(float(v), 1) for v in G[i]], "gc": [round(float(v), 1) for v in GC[i]],
              "hy": int(n_hot_years[i]), "trend": str(trend[i])} for i in range(n)]

    # 3. 事件空間形態 + DBSCAN
    pts_all, lab_all = [], []
    events = []
    for ev in A["top_events"]:
        P = np.array(A["event_points"][ev["name"]], float)       # lat, lon
        if len(P) < 5:
            continue
        k = math.cos(math.radians(P[:, 0].mean()))
        XY = np.c_[(P[:, 1] - P[:, 1].mean()) * 111.32 * k, (P[:, 0] - P[:, 0].mean()) * 110.57]
        cov = np.cov(XY.T)
        w, v = np.linalg.eigh(cov)
        major = v[:, 1]
        az = (math.degrees(math.atan2(major[0], major[1])) + 180) % 180      # 長軸方位（北起順時針，0–180）
        events.append({"name": ev["name"], "n": ev["n"], "years": ev["years"], "n_points": len(P),
                       "center": [round(float(P[:, 0].mean()), 4), round(float(P[:, 1].mean()), 4)],
                       "std_distance_km": round(float(np.sqrt((XY ** 2).sum(1).mean())), 1),
                       "ellipse_major_km": round(float(2 * np.sqrt(w[1])), 1), "ellipse_minor_km": round(float(2 * np.sqrt(w[0])), 1),
                       "elongation": round(float(np.sqrt(w[1] / max(w[0], 1e-9))), 2), "major_axis_azimuth_deg": round(az, 0)})
        pts_all.append(P); lab_all += [ev["name"]] * len(P)
    from sklearn.cluster import DBSCAN
    P = np.vstack(pts_all)
    db = DBSCAN(eps=2.0 / 6371.0, min_samples=20, metric="haversine").fit(np.radians(P))
    lab = db.labels_
    clusters = []
    for c in sorted(set(lab) - {-1}):
        m = lab == c
        names = {}
        for nm in np.array(lab_all)[m]:
            names[nm] = names.get(nm, 0) + 1
        clusters.append({"id": int(c), "n": int(m.sum()), "center": [round(float(P[m, 0].mean()), 4), round(float(P[m, 1].mean()), 4)],
                         "top_events": sorted(names.items(), key=lambda kv: -kv[1])[:3]})
    clusters.sort(key=lambda c: -c["n"])

    # 4. 與 top100 反覆熱點交叉
    H = json.loads((REPO / "data" / "ardswc_hotspots" / "top100_consolidated.json").read_text(encoding="utf-8"))
    cell_idx = {(round(float(a), 2), round(float(b), 2)): i for i, (a, b) in enumerate(zip(lat, lon))}
    hs = []
    for h in H:
        key = (round(round(h["lat"] / 0.05) * 0.05, 2), round(round(h["lon"] / 0.05) * 0.05, 2))
        i = cell_idx.get(key)
        hs.append({"rank": h["rank"], "lat": round(h["lat"], 5), "lon": round(h["lon"], 5), "county": h.get("county"), "district": h.get("district"), "n_events": h["n_independent_events"],
                   "cell_n": int(cnt[i]) if i is not None else None, "gi_z": round(float(gi[i]), 2) if i is not None else None,
                   "lisa": str(lisa[i]) if i is not None else None})
    has = [r for r in hs if r["gi_z"] is not None]
    in_hot = np.mean([r["gi_z"] > 1.96 for r in has]); in_hh = np.mean([r["lisa"] == "HH" for r in has])
    base_hot, base_hh = float(hot95.mean()), float((lisa == "HH").mean())
    from scipy.stats import spearmanr
    per_cell = {}
    for r in H:
        key = (round(round(r["lat"] / 0.05) * 0.05, 2), round(round(r["lon"] / 0.05) * 0.05, 2))
        per_cell[key] = per_cell.get(key, 0) + 1
    xs = [cnt[cell_idx[k]] for k in per_cell if k in cell_idx]; ys = [per_cell[k] for k in per_cell if k in cell_idx]
    rho, prho = spearmanr(xs, ys)
    cross = {"n_hotspots": len(H), "n_matched_cells": len(has), "share_in_gi95_hot_cells": round(float(in_hot), 3), "baseline_share_hot_cells": round(base_hot, 3),
             "share_in_HH_cluster": round(float(in_hh), 3), "baseline_share_HH": round(base_hh, 3),
             "spearman_photo_count_vs_hotspots_per_cell": {"rho": round(float(rho), 3), "p": round(float(prho), 4), "n_cells": len(xs)},
             "hotspots_outside_hot_cells": [r for r in has if r["gi_z"] <= 1.96][:20]}

    res = {"generated_from": "data/analytics.json（grid、event_points）＋ top100_consolidated.json", "generated": A.get("generated"),
           "method_note": "筆數是拍攝張數，受可達性與調查人力影響；以下是『紀錄密度』的空間集中，不是致災風險。",
           "global_moran": glob,
           "gi_star": {"hot_99": int(hot99.sum()), "hot_95": int(hot95.sum()), "cold_95": int((gi < -1.96).sum()), "n_cells": n,
                       "top_hot_cells": cell_rows(hot95)},
           "lisa": {k: int((lisa == k).sum()) for k in ("HH", "LL", "HL", "LH", "ns")},
           "events": events, "dbscan": {"eps_km": 2.0, "min_samples": 20, "n_points": int(len(P)), "n_clusters": len(clusters),
                                        "noise_share": round(float((lab == -1).mean()), 3), "clusters": clusters[:25]},
           "temporal": temporal, "top100_crosscheck": cross, "top100_hotspots": hs, "cells": cells}
    OUT.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    STATIC.write_bytes(OUT.read_bytes())
    print(f"Moran's I = {glob['I']} (z={glob['z_perm']}, p={glob['p_one_sided']}), 孤立格 {iso}")
    print("Gi*: 熱點(95%)", int(hot95.sum()), "（99%", int(hot99.sum()), "）冷點", int((gi < -1.96).sum()), "；LISA", res["lisa"])
    print("時間：年份 %d–%d（%d 年，每年≥%d 筆）；逐年熱格 %s；趨勢 %s" % (YEARS[0], YEARS[-1], len(YEARS), MIN_YEAR_N, t_hot, temporal["trend_counts"]))
    print("事件 %d 個；DBSCAN %d 群，雜訊 %.0f%%" % (len(events), len(clusters), 100 * (lab == -1).mean()))
    print("top100 熱點落在 Gi* 熱格 %.0f%%（基準 %.0f%%）；落在 HH 群 %.0f%%（基準 %.0f%%）；Spearman ρ=%.2f (p=%.3f)"
          % (100 * in_hot, 100 * base_hot, 100 * in_hh, 100 * base_hh, rho, prho))
    print("→", OUT, OUT.stat().st_size // 1024, "KB")
    build_fine_grid(0.02, "2km")


if __name__ == "__main__":
    main()
