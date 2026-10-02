# -*- coding: utf-8 -*-
"""
預警輔助訓練：四個控制曝險偏差的實驗（接 warning_trainer.py 的樣本）。

  prep     由熱點事件日期建 case 表：每個 (熱點, 事件) 一列 kind=event，並在「同一地點」前後各 30 天建一個對照 kind=ctrl
  gee      以 GEE 補欄位：built_frac_1km（GHSL 2020 建成面積比，當曝險代理）、SMAP L4 根系層／表層土壤含水量
           （事件日前 3 天均值、前 30 天均值）。專案 ID 用 GEE_PROJECT。
  analyze  E1 時間切片：潛勢溪流圖是 2021 年版——事件發生於 2022 年後的熱點，圖層不可能「看過」，用來檢查資訊洩漏
           E2 曝險分層：依 built_frac 分三層，各層分別量測圖層的 AUC（權重是否隨曝險而變）
           E3 case-crossover：同地點、事件日 vs 前後 30 天，比較 SMAP 含水量——地點相同，曝險偏差被消掉，只剩「觸發」
           E4 曝險配對對照：負例按 built_frac 分層重抽，使正負例曝險分布一致後再看圖層 AUC

限制：SMAP L4 約 9 km，無法分辨邊坡；GEE 的 L4 資料有約一年延遲；事件日期為通報日而非崩塌發生日（可能晚數日）；
  每個熱點多個事件在統計上不獨立；對照日可能剛好落在另一次事件內（未排除）。
用法：python scripts/warning_experiments.py prep --hotspots <list.json>
      python scripts/warning_experiments.py gee
      python scripts/warning_experiments.py analyze
"""
import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "warning_trainer"
SMAP_FROM, SMAP_TO = dt.date(2016, 1, 1), dt.date(2024, 12, 31)


def prep(a):
    hs = json.loads(Path(a.hotspots).read_text(encoding="utf-8"))
    rows = []
    for h in hs:
        for k, e in enumerate(h["events"]):
            d0 = dt.date.fromisoformat(e["first_date"])
            if not (SMAP_FROM <= d0 <= SMAP_TO):
                continue
            for kind, off in (("event", 0), ("ctrl", -30), ("ctrl", 30)):
                d = d0 + dt.timedelta(days=off)
                rows.append({"id": f"{h['rank']}_{k}_{kind}{off:+d}", "rank": h["rank"], "kind": kind, "lon": h["lon"], "lat": h["lat"],
                             "date": d.isoformat(), "event_year": d0.year})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(len(rows), "列 →", OUT / "cases.csv", "（事件", sum(r["kind"] == "event" for r in rows), "）")


def gee(a):
    import ee
    ee.Initialize(project=a.project or os.environ.get("GEE_PROJECT"))
    smap = None
    for ver in ("007", "008"):
        c = ee.ImageCollection(f"NASA/SMAP/SPL4SMGP/{ver}")
        if c.limit(1).size().getInfo():
            smap = c
            print("SMAP L4 版本", ver)
            break
    built = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020").select("built_surface")
    rows = list(csv.DictReader(open(OUT / "cases.csv", encoding="utf-8")))
    sm_rows = []
    for i in range(0, len(rows), 100):
        chunk = rows[i:i + 100]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                       {"id": r["id"], "t": int(dt.datetime.fromisoformat(r["date"]).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)})
            for r in chunk])

        def f(ft):
            t = ee.Date(ft.get("t"))
            pt = ft.geometry()

            def mean(band, days, lag=0):
                # lag：窗尾落後事件日幾天。通報日已含觸發降雨，lag=3 的窗才是「前期」含水量
                img = smap.filterDate(t.advance(-days - lag, "day"), t.advance(1 - lag, "day")).select(band).mean()
                return img.reduceRegion(ee.Reducer.first(), pt, 9000).get(band)
            b = built.reduceRegion(ee.Reducer.mean(), pt.buffer(1000), 100).get("built_surface")
            return ft.set({"sm_rz_3d": mean("sm_rootzone", 3), "sm_rz_30d": mean("sm_rootzone", 30),
                           "sm_surf_3d": mean("sm_surface", 3),
                           "sm_rz_ante3d": mean("sm_rootzone", 3, 3), "sm_rz_ante30d": mean("sm_rootzone", 30, 3), "sm_surf_ante3d": mean("sm_surface", 3, 3), "built_frac_1km": ee.Number(b).divide(10000)})
        for ft in fc.map(f).getInfo()["features"]:
            sm_rows.append(ft["properties"])
        print(f"  GEE {min(i + 100, len(rows))}/{len(rows)}", flush=True)
    ids = {r["id"]: r for r in rows}
    cols = ["id", "rank", "kind", "lon", "lat", "date", "event_year", "sm_rz_3d", "sm_rz_30d", "sm_surf_3d", "sm_rz_ante3d", "sm_rz_ante30d", "sm_surf_ante3d", "built_frac_1km"]
    with open(OUT / "cases_gee.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in sm_rows:
            r = {**ids[p["id"]], **p}
            w.writerow({k: r.get(k) for k in cols})
    samp = list(csv.DictReader(open(OUT / "samples.csv", encoding="utf-8")))
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]).buffer(1000), {"id": r["id"]}) for r in samp])
    res = built.reduceRegions(fc, ee.Reducer.mean(), 100).getInfo()["features"]
    with open(OUT / "samples_built.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "built_frac_1km"])
        for ft in res:
            v = ft["properties"].get("mean")
            w.writerow([ft["properties"]["id"], "" if v is None else v / 10000])
    print("→ cases_gee.csv, samples_built.csv")


def rain(a):
    """GEE：NASA GPM IMERG V07（0.1°、30 分鐘，mm/hr）→ 每個 case 的累積雨量：
    r3d（含通報日的前 3 天）、r7d、r30d、rmax1d（3 天內單日最大）、ante30d（通報前 3 天為止的 30 天，前期濕度代理）。"""
    import ee
    ee.Initialize(project=a.project or os.environ.get("GEE_PROJECT"))
    col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
    rows = list(csv.DictReader(open(OUT / "cases.csv", encoding="utf-8")))
    res_rows = []
    for i in range(0, len(rows), 100):
        chunk = rows[i:i + 100]
        fc = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([float(r["lon"]), float(r["lat"])]),
                       {"id": r["id"], "t": int(dt.datetime.fromisoformat(r["date"]).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)})
            for r in chunk])

        def f(ft):
            t = ee.Date(ft.get("t"))
            pt = ft.geometry()

            def tot(d0, d1):          # 日期區間 [t+d0, t+d1) 的累積雨量（mm）
                img = col.filterDate(t.advance(d0, "day"), t.advance(d1, "day")).sum().multiply(0.5)
                return ee.Number(img.reduceRegion(ee.Reducer.first(), pt, 11000).get("precipitation"))
            days = ee.List([tot(-2, -1), tot(-1, 0), tot(0, 1)])
            return ft.set({"r3d": tot(-2, 1), "r7d": tot(-6, 1), "r30d": tot(-29, 1), "rmax1d": days.reduce(ee.Reducer.max()),
                           "ante30d": tot(-32, -2)})
        for ft in fc.map(f).getInfo()["features"]:
            res_rows.append(ft["properties"])
        print(f"  IMERG {min(i + 100, len(rows))}/{len(rows)}", flush=True)
    cols = ["id", "r3d", "r7d", "r30d", "rmax1d", "ante30d"]
    with open(OUT / "cases_rain.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for p in res_rows:
            w.writerow({k: p.get(k) for k in cols})
    print("→ cases_rain.csv")


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(set(y)) == 2 else float("nan")


def analyze(a):
    out = {}
    samp = {r["id"]: r for r in csv.DictReader(open(OUT / "samples.csv", encoding="utf-8"))}
    lay = {r["id"]: r for r in csv.DictReader(open(OUT / "layers.csv", encoding="utf-8"))}
    bu = {r["id"]: r for r in csv.DictReader(open(OUT / "samples_built.csv", encoding="utf-8"))}
    ev = {}
    for r in csv.DictReader(open(OUT / "cases_gee.csv", encoding="utf-8")):
        if r["kind"] == "event":
            ev.setdefault(r["rank"], []).append(int(r["event_year"]))
    ids = [i for i in samp if i in lay and bu.get(i, {}).get("built_frac_1km") not in (None, "")]
    y = np.array([int(samp[i]["y"]) for i in ids])
    dist = np.array([float(lay[i]["debris_dist_km"]) for i in ids])
    slope = np.array([float(samp[i]["slope_mean_deg"]) for i in ids])
    built = np.array([float(bu[i]["built_frac_1km"]) for i in ids])
    last = np.array([max(ev.get(i, [0])) if samp[i]["y"] == "1" else 0 for i in ids])

    neg = y == 0
    for name, m in (("last_event<=2021", (y == 1) & (last <= 2021)), ("last_event>=2022", (y == 1) & (last >= 2022))):
        sel = m | neg
        out.setdefault("E1_time_slice", {})[name] = {"n_pos": int(m.sum()), "auc_debris_dist": round(auc(y[sel], -dist[sel]), 3)}
    print("E1", out["E1_time_slice"])

    # 建成比例大量為 0（山區），分位數會並列 → 手動分層：0／>0 的下半／>0 的上半
    pos_med = float(np.median(built[built > 0.001])) if (built > 0.001).any() else 0.01
    strata = np.where(built <= 0.001, 0, np.where(built <= pos_med, 1, 2))
    for k, nm in enumerate(("none", "low", "high")):
        m = strata == k
        if m.sum() == 0 or len(set(y[m])) < 2:
            continue
        out.setdefault("E2_exposure_strata", {})[nm] = {
            "built_range": [round(float(built[m].min()), 3), round(float(built[m].max()), 3)],
            "n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()),
            "auc_debris_dist": round(auc(y[m], -dist[m]), 3), "auc_slope_low": round(auc(y[m], -slope[m]), 3)}
    out["E2_auc_of_builtfrac_alone"] = round(auc(y, built), 3)
    print("E2", json.dumps(out["E2_exposure_strata"], ensure_ascii=False), "built 本身的 AUC", out["E2_auc_of_builtfrac_alone"])

    rng = np.random.default_rng(3)
    keep = []
    for k in range(3):
        p = np.where((strata == k) & (y == 1))[0]
        n = np.where((strata == k) & (y == 0))[0]
        if len(p) == 0 or len(n) == 0:
            continue
        keep += list(p) + list(rng.choice(n, size=min(len(n), len(p)), replace=False))
    keep = np.array(keep)
    out["E4_exposure_matched"] = {"n": int(len(keep)), "auc_debris_dist": round(auc(y[keep], -dist[keep]), 3),
                                  "auc_slope_low": round(auc(y[keep], -slope[keep]), 3), "auc_builtfrac": round(auc(y[keep], built[keep]), 3)}
    print("E4", out["E4_exposure_matched"])

    tp = OUT / "tiles.csv"
    if tp.exists():                                    # 地調所圖磚圖層（tile_layer_features.py）
        tl = {r["id"]: r for r in csv.DictReader(open(tp, encoding="utf-8"))}
        res = {}
        for col in ("geosens_in", "dislope_in", "histls_in"):
            v = np.array([int(tl[i][col]) for i in ids])
            res[col] = {"in_frac_pos": round(float(v[y == 1].mean()), 3), "in_frac_neg": round(float(v[y == 0].mean()), 3),
                        "auc_all": round(auc(y, v), 3), "auc_exposure_matched": round(auc(y[keep], v[keep]), 3)}
        out["tile_layers"] = res
        print("TILES", json.dumps(res, ensure_ascii=False))
    rows = list(csv.DictReader(open(OUT / "cases_gee.csv", encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault(r["id"].rsplit("_", 1)[0], []).append(r)
    res = {}
    from scipy.stats import binomtest
    for var in ("sm_rz_3d", "sm_surf_3d", "sm_rz_ante3d", "sm_rz_ante30d", "sm_surf_ante3d"):
        diffs = []
        for g in by.values():
            e = [r for r in g if r["kind"] == "event"]
            c = [r for r in g if r["kind"] == "ctrl" and r["id"].endswith("-30")]   # 只用事件前 30 天的對照（事後對照的窗口會含到事件降雨）
            try:
                ev_v = float(e[0][var])
                cv = [float(r[var]) for r in c if r[var] not in ("", None)]
            except (ValueError, IndexError, TypeError):
                continue
            if cv:
                diffs.append(ev_v - float(np.mean(cv)))
        d = np.array(diffs)
        if len(d):
            res[var] = {"n_pairs": int(len(d)), "mean_diff": round(float(d.mean()), 4), "frac_event_wetter": round(float((d > 0).mean()), 3),
                        "sign_test_p": round(float(binomtest(int((d > 0).sum()), len(d), 0.5).pvalue), 4)}
    rp = OUT / "cases_rain.csv"
    if rp.exists():                                    # E5：真實降雨 case-crossover（含配對 AUC、與 SMAP 合併）
        rr = {r["id"]: r for r in csv.DictReader(open(rp, encoding="utf-8"))}
        r5 = {}
        pairs = []
        for g in by.values():
            e = [r for r in g if r["kind"] == "event"]
            c = [r for r in g if r["kind"] == "ctrl" and r["id"].endswith("-30")]
            if not e or not c or e[0]["id"] not in rr or c[0]["id"] not in rr:
                continue
            ve, vc = rr[e[0]["id"]], rr[c[0]["id"]]
            try:
                pairs.append(({k: float(ve[k]) for k in ("r3d", "r7d", "r30d", "rmax1d", "ante30d")},
                              {k: float(vc[k]) for k in ("r3d", "r7d", "r30d", "rmax1d", "ante30d")},
                              float(e[0]["sm_rz_ante3d"]) - float(c[0]["sm_rz_ante3d"])))
            except (ValueError, TypeError, KeyError):
                continue
        for k in ("r3d", "r7d", "r30d", "rmax1d", "ante30d"):
            d = np.array([pe[k] - pc[k] for pe, pc, _ in pairs])
            r5[k] = {"n_pairs": int(len(d)), "median_event_mm": round(float(np.median([pe[k] for pe, _, _ in pairs])), 1),
                     "median_ctrl_mm": round(float(np.median([pc[k] for _, pc, _ in pairs])), 1),
                     "frac_event_wetter": round(float((d > 0).mean()), 3), "paired_auc": round(float(((d > 0) + 0.5 * (d == 0)).mean()), 3),
                     "sign_test_p": round(float(binomtest(int((d > 0).sum()), len(d), 0.5).pvalue), 5)}
        out["E5_rain_case_crossover"] = r5
        print("E5", json.dumps(r5, ensure_ascii=False))
        # 雨量門檻：事件的 r3d 分佈 vs 對照（命中率／誤報率）
        ev3 = np.array([pe["r3d"] for pe, _, _ in pairs]); ct3 = np.array([pc["r3d"] for _, pc, _ in pairs])
        thr = {}
        for t in (20, 50, 100, 150, 200):
            thr[str(t)] = {"hit_rate": round(float((ev3 >= t).mean()), 3), "false_alarm_rate_on_ctrl": round(float((ct3 >= t).mean()), 3)}
        out["E5_r3d_threshold"] = thr
        print("E5 threshold(r3d mm)", json.dumps(thr))
        # 條件 logistic（配對差分，無截距）：雨量差、前期含水量差
        from sklearn.linear_model import LogisticRegression
        Xd = np.array([[np.log1p(pe["r3d"]) - np.log1p(pc["r3d"]), np.log1p(pe["ante30d"]) - np.log1p(pc["ante30d"]), dsm] for pe, pc, dsm in pairs])
        Xs = np.vstack([Xd, -Xd]); ys = np.r_[np.ones(len(Xd)), np.zeros(len(Xd))]
        m = LogisticRegression(fit_intercept=False, max_iter=1000).fit(Xs, ys)
        out["E5_conditional_logit"] = {"features": ["log1p r3d 差", "log1p ante30d 差", "SMAP 根系層前期差"], "coef": m.coef_[0].round(3).tolist(),
                                       "paired_auc": round(float((Xd @ m.coef_[0] > 0).mean()), 3)}
        print("E5 logit", out["E5_conditional_logit"])
    out["E3_case_crossover"] = res
    print("E3", json.dumps(res, ensure_ascii=False))
    (OUT / "experiments.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", OUT / "experiments.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "gee", "rain", "analyze"])
    ap.add_argument("--hotspots", default=str(REPO / "data/ardswc_hotspots/top100_consolidated.json"))
    ap.add_argument("--project", default=None)
    a = ap.parse_args()
    {"prep": prep, "gee": gee, "rain": rain, "analyze": analyze}[a.cmd](a)
