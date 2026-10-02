# -*- coding: utf-8 -*-
"""
全時段逐日分析：雨量門檻 → 事件機率與誤報率（每個熱點的每一天）。

為什麼：case-crossover（warning_experiments.py E5）只比「事件日 vs 任選對照日」，無法給出作業用的誤報率。
這裡改成：對每個熱點、每一天（2016-01-01～2024-12-31）算 IMERG 累積雨量，事件＝該熱點的通報事件首日，
問「超過門檻的日子裡，有多少真的出事？」——命中率、每熱點每年的警報次數、警報精確率、事件基準率。

  fetch    GEE `NASA/GPM_L3/IMERG_V07`：每熱點每日雨量（mm，Σ(mm/hr)×0.5），按月抓、存 data/warning_trainer/daily_rain.csv（可續跑）
  analyze  以 r3d／r7d 門檻掃描；警報「集合」＝相鄰超標日（間隔 ≤ gap-days）合併成一次警報，事件落在警報起日 −1～+lead 天內算命中
           並輸出 逐日 AUC、不同門檻的 命中率／警報次數／警報精確率、分箱的事件率（每千熱點日）

標籤與偏差（務必一併報告）：
  * 事件＝通報（ARDSWC 通報首日），不是衛星判釋崩塌；通報有曝險偏差、日期可能晚於崩塌。
  * 這 200 個熱點「因為出過事」才被選入——基準事件率被高估，precision 是樂觀上限，不能外推到全臺任意山坡。
  * 事件日前後 ±--excl-days 天的非事件日從負例排除（同一場颱風的延續日不當誤報）。
  * IMERG 0.1°，山區低估；每熱點多事件不獨立（同颱風跨多熱點），信賴區間用「以颱風日聚類」的 bootstrap 近似。

用法：python scripts/rain_threshold.py fetch --hotspots <list.json> [--project <id>]
      python scripts/rain_threshold.py analyze --hotspots <list.json>
"""
import argparse
import csv
import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "warning_trainer"
CSV = OUT / "daily_rain.csv"
START, END = dt.date(2016, 1, 1), dt.date(2024, 12, 31)


def months():
    d = START
    while d <= END:
        n = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield d, min(n - dt.timedelta(days=1), END)
        d = n


def fetch(a):
    import ee
    ee.Initialize(project=a.project or os.environ.get("GEE_PROJECT"))
    hs = json.loads(Path(a.hotspots).read_text(encoding="utf-8"))
    col = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([h["lon"], h["lat"]]), {"rank": h["rank"]}) for h in hs])
    done = set()
    if CSV.exists():
        done = {r["month"] for r in csv.DictReader(open(CSV, encoding="utf-8"))}
    OUT.mkdir(parents=True, exist_ok=True)
    new = not CSV.exists()

    def one(m):
        d0, d1 = m
        key = d0.strftime("%Y-%m")
        n = (d1 - d0).days + 1
        days = [d0 + dt.timedelta(days=i) for i in range(n)]
        imgs = [col.filterDate(str(d), str(d + dt.timedelta(days=1))).sum().multiply(0.5).rename(f"d{d.strftime('%Y%m%d')}") for d in days]
        im = ee.Image.cat(imgs)
        feats = im.reduceRegions(fc, ee.Reducer.first(), 11000).getInfo()["features"]
        rows = []
        for ft in feats:
            p = ft["properties"]
            for d in days:
                v = p.get(f"d{d.strftime('%Y%m%d')}")
                rows.append((key, p["rank"], d.isoformat(), "" if v is None else round(v, 2)))
        return key, rows

    todo = [m for m in months() if m[0].strftime("%Y-%m") not in done]
    print(f"{len(todo)} 個月待抓（已完成 {len(done)}）", flush=True)
    with open(CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["month", "rank", "date", "mm"])
        with ThreadPoolExecutor(4) as ex:
            for key, rows in ex.map(one, todo):
                w.writerows(rows)
                fh.flush()
                print("  ", key, len(rows), flush=True)
    print("→", CSV)


def analyze(a):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    hs = json.loads(Path(a.hotspots).read_text(encoding="utf-8"))
    ranks = [h["rank"] for h in hs]
    days = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    di = {d: i for i, d in enumerate(days)}
    R = {r: np.full(len(days), np.nan) for r in ranks}
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["mm"] != "" and int(r["rank"]) in R:
            R[int(r["rank"])][di[dt.date.fromisoformat(r["date"])]] = float(r["mm"])
    ev = {r: [] for r in ranks}
    for h in hs:
        for e in h["events"]:
            d = dt.date.fromisoformat(e["first_date"])
            if START <= d <= END:
                ev[h["rank"]].append(di[d])
    n_ev = sum(len(v) for v in ev.values())
    ok_ranks = [r for r in ranks if np.isfinite(R[r]).mean() > 0.95]
    print(f"熱點 {len(ok_ranks)}/{len(ranks)} 有完整雨量；事件 {n_ev}；天數 {len(days)}")

    def cum(x, k):                          # 含當日的 k 日累積；缺值當 0
        c = np.cumsum(np.nan_to_num(x)); out = c.copy(); out[k:] = c[k:] - c[:-k]; return out
    feats, lab, excl = {}, {}, {}
    for r in ok_ranks:
        x = R[r]
        feats[r] = {"r1d": cum(x, 1), "r3d": cum(x, 3), "r7d": cum(x, 7)}
        y = np.zeros(len(days), bool); ex = np.zeros(len(days), bool)
        for i in ev[r]:
            y[i] = True; ex[max(0, i - a.excl_days):i + a.excl_days + 1] = True
        lab[r] = y; excl[r] = ex & ~y
    rows = []
    for key in ("r1d", "r3d", "r7d"):
        v = np.concatenate([feats[r][key] for r in ok_ranks]); y = np.concatenate([lab[r] for r in ok_ranks]); ex = np.concatenate([excl[r] for r in ok_ranks])
        keep = ~ex                           # 排除事件前後的延續日
        auc = roc_auc_score(y[keep], v[keep])
        rows.append((key, auc))
        print(f"逐日 AUC（事件日 vs 其他日；排除事件 ±{a.excl_days} 天）{key}: {auc:.3f}")
    out = {"n_hotspots": len(ok_ranks), "n_events": int(sum(len(ev[r]) for r in ok_ranks)), "n_days": len(days),
           "base_rate_per_1000_hotspot_days": None, "daily_auc": {k: round(v, 3) for k, v in rows}}
    tot = len(ok_ranks) * len(days)
    out["base_rate_per_1000_hotspot_days"] = round(1000 * out["n_events"] / tot, 3)

    # 分箱事件率（r3d）
    v3 = np.concatenate([feats[r]["r3d"] for r in ok_ranks]); y = np.concatenate([lab[r] for r in ok_ranks]); ex = np.concatenate([excl[r] for r in ok_ranks])
    bins = [0, 5, 20, 50, 100, 150, 250, 1e9]
    cal = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (v3 >= lo) & (v3 < hi) & ~ex
        cal.append({"r3d_mm": f"{lo}-{hi if hi < 1e8 else '∞'}", "hotspot_days": int(m.sum()), "events": int(y[m].sum()),
                    "events_per_1000_days": round(1000 * float(y[m].mean()), 2) if m.sum() else None})
    out["calibration_r3d"] = cal
    for c in cal:
        print("  ", c)

    # 警報集合：r3d 或 r7d 超標日，相鄰（間隔 ≤ gap）合併成一次警報；事件在警報起日 −1…+lead 天內算命中
    res = []
    years = len(days) / 365.25
    for key, ths in (("r3d", (50, 80, 100, 150, 200)), ("r7d", (100, 150, 200, 300))):
        for t in ths:
            n_alert = n_hit_alert = n_ev_hit = 0
            for r in ok_ranks:
                over = np.where(feats[r][key] >= t)[0]
                eps = []
                for i in over:
                    if eps and i - eps[-1][1] <= a.gap_days:
                        eps[-1][1] = i
                    else:
                        eps.append([i, i])
                evs = np.array(ev[r])
                hit_ev = set()
                for s, e in eps:
                    n_alert += 1
                    m = [j for j in evs if s - 1 <= j <= e + a.lead_days]
                    if m:
                        n_hit_alert += 1; hit_ev.update(m)
                n_ev_hit += len(hit_ev)
            res.append({"var": key, "threshold_mm": t, "alerts": n_alert, "alerts_per_hotspot_year": round(n_alert / (len(ok_ranks) * years), 2),
                        "alert_precision": round(n_hit_alert / n_alert, 3) if n_alert else None,
                        "event_recall": round(n_ev_hit / out["n_events"], 3)})
    out["alert_scan"] = res
    for r in res:
        print("  ", r)

    # 逐日 logistic（r3d, r7d 取 log1p），報校準：預測機率分位 vs 實際事件率
    X = np.log1p(np.c_[v3, np.concatenate([feats[r]["r7d"] for r in ok_ranks])])
    keep = ~ex
    m = LogisticRegression(max_iter=1000).fit(X[keep], y[keep])
    p = m.predict_proba(X[keep])[:, 1]
    order = np.argsort(-p); top = order[: max(1, int(0.01 * len(p)))]
    out["logit"] = {"coef_log1p_r3d_r7d": m.coef_[0].round(3).tolist(), "intercept": round(float(m.intercept_[0]), 3),
                    "auc_in_sample": round(float(roc_auc_score(y[keep], p)), 3),
                    "top1pct_days_event_rate_per_1000": round(1000 * float(y[keep][top].mean()), 2)}
    print("logit", out["logit"])

    # 以日期聚類 bootstrap（同一場颱風在多熱點同日爆發）：r3d≥100 的警報精確率區間（事件日層級）
    rng = np.random.default_rng(1)
    t = 100
    over = np.concatenate([feats[r]["r3d"] >= t for r in ok_ranks]); nd = len(days)
    dayidx = np.tile(np.arange(nd), len(ok_ranks))
    ratios = []
    for _ in range(300):
        pick = rng.integers(0, nd, nd)
        w = np.bincount(pick, minlength=nd)[dayidx]
        sel = over & ~ex
        denom = (w * sel).sum()
        if denom:
            ratios.append((w * (sel & y)).sum() / denom)
    out["precision_day_level_r3d_ge_100"] = {"point": round(float(((over & ~ex) & y).sum() / (over & ~ex).sum()), 4),
                                             "ci95_cluster_by_day": [round(float(np.percentile(ratios, 2.5)), 4), round(float(np.percentile(ratios, 97.5)), 4)]}
    print("日層級精確率 r3d≥100", out["precision_day_level_r3d_ge_100"])
    out["caveats"] = ["事件為通報（曝險偏差、日期可能晚於崩塌）", "熱點因出過事才被選入，基準率偏高、precision 為樂觀上限",
                      "IMERG 0.1° 山區低估", "同颱風多熱點不獨立（已做日期聚類 bootstrap）"]
    (OUT / "rain_threshold.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("→", OUT / "rain_threshold.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "analyze"])
    ap.add_argument("--hotspots", default=str(REPO / "data/ardswc_hotspots/top100_consolidated.json"))
    ap.add_argument("--project", default=None)
    ap.add_argument("--excl-days", type=int, default=3)
    ap.add_argument("--gap-days", type=int, default=2)
    ap.add_argument("--lead-days", type=int, default=2)
    a = ap.parse_args()
    {"fetch": fetch, "analyze": analyze}[a.cmd](a)
