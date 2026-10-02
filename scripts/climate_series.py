# -*- coding: utf-8 -*-
"""
人工覆核頁的降雨／土壤濕度背景資料：每個覆核熱點的逐日 IMERG 雨量與 SMAP L4 根系層含水量，存成 data/climate_series.json。

資料（GEE，專案 ID 用 GEE_PROJECT）：
  * 雨量：NASA/GPM_L3/IMERG_V07（0.1°，30 分鐘，mm/hr）→ 逐日累積 mm（Σ×0.5）。單點取最近像元。
  * 土壤濕度：NASA/SMAP/SPL4SMGP/007 sm_rootzone（根系層 0–100 cm，約 9 km，m³/m³）→ 逐日平均。
    GEE 上 L4 資料有約一年延遲，之後的日期留空（null），覆核頁會標「無資料」。
用途：覆核者判讀「前後期影像之間發生過什麼降雨」，區分豪雨後新崩塌、與乾濕季／河床變動；不是變遷訊號本身。
限制：0.1°／9 km 解析度，山區單點代表性有限；IMERG 為衛星估計，山區易低估。

輸出格式（陣列索引 = 自 start 起的第 i 天；雨量為 0.1 mm 整數，缺值 -1；SM 為 0.001 整數 ×，缺值 -1）：
  {"start":"2016-01-01","end":"...","rain_unit":"0.1mm","sm_unit":"0.001","ranks":{"7":{"rain":[...],"sm":[...]}}}
用法：python scripts/climate_series.py [--project <id>]   # 可續跑（--refresh 重抓）
"""
import argparse
import datetime as dt
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
OUT = REPO / "data" / "climate_series.json"
START = dt.date(2016, 1, 1)


def month_ranges(end):
    d = START
    while d <= end:
        n = dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield d, min(n - dt.timedelta(days=1), end)
        d = n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--end", default=None, help="預設昨天")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    import ee
    ee.Initialize(project=a.project or os.environ.get("GEE_PROJECT"))
    import review_queue as RQ
    a_items, base_items = RQ.build_items(REPO / "review")
    pts = {it["rank"]: (it["lon"], it["lat"]) for it in a_items + base_items}
    end = dt.date.fromisoformat(a.end) if a.end else dt.date.today() - dt.timedelta(days=1)
    n_days = (end - START).days + 1
    rain = ee.ImageCollection("NASA/GPM_L3/IMERG_V07").select("precipitation")
    sm = ee.ImageCollection("NASA/SMAP/SPL4SMGP/007").select("sm_rootzone")
    EMPTY_R = ee.ImageCollection([ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename("precipitation")])
    EMPTY_S = ee.ImageCollection([ee.Image.constant(0).updateMask(ee.Image.constant(0)).rename("sm_rootzone")])
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(list(p)), {"rank": r}) for r, p in pts.items()])
    print(f"{len(pts)} 個熱點，{START} ～ {end}（{n_days} 天）", flush=True)

    def month(m):
        d0, d1 = m
        days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
        bands = []
        for d in days:
            k = d.strftime("%Y%m%d")
            nxt = str(d + dt.timedelta(days=1))
            # 該日無影像時 sum()/mean() 會得到「無波段」影像而炸掉：併入一張全遮罩的空影像，保證有波段（值為 null）
            bands.append(rain.filterDate(str(d), nxt).merge(EMPTY_R).sum().multiply(0.5).rename(f"r{k}"))
            bands.append(sm.filterDate(str(d), nxt).merge(EMPTY_S).mean().rename(f"s{k}"))
        for attempt in range(3):
            try:
                feats = ee.Image.cat(bands).reduceRegions(fc, ee.Reducer.first(), 9000).getInfo()["features"]
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
        res = {}
        for ft in feats:
            p = ft["properties"]
            res[p["rank"]] = [(p.get(f"r{d.strftime('%Y%m%d')}"), p.get(f"s{d.strftime('%Y%m%d')}")) for d in days]
        return d0, res

    series = {r: {"rain": [-1] * n_days, "sm": [-1] * n_days} for r in pts}
    with ThreadPoolExecutor(4) as ex:
        for d0, res in ex.map(month, list(month_ranges(end))):
            i0 = (d0 - START).days
            for r, vals in res.items():
                for j, (rv, sv) in enumerate(vals):
                    if rv is not None:
                        series[r]["rain"][i0 + j] = int(round(rv * 10))
                    if sv is not None:
                        series[r]["sm"][i0 + j] = int(round(sv * 1000))
            print("  ", d0.strftime("%Y-%m"), flush=True)
    last_sm = max((i for s in series.values() for i, v in enumerate(s["sm"]) if v >= 0), default=-1)
    last_rain = max((i for s in series.values() for i, v in enumerate(s["rain"]) if v >= 0), default=-1)
    meta = {"start": str(START), "end": str(end), "rain_unit": "0.1mm", "sm_unit": "0.001 m3/m3", "missing": -1,
            "last_rain_date": str(START + dt.timedelta(days=last_rain)), "last_sm_date": str(START + dt.timedelta(days=last_sm)),
            "sources": {"rain": "NASA/GPM_L3/IMERG_V07", "sm": "NASA/SMAP/SPL4SMGP/007 sm_rootzone"},
            "ranks": {str(r): s for r, s in series.items()}}
    OUT.write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("雨量最後日", meta["last_rain_date"], "；SMAP 最後日", meta["last_sm_date"], "→", OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
