# -*- coding: utf-8 -*-
"""
歷史影像平台「大數據視覺化分析」頁的資料：把 10 萬筆影像紀錄彙整成小型 JSON（data/analytics.json）。

資料來源：build_hotspots.py 抓下的原始快取（--raw-dir；四類 PhotoType 各一個 type_<n>.json，欄位為平台 API 原樣：
EventID、PhotoType、Lat、Lng、County、Town、PhotoDate、DisasterYear、DisasterName、Description、Note、Source…）。
本腳本不連網；要更新資料先跑 `python scripts/build_hotspots.py --raw-dir <快取> --refresh`。

輸出內容（皆為彙整值，原始逐筆影像不放進網站）：
  totals            總筆數、有座標筆數、各類別筆數
  by_year           災害年份 × 類別筆數
  by_month          拍攝月份 × 類別筆數（災害事件＋媒體報導；看季節性）
  by_county         縣市 × 類別筆數
  uav_share         每年「空拍」筆數與占比（備註／描述含「空拍」）
  keywords          描述／備註中的關鍵字次數（災害事件＋媒體報導）
  top_events        筆數最多的災害名稱（含年份範圍、座標範圍、各縣市筆數前三）
  event_points      上述事件的座標（4 位小數，每事件最多 --max-points 筆）供地圖疊點
  grid              0.05°（約 5 km）網格計數（災害事件＋媒體報導，有座標），供密度圖
限制：筆數代表「拍了多少張」，不是災害次數（同一場災害可拍很多張；道路可達處拍得多），
  解讀時需與「獨立事件」（build_hotspots.py）區分；空拍占比上升會讓近年筆數膨脹。
用法：python scripts/build_analytics.py --raw-dir <快取目錄>
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "analytics.json"
TYPES = {"0": "災害事件", "6": "重要地景", "8": "媒體報導", "10": "出版品照片"}
KEYWORDS = ["崩塌", "土石流", "堰塞湖", "道路中斷", "坍方", "淹水", "護岸", "橋", "潛勢溪流", "不安定土砂", "地震", "颱風", "豪雨", "空拍"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--top-events", type=int, default=40)
    ap.add_argument("--max-points", type=int, default=1500)
    ap.add_argument("--date", default=None, help="資料抓取日（預設今天）")
    a = ap.parse_args()
    raw = Path(a.raw_dir)
    rows = []
    for t, name in TYPES.items():
        for r in json.loads((raw / f"type_{t}.json").read_text(encoding="utf-8")):
            r["_t"] = name
            rows.append(r)
    geo = [r for r in rows if r.get("Lat") and r.get("Lng")]
    haz = [r for r in geo if r["_t"] in ("災害事件", "媒體報導")]

    by_year = defaultdict(Counter)
    for r in rows:
        if r.get("DisasterYear"):
            by_year[int(r["DisasterYear"])][r["_t"]] += 1
    by_month = defaultdict(Counter)
    for r in haz:
        d = str(r.get("PhotoDate") or "")
        if len(d) >= 7 and d[:4].isdigit():
            by_month[int(d[5:7])][r["_t"]] += 1
    by_county = defaultdict(Counter)
    for r in geo:
        by_county[r.get("County") or "未標示"][r["_t"]] += 1
    uav = defaultdict(lambda: [0, 0])
    for r in rows:
        y = str(r.get("PhotoDate") or "")[:4]
        if y.isdigit():
            uav[int(y)][1] += 1
            if "空拍" in (r.get("Note") or "") + (r.get("Description") or ""):
                uav[int(y)][0] += 1
    kw = {k: Counter() for k in KEYWORDS}
    for r in haz:
        txt = (r.get("Description") or "") + " " + (r.get("Note") or "") + " " + (r.get("DisasterName") or "")
        for k in KEYWORDS:
            if k in txt:
                kw[k][r["_t"]] += 1

    names = Counter((r.get("DisasterName") or "").strip() for r in haz if (r.get("DisasterName") or "").strip())
    top = [n for n, _ in names.most_common(a.top_events)]
    events, pts = [], {}
    for n in top:
        rs = [r for r in haz if (r.get("DisasterName") or "").strip() == n]
        years = sorted({int(r["DisasterYear"]) for r in rs if r.get("DisasterYear")})
        cty = Counter(r.get("County") or "" for r in rs).most_common(3)
        events.append({"name": n, "n": len(rs), "years": [years[0], years[-1]] if years else None,
                       "counties": [[c, k] for c, k in cty],
                       "bbox": [min(r["Lng"] for r in rs), min(r["Lat"] for r in rs), max(r["Lng"] for r in rs), max(r["Lat"] for r in rs)]})
        step = max(1, len(rs) // a.max_points)
        pts[n] = [[round(r["Lat"], 4), round(r["Lng"], 4)] for r in rs[::step]]
    grid = Counter((round(r["Lat"] / 0.05) * 0.05, round(r["Lng"] / 0.05) * 0.05) for r in haz if 21.5 < r["Lat"] < 26.5 and 118 < r["Lng"] < 123)

    out = {
        "generated_from": str(raw.name), "generated": a.date or __import__("datetime").date.today().isoformat(),
        "totals": {"all": len(rows), "geolocated": len(geo), "by_type": dict(Counter(r["_t"] for r in rows)),
                   "geolocated_by_type": dict(Counter(r["_t"] for r in geo)), "hazard_geolocated": len(haz)},
        "by_year": {str(y): dict(c) for y, c in sorted(by_year.items())},
        "by_month": {str(m): dict(c) for m, c in sorted(by_month.items())},
        "by_county": {k: dict(c) for k, c in sorted(by_county.items(), key=lambda kv: -sum(kv[1].values()))},
        "uav_share": {str(y): {"uav": u, "all": n} for y, (u, n) in sorted(uav.items())},
        "keywords": {k: dict(c) for k, c in kw.items()},
        "top_events": events, "event_points": pts,
        "grid": [[round(la, 2), round(lo, 2), n] for (la, lo), n in grid.items()],
        "note": "筆數＝照片／紀錄張數，不是災害次數；同一場災害可拍很多張，道路可達處拍得多。",
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(rows)} 筆、有座標 {len(geo)}、災害事件＋媒體報導 {len(haz)}；top 事件 {len(events)}；grid {len(out['grid'])} 格 → {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
