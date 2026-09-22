# -*- coding: utf-8 -*-
"""
從水保署歷史影像平台即時資料重建「Top 100 長期複發熱點」（2026-09-22 起的版本）。

取代 2026-09-05 的舊快照做法（舊版用 GetEventPositionList，該 API 沒有災害年份欄位，只能用
建檔年份；原始聚合腳本也未保存）。本腳本把整個流程寫死成可重現的規則：

1. 資料來源：https://photo.ardswc.gov.tw/api/v1/rest/dataset/metadata/<PhotoType>?page=N
   （每頁 1000 筆，四類 PhotoType 全抓）。原始回應快取到 --raw-dir，同一份快取可重跑出同樣結果。
2. events_trimmed.json：四類中「有座標」的紀錄全部保留（地圖疊點與統計用），
   欄位 id/lat/lon/photo_type/photo_type_label/year/county/town；year = DisasterYear（災害年份）。
3. 熱點只用 PhotoType=0「災害事件」與 PhotoType=8「媒體報導」（報導的災情地點）：
   重要地景、出版品照片不代表災害發生，不計入複發（2026-09-22 輔導委員建議）。
4. 空間聚合：以台灣中心緯度（23.7°）換算的等距 250 m 網格（與影像框、地形起伏因子同尺度）。
5. 獨立災害事件去重（同一場災害的後續追蹤、巡查、重複拍攝不應被算成多次災害）：
   - 名稱正規化：去掉「暨…」後綴、民國年前綴（「99萊羅克台東大武001」→「萊羅克台東大武001」）與
     尾端編號；「921集集大地震／九二一地震」統一為「921地震」。
   - 泛稱（空白、「其他」「其他事件」「豪雨」「經典案例」「YYYY年其他災害」、以「其他」開頭者）另行處理。
   - 具名事件：同網格內「核心名稱」相同或互為子字串（核心名稱＝去掉颱風/豪雨/地震/熱帶低壓/水災等字尾，
     例如「萊羅克」）、且災害年份相差 ≤1 年 → 同一事件（跨年的後續追蹤照，如莫蘭蒂 2016/2017）。
   - 泛稱紀錄：拍攝日期落在某具名事件 ±30 天內 → 併入該事件；其餘依拍攝日期 ≤30 天串成同一事件。
   複發性 = 獨立事件數（n_independent_events）；另保留不同災害年份數與原始紀錄筆數供對照。
6. 排序：獨立事件數 ↓ → 不同災害年份數 ↓ → 最近災害年份 ↓（原始紀錄筆數不參與排序，
   避免「拍得多」被當成「災害多」）。
7. 去重疊：依序挑選，與已選熱點中心距離 < 400 m 的網格略過（相鄰網格常是同一處邊坡被網格線切開），
   直到湊滿 100 個。熱點中心 = 該格紀錄座標平均；行政區 = 該格紀錄 County/Town 眾數。

用法：
  python scripts/build_hotspots.py --raw-dir <快取目錄>            # 有快取就用快取
  python scripts/build_hotspots.py --raw-dir <快取目錄> --refresh  # 強制重抓
"""
import argparse
import json
import math
import re
import time
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data" / "ardswc_hotspots"
API = "https://photo.ardswc.gov.tw/api/v1/rest/dataset/metadata"
PHOTO_TYPES = {"0": "災害事件", "6": "重要地景", "8": "媒體報導", "10": "出版品照片"}
HOTSPOT_TYPES = {"0", "8"}
SAME_EVENT_DAYS = 30
CELL_M = 250.0
MIN_SEP_M = 400.0
N_HOTSPOTS = 100
LAT0 = 23.7
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))


def fetch_type(pt, raw_dir, refresh):
    fp = raw_dir / f"type_{pt}.json"
    if fp.exists() and not refresh:
        return json.loads(fp.read_text(encoding="utf-8"))
    rows, page = [], 1
    while True:
        req = urllib.request.Request(f"{API}/{pt}?page={page}", headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(4):
            try:
                d = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
                if d is None:
                    raise ValueError("null page")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  retry type {pt} page {page}: {e}", flush=True)
                time.sleep(4)
        else:
            raise SystemExit(f"無法取得 type {pt} page {page}")
        rows += d
        if len(d) < 1000:
            break
        page += 1
        time.sleep(0.4)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def has_coord(r):
    try:
        lat, lon = float(r.get("Lat")), float(r.get("Lng"))
    except (TypeError, ValueError):
        return False
    return 21.5 < lat < 26.5 and 118.0 < lon < 122.5   # 台澎金馬範圍外視為座標錯誤


def dist_m(a, b):
    return math.hypot((a[0] - b[0]) * M_PER_DEG_LAT, (a[1] - b[1]) * M_PER_DEG_LON)


_GENERIC = {"", "其他", "其他事件", "其他災害", "豪雨", "經典案例", "地震", "颱風", "（未命名）"}
_SUFFIX = re.compile(r"(大地震|地震|颱風|豪雨|熱帶低壓|水災|西南氣流|事件)+$")


def _norm(name):
    n = (name or "").strip().split("暨")[0].strip()
    if re.search(r"921|九二一", n):
        return "921地震"
    m = re.match(r"^(\d{2,3})(?=\D)", n)            # 民國年前綴（2–3 位數字接非數字）；MMDD 為 4 位不受影響
    if m:
        n = n[m.end():]
    return re.sub(r"\d{3}$", "", n).strip()           # 尾端流水號


def _generic(name):
    n = _norm(name)
    return n in _GENERIC or n.startswith("其他") or re.fullmatch(r"\d{4}年其他災害", (name or "").strip()) is not None


def _core(name):
    n = _norm(name)
    return _SUFFIX.sub("", n) or n


def _day(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").toordinal()
    except ValueError:
        return None


def independent_events(evs):
    """同一網格內的紀錄 → 獨立災害事件清單（規則見模組說明第 5 點）。"""
    named, generic = defaultdict(list), []
    for e in evs:
        (generic if _generic(e["name"]) else named[(int(e["year"]), _norm(e["name"]))]).append(e)
    # 具名事件：核心名稱相同或互為子字串、且年份差 ≤1 → 合併（union-find）
    keys = sorted(named)
    parent = {k: k for k in keys}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            (ya, na), (yb, nb) = keys[a], keys[b]
            ca, cb = _core(na), _core(nb)
            if abs(ya - yb) <= 1 and len(ca) >= 2 and len(cb) >= 2 and (ca in cb or cb in ca):
                parent[find(keys[b])] = find(keys[a])
    groups = defaultdict(list)
    for k in keys:
        groups[find(k)].append(k)
    events = []
    for root, ks in groups.items():
        recs = [r for k in ks for r in named[k]]
        names = Counter(r["name"] for r in recs)
        events.append({"year": min(k[0] for k in ks), "name": names.most_common(1)[0][0],
                       "aliases": sorted(set(names) - {names.most_common(1)[0][0]}), "records": recs})
    # 泛稱紀錄：先嘗試併入 ±30 天內的具名事件，其餘依日期串接
    rest = []
    for e in generic:
        d = _day(e["date"])
        host = None
        if d is not None:
            for ev in events:
                if any(abs(d - (_day(r["date"]) or -10**6)) <= SAME_EVENT_DAYS for r in ev["records"]):
                    host = ev
                    break
        (host["records"].append(e) if host else rest.append(e))
    rest.sort(key=lambda e: (e["date"] or "", e["year"]))
    chain = None
    for e in rest:
        d = _day(e["date"])
        if chain and d is not None and chain["_last_day"] is not None and d - chain["_last_day"] <= SAME_EVENT_DAYS:
            chain["records"].append(e)
            chain["_last_day"] = d
        else:
            chain = {"year": int(e["year"]), "name": "（未命名／泛稱）", "aliases": [], "records": [e], "_last_day": d}
            events.append(chain)
    out = []
    for ev in events:
        dates = sorted(r["date"] for r in ev["records"] if r["date"])
        aliases = sorted(set(ev["aliases"]) | ({r["name"] for r in ev["records"]} - {ev["name"]}))
        out.append({"year": ev["year"], "name": ev["name"], "aliases": [a for a in aliases if a][:6],
                    "n_records": len(ev["records"]),
                    "first_date": dates[0] if dates else None, "last_date": dates[-1] if dates else None,
                    "types": sorted({r["photo_type_label"] for r in ev["records"]})})
    return sorted(out, key=lambda x: (x["year"], x["first_date"] or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    raw = {pt: fetch_type(pt, args.raw_dir, args.refresh) for pt in PHOTO_TYPES}
    fetched = datetime.fromtimestamp(min((args.raw_dir / f"type_{pt}.json").stat().st_mtime for pt in PHOTO_TYPES),
                                     tz=timezone.utc).astimezone().strftime("%Y-%m-%d")

    # ── events_trimmed.json（四類、有座標、依 EventID 去重）
    events, seen = [], set()
    for pt, rows in raw.items():
        for r in rows:
            eid = str(r.get("EventID") or "")
            if not eid or eid in seen or not has_coord(r):
                continue
            seen.add(eid)
            events.append({"id": eid, "lat": round(float(r["Lat"]), 6), "lon": round(float(r["Lng"]), 6),
                           "photo_type": pt, "photo_type_label": PHOTO_TYPES[pt],
                           "year": str(r["DisasterYear"]) if r.get("DisasterYear") else None,
                           "county": r.get("County") or None, "town": r.get("Town") or None,
                           "name": (r.get("DisasterName") or "").strip(), "date": (r.get("PhotoDate") or "")[:10]})
    total_all = sum(len({str(r.get("EventID")) for r in rows}) for rows in raw.values())

    # ── 250 m 網格聚合（災害事件＋媒體報導、且要有災害年份）
    cells = defaultdict(list)
    for e in events:
        if e["photo_type"] not in HOTSPOT_TYPES or not e["year"]:
            continue
        key = (math.floor(e["lat"] * M_PER_DEG_LAT / CELL_M), math.floor(e["lon"] * M_PER_DEG_LON / CELL_M))
        cells[key].append(e)
    stats = []
    for key, evs in cells.items():
        indep = independent_events(evs)
        years = sorted({ev["year"] for ev in indep})
        stats.append({"evs": evs, "indep": indep, "n_indep": len(indep), "years": years,
                      "n_years": len(years), "n": len(evs), "last": years[-1]})
    stats.sort(key=lambda s: (-s["n_indep"], -s["n_years"], -s["last"]))

    chosen = []
    for s in stats:
        c = (sum(e["lat"] for e in s["evs"]) / s["n"], sum(e["lon"] for e in s["evs"]) / s["n"])
        if any(dist_m(c, (h["lat"], h["lon"])) < MIN_SEP_M for h in chosen):
            continue
        mode = lambda k: (Counter(e[k] for e in s["evs"] if e[k]).most_common(1) or [(None, 0)])[0][0]
        chosen.append({
            "rank": len(chosen) + 1, "county": mode("county") or "", "district": mode("town") or "",
            "lat": round(c[0], 5), "lon": round(c[1], 5),
            "n_independent_events": s["n_indep"], "n_distinct_years": s["n_years"], "years": s["years"],
            "n_events": s["n"],
            "n_by_type": dict(Counter(PHOTO_TYPES[e["photo_type"]] for e in s["evs"])),
            "events": s["indep"],
            "method": "sentinel2", "change_score": None, "deep_verify_caveat": None,
        })
        if len(chosen) == N_HOTSPOTS:
            break

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "events_trimmed.json").write_text(json.dumps(events, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (OUT / "top100_consolidated.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "source": API, "fetched": fetched, "platform_total_records": total_all,
        "geolocated_records": len(events),
        "geolocated_by_type": dict(Counter(e["photo_type_label"] for e in events)),
        "hotspot_basis_types": [PHOTO_TYPES[t] for t in sorted(HOTSPOT_TYPES)],
        "hotspot_basis_records": sum(len(c) for c in cells.values()),
        "same_event_rule": f"同網格：具名災害依『災害年份＋名稱』合併；泛稱/空白依拍攝日期 ≤{SAME_EVENT_DAYS} 天串接",
        "grid_cells": len(cells), "cell_m": CELL_M, "min_separation_m": MIN_SEP_M,
        "ranking": "n_independent_events desc, n_distinct_years desc, latest year desc",
        "independent_events_distribution_top100": dict(sorted(Counter(h["n_independent_events"] for h in chosen).items())),
        "records_before_dedup_top100": sum(h["n_events"] for h in chosen),
        "independent_events_top100": sum(h["n_independent_events"] for h in chosen),
    }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
