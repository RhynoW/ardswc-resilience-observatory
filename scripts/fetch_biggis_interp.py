# -*- coding: utf-8 -*-
"""
抓取 BigGIS「災害事件衛星影像判識成果清單」API，快取並整理成可用的官方崩塌判釋多邊形。

來源：GET https://gis.ardswc.gov.tw/api/ardswc/eventfiles（公開、免登入；JSON 陣列，欄位 Id/Year/Date/EventType/EventName/FileType/DisName/UrlType/AccessUrl）
  FileType == "崩塌判釋" 的項目是 KML／KMZ（農業部農村發展及水土保持署 BigGIS，災後衛星影像判釋的新增崩塌範圍），檔案放在 geodac.tw。
輸出
  data/biggis_interp/eventfiles_raw.json    API 原始回應（含災前／災後影像圖磚網址等，供追溯；--refresh 才重抓）
  data/biggis_interp/raw/<Id>.kml           原始 KML 快取（不進版控）
  data/biggis_interp/polygons.json          整理後多邊形（簡化 ≈3 m、5 位小數）：events[] ＋ polys[[事件索引, 面積公頃, [lon,lat,…]]]，供驗證腳本使用
  webapp/change_detect_viewer/static/biggis_interp.json   網頁疊圖用（簡化 ≈8 m、4 位小數，較小）
注意（寫進文件與介面）
  * 這是「官方衛星影像判釋」，不是現地確認，也不是與水保署體系完全獨立的資料；本站稱它為「官方判釋」，不稱真值。
  * 同一事件可能有多個判釋檔（例如凱米颱風的函文版與 Sentinel-2／Pleiades 補充判釋），範圍可能重疊；統計時以「事件」或空間聯集處理。
  * 只取外環（忽略內洞），面積略為高估；座標位置精度未獨立驗證。
  * 資料使用條款尚未確認（轉存、展示前請先確認並標註來源）；API 無版本與穩定性保證，故離線快取，網頁不即時呼叫。
用法：python scripts/fetch_biggis_interp.py [--refresh]
"""
import argparse
import datetime as dt
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

REPO = Path(__file__).resolve().parent.parent
API = "https://gis.ardswc.gov.tw/api/ardswc/eventfiles"
OUT = REPO / "data" / "biggis_interp"
RAW = OUT / "raw"
STATIC = REPO / "webapp" / "change_detect_viewer" / "static" / "biggis_interp.json"
UA = {"User-Agent": "Mozilla/5.0 ardswc-resilience-observatory (research; cached)"}


def http_get(u, tries=3):
    p = urllib.parse.urlsplit(u)
    u2 = urllib.parse.urlunsplit((p.scheme, p.netloc, urllib.parse.quote(p.path), p.query, ""))
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u2, headers=UA), timeout=120).read()
        except Exception:  # noqa: BLE001
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def coords(txt):
    out = []
    for tok in txt.split():
        a = tok.split(",")
        if len(a) >= 2:
            try:
                out.append((float(a[0]), float(a[1])))
            except ValueError:
                pass
    return out


PM = re.compile(r"<Placemark\b.*?</Placemark>", re.S)
NAME = re.compile(r"<name>(.*?)</name>", re.S)
POLY = re.compile(r"<Polygon\b.*?</Polygon>", re.S)
OUTER = re.compile(r"<outerBoundaryIs>.*?<coordinates>(.*?)</coordinates>", re.S)
HAREA = re.compile(r"\(([\d.]+)\s*公頃\)")


def parse_kml(txt):
    """以正規表示式抽出 Placemark 的多邊形外環（不經 XML 解析：這些 KML 的命名空間常不完整）。"""
    polys = []
    for pm in PM.findall(txt):
        m = NAME.search(pm)
        name = re.sub(r"<!\[CDATA\[|\]\]>", "", m.group(1)).strip() if m else ""
        for pg in POLY.findall(pm):
            o = OUTER.search(pg)
            if o:
                ring = coords(o.group(1))
                if len(ring) >= 4:
                    polys.append((name, ring))
    return polys


def area_ha(ring):
    lat0 = float(np.mean([p[1] for p in ring])); k = math.cos(math.radians(lat0))
    xy = np.array([(p[0] * 111320 * k, p[1] * 110570) for p in ring])
    x, y = xy[:, 0], xy[:, 1]
    return abs(0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))) / 1e4


def simplify(ring, tol_m, nd):
    k = math.cos(math.radians(float(np.mean([p[1] for p in ring]))))
    pg = Polygon(ring)
    if not pg.is_valid:
        pg = pg.buffer(0)
        if pg.is_empty:
            return None
        if pg.geom_type != "Polygon":
            pg = max(pg.geoms, key=lambda g: g.area)
    s = pg.simplify(tol_m / 111320 / max(k, 0.5), preserve_topology=True)
    if s.is_empty or s.geom_type != "Polygon":
        s = pg
    c = list(s.exterior.coords)
    flat = []
    for lo, la in c:
        flat += [round(lo, nd), round(la, nd)]
    return flat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="重抓 API 清單與所有 KML（預設用快取）")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); RAW.mkdir(exist_ok=True)
    lp = OUT / "eventfiles_raw.json"
    if lp.exists() and not a.refresh:
        listing = json.loads(lp.read_text(encoding="utf-8"))
        fetched = dt.datetime.fromtimestamp(lp.stat().st_mtime).date().isoformat()
    else:
        listing = json.loads(http_get(API).decode("utf-8"))
        lp.write_text(json.dumps(listing, ensure_ascii=False), encoding="utf-8")
        fetched = dt.date.today().isoformat()
    L = sorted([x for x in listing if x["FileType"] == "崩塌判釋"], key=lambda x: (x["Date"], x["Id"]))
    print(f"API {len(listing)} 筆；崩塌判釋 {len(L)} 個檔（{len({(x['Date'], x['EventName']) for x in L})} 個事件）")
    events, polys3, polysW, fails = [], [], [], []
    for x in L:
        rp = RAW / f"{x['Id']}.kml"
        try:
            if rp.exists() and not a.refresh:
                txt = rp.read_text(encoding="utf-8", errors="replace")
            else:
                b = http_get(x["AccessUrl"])
                if x["AccessUrl"].lower().endswith(".kmz"):
                    z = zipfile.ZipFile(io.BytesIO(b))
                    b = z.read([n for n in z.namelist() if n.lower().endswith(".kml")][0])
                txt = b.decode("utf-8", "replace"); rp.write_text(txt, encoding="utf-8")
            P = parse_kml(txt)
        except Exception as e:  # noqa: BLE001
            fails.append((x["Id"], x["Date"], x["EventName"], f"{type(e).__name__}: {str(e)[:60]}")); continue
        ei = len(events); tot = 0.0; n = 0
        for name, ring in P:
            ar = area_ha(ring)
            s3, sw = simplify(ring, 3, 5), simplify(ring, 8, 4)
            if not s3 or not sw:
                continue
            polys3.append([ei, round(ar, 3), s3]); polysW.append([ei, round(ar, 2), sw]); tot += ar; n += 1
        events.append({"i": ei, "id": x["Id"], "date": x["Date"], "year": x["Year"], "event": x["EventName"], "type": x["EventType"],
                       "dis": x["DisName"], "url": x["AccessUrl"], "n": n, "area_ha": round(tot, 1)})
    meta = {"source": API, "fetched": fetched, "n_files": len(events), "n_polygons": len(polys3), "n_failed": len(fails),
            "note": "BigGIS 災害事件衛星影像判釋成果（崩塌判釋 KML/KMZ）；官方衛星判釋，非現地確認；同事件多檔可能重疊；僅外環；使用條款待確認。",
            "attribution": "農業部農村發展及水土保持署 BigGIS 巨量空間資訊系統（gis.ardswc.gov.tw）"}
    (OUT / "polygons.json").write_text(json.dumps({"meta": meta, "events": events, "polys": polys3}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    STATIC.write_text(json.dumps({"meta": meta, "events": events, "polys": polysW}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"事件檔 {len(events)}、多邊形 {len(polys3)}、總面積 {sum(e['area_ha'] for e in events):.0f} 公頃；失敗 {len(fails)}")
    for f in fails:
        print("  失敗", f)
    print("polygons.json", (OUT / "polygons.json").stat().st_size // 1024, "KB；static", STATIC.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
