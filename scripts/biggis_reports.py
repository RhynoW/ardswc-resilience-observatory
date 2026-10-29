"""水保署 BigGIS「崩塌報告專區」新生崩塌判釋報告 → 高雄 POC 坡面的官方佐證（只作佐證，不改候選層級）。
來源：https://gis.ardswc.gov.tw/news/collapses（2026-10-28 上線；每份報告一個 PDF：/news/download/<id>）。
每份報告解析三段：
  標題與事件（列表頁的標題、發布日期）
  五、新生崩塌－縣市及鄉鎮區統計：各鄉鎮「完成判釋比率」（雲覆未判釋的部分；與本站「資料不足」同一概念）與面積
  十五、新生崩塌列表：每處的縣市、鄉鎮、村里、面積（ha）、TWD97 座標、土石流潛勢溪流、類別、流域、影像種類與日期
PDF 存 data/biggis_interp/poc/reports/（gitignore，可重抓）。點位以 TWD97（EPSG:3826）落在哪個 SUv2 原始坡面（su_original）判定。
輸出：data/biggis_interp/poc/su_reports.json（meta：報告清單、POC 鄉鎮完成率；su：su_id → 該坡面內的新生崩塌點）；
     webapp/.../static/poc/su/report_points.json（坡面頁地圖疊圖用點位）；data/biggis_interp/poc/reports_parsed.json（全部解析結果）
用法：python scripts/biggis_reports.py [--refetch]
"""
import json
import re
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import fitz
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
PDF_DIR = POC / "reports"
SU = POC / "su_export" / "slope_units_SUv2.gpkg"
WEB = REPO / "webapp" / "change_detect_viewer" / "static" / "poc" / "su"
BASE = "https://gis.ardswc.gov.tw"
POC_TOWNS = ("桃源區", "那瑪夏區", "六龜區", "甲仙區")
UA = {"User-Agent": "Mozilla/5.0 (TASTI_SatDashboard research; OSSInt 2026)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return r.read()


def listing():
    """列表頁（分頁到沒有新項目為止）→ [{id, title, date}]"""
    out, seen = [], set()
    for page in range(1, 20):
        s = get(f"{BASE}/news/collapses?page={page}").decode("utf-8", "replace")
        items = re.findall(r'<li class="list-group-item"[^>]*>\s*([^<]+?)\s*<div[^>]*>\s*<a href="/news/collapses/(\d+)".*?<div[^>]*>\s*(\d{4}-\d{2}-\d{2})', s, re.S)
        new = [(t, i, d) for t, i, d in items if i not in seen]
        if not new:
            break
        for t, i, d in new:
            seen.add(i)
            out.append({"id": int(i), "title": re.sub(r"\s+", "", t), "date": d})
    return out


def clean(c):
    return re.sub(r"\s+", "", c or "")


COUNTIES = ("臺北市", "新北市", "桃園市", "臺中市", "臺南市", "高雄市", "基隆市", "新竹市", "嘉義市", "新竹縣", "苗栗縣", "彰化縣", "南投縣",
            "雲林縣", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣", "臺東縣", "澎湖縣", "金門縣", "連江縣")
SENSOR_FIX = {"PlametScope": "PlanetScope"}


def admin(s):
    """縣市＋鄉鎮區＋村里（PDF 表格常把「高雄」「市」或「桃源」「區」拆到不同格、或漏掉「市」）→ 三段。
    只有兩字前綴時（如「嘉義」「新竹」可能是縣或市），依後面是鄉／鎮（縣）或區（市）判斷。"""
    s = s.replace("台", "臺")
    county = next((c for c in COUNTIES if s.startswith(c)), None)
    rest = s[len(county):] if county else s[2:]
    if not county:
        cand = [c for c in COUNTIES if s.startswith(c[:2])]
        town_m = re.match(r".{1,3}?([鄉鎮區市])", rest)
        want = "縣" if town_m and town_m.group(1) in "鄉鎮" else "市"
        county = next((c for c in cand if c.endswith(want)), cand[0] if cand else "")
    m = re.match(r"(.{1,3}?[鄉鎮區市])(.*)", rest)
    return county, (m.group(1) if m else rest), (m.group(2) if m else "")


HEADER = re.compile(r"編號|^編$|^號$|縣市|鄉鎮|村里|TWD97|\(ha\)|潛勢溪流|類別|影像種類|註記|國家公|水庫集|都市計|流域|^面$|^積$")


def list_rows(d, start):
    """十五、新生崩塌列表的資料列。每頁最後一列常被分頁切開：前半在上一頁、後半（如「市」「區」「DF060」「20250805」）
    出現在下一頁表頭之後、編號欄空白——把這種殘列逐欄接回上一列（欄數不同時依非空欄順序接）。"""
    rows, n_merge = [], 0
    for pn in range(start, d.page_count):
        for tb in d[pn].find_tables().tables:
            for row in tb.extract():
                c = [clean(x) for x in row]
                if not any(c):
                    continue
                if c[0].isdigit():
                    rows.append(c)
                    continue
                if any(HEADER.search(v) for v in c if v):
                    continue
                if rows:  # 跨頁殘列
                    prev = rows[-1]
                    if len(prev) == len(c):
                        rows[-1] = [a + b for a, b in zip(prev, c)]
                    else:
                        rows[-1] = prev + [v for v in c if v]
                    n_merge += 1
    return rows, n_merge


def parse(pdf):
    d = fitz.open(pdf)
    text = "".join(p.get_text() for p in d)
    # 五、鄉鎮完成判釋比率（文字順序：縣市 鄉鎮 比率 面積）
    sec = text[text.find("五、"):text.find("六、")] if "五、" in text else ""
    cov = {}
    for cty, town, pct, ha in re.findall(r"(\S{2}[縣市])\s+(\S{1,4}[鄉鎮區市])\s+(\d{1,3}(?:\.\d+)?)\s+([\d.]+)", sec):
        cov[(cty + town).replace("台", "臺")] = {"pct": float(pct), "ha": float(ha)}
    m = re.search(r"共判釋\s*(\d+)\s*處新生崩塌地，總\s*面積為\s*([\d.]+)\s*公頃", text)
    m2 = re.search(r"截至\s*(\d+)\s*月\s*(\d+)\s*日完成\s*([\d.]+)\s*％", text)
    # 十五、新生崩塌列表（表格；以 TWD97 兩欄辨識資料列）
    pts, fixes = [], []
    start = next((i for i in range(d.page_count) if "新生崩塌列表" in d[i].get_text()), d.page_count)
    rows, n_merge = list_rows(d, start)
    for row in rows:
        # 各報告欄位位置不同、常夾空欄：先去掉空欄，前四欄依序為編號、縣市、鄉鎮區、村里
        c = [v for v in row if v]
        if len(c) < 7:
            continue
        xy = [int(v) for v in c if re.fullmatch(r"\d{6,7}", v)]
        X = next((v for v in xy if 100000 <= v <= 400000), None)
        Y = next((v for v in xy if 2400000 <= v <= 2800000), None)
        k = next((i for i in range(1, len(c)) if re.fullmatch(r"\d{1,4}(\.\d+)?", c[i])), None)
        if X is None or Y is None or k is None:
            continue
        county, town, vill = admin("".join(c[1:k]))
        for i, v in enumerate(c):  # 日期漏一碼（如「0250725」→ 20250725）
            mm = re.fullmatch(r"(\D*)(0\d{2}[01]\d[0-3]\d)", v)
            if i > k and mm:
                c[i] = mm.group(1) + "2" + mm.group(2)
                fixes.append(f"{c[0]}:{v}→{c[i]}")
        im = next((re.fullmatch(r"(.*?)(20\d{6})", v) for v in c[4:] if re.fullmatch(r"(.*?)(20\d{6})", v)), None)
        df = next((v for v in c if re.search(r"DF\d+", v)), None)
        sensor = (im.group(1) if im else "") or next((v for v in c if re.search(r"Pl[a-z]*Scope|Sentinel|SPOT|Pl[eé]iades|福衛|WorldView|Landsat", v, re.I)), None)
        if sensor in SENSOR_FIX:
            fixes.append(f"{c[0]}:{sensor}→{SENSOR_FIX[sensor]}")
        pts.append({"no": int(c[0]), "county": county, "town": town, "vill": vill, "ha": float(c[k]), "x": X, "y": Y,
                    "df": df, "img": SENSOR_FIX.get(sensor, sensor),
                    "img_date": f"{im.group(2)[:4]}-{im.group(2)[4:6]}-{im.group(2)[6:]}" if im else None})
    return {"n": int(m.group(1)) if m else None, "ha": float(m.group(2)) if m else None,
            "done_pct": float(m2.group(3)) if m2 else None, "done_md": f"{int(m2.group(1)):02d}-{int(m2.group(2)):02d}" if m2 else None,
            "coverage": cov, "points": pts, "parse": {"rows_merged_across_pages": n_merge, "typo_fixes": fixes}}


def main(refetch=False):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    reps = listing()
    print("報告", len(reps))
    for r in reps:
        f = PDF_DIR / f"{r['id']}.pdf"
        if refetch or not f.exists():
            f.write_bytes(get(f"{BASE}/news/download/{r['id']}"))
            time.sleep(1)
        r.update(parse(f))
        r["url"] = f"{BASE}/news/collapses/{r['id']}"
        r["event"] = re.sub(r"-?新生崩塌判釋報告$", "", r["title"])
        print(r["id"], r["date"], r["event"], "處", r["n"], "列表", len(r["points"]), "鄉鎮", len(r["coverage"]))
    # 全部報告的完整解析結果（含全臺點位），供 landslide_report.py 自動產製報告 PDF／CSV
    (POC / "reports_parsed.json").write_text(json.dumps(reps, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    su = gpd.read_file(SU, layer="su_original")[["su_id", "geometry"]]
    rows = [dict(p, rid=r["id"]) for r in reps for p in r["points"]]
    g = gpd.GeoDataFrame(rows, geometry=[Point(p["x"], p["y"]) for p in rows], crs=3826)
    j = gpd.sjoin(g, su, predicate="within", how="inner")
    by_r = {r["id"]: r for r in reps}
    out = {}
    for _, p in j.sort_values(["img_date", "rid"], ascending=False).iterrows():
        r = by_r[p["rid"]]
        cov = r["coverage"].get(p["county"] + p["town"], {}).get("pct")
        out.setdefault(p["su_id"], []).append({"rid": int(p["rid"]), "ev": r["event"], "pub": r["date"], "img_date": p["img_date"], "img": p["img"],
                                               "ha": p["ha"], "vill": p["vill"], "df": p["df"], "cov": cov})
    # POC 範圍內（有點位落在坡面的鄉鎮＋四個主要區）各報告的完成判釋比率
    in_poc = set(j["county"] + j["town"]) | {"高雄市" + t for t in POC_TOWNS}
    towns = {}
    for r in reps:
        for k, v in r["coverage"].items():
            if k in in_poc:
                towns.setdefault(k, []).append([r["id"], r["event"], v["pct"], v["ha"]])
    meta = {"source": "農業部農村發展及水土保持署 BigGIS 入口網「崩塌報告專區」新生崩塌判釋報告（PDF 解析）", "list_url": f"{BASE}/news/collapses",
            "fetched": time.strftime("%Y-%m-%d"), "n_reports": len(reps), "n_points": len(rows), "n_points_in_su": len(j), "n_su": len(out),
            "reports": [{k: r[k] for k in ("id", "event", "date", "url", "n", "ha", "done_pct", "done_md")} | {"n_list": len(r["points"]), "parse": r["parse"]} for r in reps],
            "poc_town_coverage": towns,
            "parse_note": "PDF 表格解析：跨頁切開的列逐欄接回上一列（rows_merged_across_pages）；原始資料錯字照實修正並記錄（typo_fixes，如 PlametScope、日期漏一碼）。"
                          "已知原始資料不一致：0728豪雨報告的鄉鎮統計表中花蓮縣秀林鄉與卓溪鄉面積疑似對調（列表加總為 0.74／2.37 ha，統計表為 2.37／0.74 ha）。",
            "note": "官方判釋為衛星影像（PlanetScope、Sentinel-2 等）人工圈繪 ≥0.1 ha 新生崩塌；完成判釋比率 <100% 表示部分地區受雲覆尚未判釋。只作佐證，不改候選層級。"}
    # 網頁地圖疊圖：落在 POC 坡面內的官方點位（經緯度、報告、事件、影像日期、面積）
    ll = j.to_crs(4326)
    pts = [[round(g_.x, 5), round(g_.y, 5), int(p["rid"]), by_r[p["rid"]]["event"], p["img_date"], p["ha"]] for g_, (_, p) in zip(ll.geometry, j.iterrows())]
    (WEB / "report_points.json").write_text(json.dumps({"source": meta["source"], "list_url": meta["list_url"], "fetched": meta["fetched"], "points": pts},
                                                       ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (POC / "su_reports.json").write_text(json.dumps({"meta": meta, "su": out}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("n_reports", "n_points", "n_points_in_su", "n_su")}, ensure_ascii=False))
    print("POC 鄉鎮完成率", json.dumps(towns, ensure_ascii=False))


if __name__ == "__main__":
    main("--refetch" in sys.argv)
