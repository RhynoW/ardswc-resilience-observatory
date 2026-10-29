"""新生崩塌坡面報告自動產製（PDF＋CSV＋GeoJSON）：人工圈繪的新生崩塌點位 → 統計表、地圖、坡面清單與點位檔。
自動化的分工：圈繪新生崩塌由人做（坡面尺度的衛星指標只看得到占坡面兩成以上的崩塌）；圈繪完成後，
報告的統計、地圖、列表與每個坡面的證據鏈由程式在幾分鐘內產出，並同時輸出機器可讀的點位檔。
輸入（二擇一）：
  --report <id>｜--all：水保署 BigGIS 崩塌報告專區的報告（biggis_reports.py 解析結果 data/biggis_interp/poc/reports_parsed.json）
  --csv <檔案> --event <事件名>：自備點位（欄位 x,y［TWD97］或 lon,lat；ha；img_date；可選 img、county、town、vill、df）
輸出：webapp/change_detect_viewer/static/reports/<key>/{report.pdf, landslides.csv, landslides.geojson}；static/reports/index.json（網站 /reports 清單）
報告內容：
  摘要（全區與高雄 POC）、影像說明、鄉鎮統計（附官方完成判釋率，並逐列比對官方統計表）、面積分級、土石流潛勢溪流、
  高雄 POC 地圖（20 m DTM 陰影＋坡面＋點位）、POC 坡面清單（候選層級、NDRE、道路建物、潛勢溪流）、POC 點位列表、方法與限制
用法：python scripts/landslide_report.py --all
"""
import argparse
import base64
import csv
import html
import io
import json
import re
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import Point

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
POC = REPO / "data" / "biggis_interp" / "poc"
SU = POC / "su_export" / "slope_units_SUv2.gpkg"
DTM = REPO / "data" / "dtm" / "tw_dtm20.tif"
WEB = REPO / "webapp" / "change_detect_viewer" / "static"
OUT = WEB / "reports"
SITE = "https://rhynowu-ardswc-resilience-observatory.hf.space"
FONT = Path("C:/Windows/Fonts/NotoSansTC-VF.ttf")
TIER_COLOR = {"優先人工判讀": "#c62828", "一般判讀": "#1565c0", "資料不足": "#757575"}
AREA_BINS = [(0, 0.5, "<0.5"), (0.5, 1, "0.5–1"), (1, 5, "1–5"), (5, 10, "5–10"), (10, 1e9, "≥10")]
esc = html.escape


def load_context():
    su = gpd.read_file(SU, layer="su_original")[["su_id", "geometry"]]
    rec = json.loads((WEB / "poc" / "su" / "su_records.json").read_text(encoding="utf-8"))
    idx = {r[0]: r[1] for r in json.loads((WEB / "poc" / "su" / "su_index.json").read_text(encoding="utf-8"))["rows"]}
    eff = {u["su_id"]: u["eff_area_ha"] for u in json.loads((POC / "su_meta_eff.json").read_text(encoding="utf-8"))["units"]}
    outline = su.geometry.buffer(1).union_all().buffer(-1)
    return {"su": su, "rec": rec, "tier": idx, "eff": eff, "outline": outline}


def from_csv(path):
    pts = []
    for i, r in enumerate(csv.DictReader(open(path, encoding="utf-8-sig")), 1):
        if r.get("x") and r.get("y"):
            x, y = float(r["x"]), float(r["y"])
        else:
            x, y = gpd.GeoSeries([Point(float(r["lon"]), float(r["lat"]))], crs=4326).to_crs(3826).iloc[0].coords[0]
        pts.append({"no": int(r.get("no") or i), "county": r.get("county", ""), "town": r.get("town", ""), "vill": r.get("vill", ""),
                    "ha": float(r["ha"]), "x": x, "y": y, "df": r.get("df") or None, "img": r.get("img") or None, "img_date": r.get("img_date") or None})
    return pts


def enrich(pts, ctx):
    g = gpd.GeoDataFrame(pts, geometry=[Point(p["x"], p["y"]) for p in pts], crs=3826)
    j = gpd.sjoin(g, ctx["su"], predicate="within", how="left")
    j = j[~j.index.duplicated()]
    ll = g.to_crs(4326)
    for p, sid, pt in zip(pts, j["su_id"], ll.geometry):
        p["lon"], p["lat"] = round(pt.x, 6), round(pt.y, 6)
        p["su_id"] = sid if isinstance(sid, str) else None
        r = ctx["rec"].get(p["su_id"]) if p["su_id"] else None
        p["tier"] = ctx["tier"].get(p["su_id"]) if p["su_id"] else None
        p["eff_ha"] = ctx["eff"].get(p["su_id"])
        p["ndre_drop"] = ((r or {}).get("ndre") or {}).get("drop", {}) or {}
        o, d = (r or {}).get("osm") or {}, (r or {}).get("debris") or {}
        p["road100"] = sum((o.get("road_100") or {}).values()) if o else None
        p["bld100"] = o.get("bld_100") if o else None
        p["ws"] = ";".join(w[0] for w in d.get("ws", [])) if d else ""
        p["affect"] = d.get("affect", []) if d else []
    return pts


def poc_map(pts, ctx):
    poc = [p for p in pts if p["su_id"]]
    if not poc:
        return None
    minx, miny, maxx, maxy = ctx["outline"].bounds
    with rasterio.open(DTM) as ds:
        w = from_bounds(minx, miny, maxx, maxy, ds.transform)
        f = 5
        z = ds.read(1, window=w, out_shape=(int(w.height // f), int(w.width // f))).astype(float)
    z[z < -100] = np.nan
    dy, dx = np.gradient(np.nan_to_num(z, nan=0), 20 * f)
    az, alt = np.radians(315), np.radians(45)
    slope, aspect = np.arctan(np.hypot(dx, dy)), np.arctan2(-dx, dy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    fig, ax = plt.subplots(figsize=(7.2, 7.2 * (maxy - miny) / (maxx - minx)), dpi=170)
    ax.imshow(hs, cmap="gray", extent=(minx, maxx, miny, maxy), vmin=-0.2, vmax=1.1, alpha=0.9)
    gpd.GeoSeries([ctx["outline"]], crs=3826).boundary.plot(ax=ax, color="#333", linewidth=0.8)
    sids = sorted({p["su_id"] for p in poc})
    sub = ctx["su"][ctx["su"]["su_id"].isin(sids)].copy()
    sub["tier"] = sub["su_id"].map(ctx["tier"])
    for t, c in TIER_COLOR.items():
        s = sub[sub["tier"] == t]
        if len(s):
            s.plot(ax=ax, facecolor=c, edgecolor=c, alpha=0.35, linewidth=0.6)
    xs, ys = [p["x"] for p in poc], [p["y"] for p in poc]
    ax.scatter(xs, ys, s=[8 + 30 * min(p["ha"], 5) for p in poc], c="#ff6f00", edgecolors="white", linewidths=0.6, zorder=5)
    ax.plot([maxx - 11000, maxx - 1000], [miny + 1500] * 2, color="k", lw=3)
    ax.text(maxx - 6000, miny + 2300, "10 km", ha="center", fontproperties=font_manager.FontProperties(fname=str(FONT), size=7))
    ax.annotate("N", xy=(minx + 2500, maxy - 1500), xytext=(minx + 2500, maxy - 5500), ha="center", fontsize=10,
                arrowprops=dict(arrowstyle="-|>", color="k"))
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hd = [Patch(facecolor=c, alpha=0.5, label=f"坡面：{t}") for t, c in TIER_COLOR.items() if t in set(sub["tier"])]
    hd.append(Line2D([], [], marker="o", color="w", markerfacecolor="#ff6f00", markersize=7, label="新生崩塌（大小依面積）"))
    ax.legend(handles=hd, loc="lower left", prop=font_manager.FontProperties(fname=str(FONT), size=6.5), framealpha=0.85)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    from PIL import Image
    jpg = io.BytesIO()
    Image.open(buf).convert("RGB").save(jpg, "JPEG", quality=80, optimize=True)  # 陰影底圖用 JPEG，PDF 小很多
    return base64.b64encode(jpg.getvalue()).decode()


def table(head, rows, num=()):
    th = "".join(f"<th{' class=n' if i in num else ''}>{esc(h)}</th>" for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f"<td{' class=n' if i in num else ''}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def f1(v, d=1):
    return "—" if v is None else f"{v:,.{d}f}"


def build_html(meta, pts, cov, ctx, map_b64):
    n, ha = len(pts), sum(p["ha"] for p in pts)
    dates = sorted(p["img_date"] for p in pts if p["img_date"])
    poc = [p for p in pts if p["su_id"]]
    by_su = defaultdict(list)
    for p in poc:
        by_su[p["su_id"]].append(p)
    tiers = Counter(ctx["tier"].get(s) for s in by_su)
    road = sum(1 for s, L in by_su.items() if (L[0]["road100"] or 0) > 0)
    bld = sum(1 for s, L in by_su.items() if (L[0]["bld100"] or 0) > 0)
    ws = sum(1 for s, L in by_su.items() if L[0]["ws"])
    aff = sum(1 for s, L in by_su.items() if L[0]["affect"])
    # 影像
    img = defaultdict(lambda: [0, 0.0])
    for p in pts:
        k = (p["img"] or "—", p["img_date"] or "—")
        img[k][0] += 1
        img[k][1] += p["ha"]
    img_rows = [[esc(a), esc(b), c, f1(h, 2), f1(100 * c / n, 1)] for (a, b), (c, h) in sorted(img.items(), key=lambda x: x[0][1])]
    # 鄉鎮（逐列比對官方統計表）
    town = defaultdict(lambda: [0, 0.0])
    for p in pts:
        town[(p["county"], p["town"])][0] += 1
        town[(p["county"], p["town"])][1] += p["ha"]
    t_rows, n_ok, n_cmp = [], 0, 0
    for (c, t), (k, h) in sorted(town.items(), key=lambda x: -x[1][1]):
        o = cov.get(c + t)
        if o:
            n_cmp += 1
            ok = abs(round(h, 1) - round(o["ha"], 1)) <= 0.1 + 1e-9
            n_ok += ok
            chk = "一致" if ok else f"<b>官方 {f1(o['ha'], 2)}</b>"
        else:
            chk = "—"
        t_rows.append([esc(c), esc(t), "—" if not o else f"{o['pct']:.0f}%", k, f1(h, 2), chk])
    a_rows = []
    for lo, hi, lab in AREA_BINS:
        s = [p for p in pts if lo <= p["ha"] < hi]
        a_rows.append([lab, len(s), f1(100 * len(s) / n, 1), f1(sum(p["ha"] for p in s), 2), f1(100 * sum(p["ha"] for p in s) / ha if ha else 0, 1)])
    dfc = defaultdict(lambda: [0, 0.0])
    for p in pts:
        if p["df"]:
            dfc[p["df"]][0] += 1
            dfc[p["df"]][1] += p["ha"]
    df_rows = [[esc(k), c, f1(h, 2)] for k, (c, h) in sorted(dfc.items(), key=lambda x: -x[1][1])[:40]]
    su_rows = []
    for s, L in sorted(by_su.items(), key=lambda x: -sum(p["ha"] for p in x[1])):
        p0, h = L[0], sum(p["ha"] for p in L)
        tier = ctx["tier"].get(s, "—")
        dr = p0["ndre_drop"].get("date") if p0["ndre_drop"] else None
        su_rows.append([f'<a href="{SITE}/su#{esc(s)}">{esc(s)}</a>', f'<span style="color:{TIER_COLOR.get(tier, "#333")}">{esc(tier)}</span>', len(L), f1(h, 2),
                        f1(100 * h / p0["eff_ha"], 0) + "%" if p0["eff_ha"] else "—", esc(dr or "—"),
                        f1((p0["road100"] or 0) / 1000, 2) if p0["road100"] is not None else "—", "—" if p0["bld100"] is None else p0["bld100"],
                        esc(p0["ws"] or "—"), esc("、".join(f"{a[0]}（{a[1] if a[1] is not None else '未載'}戶）" for a in p0["affect"]) or "—")])
    p_rows = [[p["no"], esc(p["town"]), esc(p["vill"]), f1(p["ha"], 2), esc(p["img_date"] or "—"), esc(p["img"] or "—"), esc(p["df"] or "—"), esc(p["su_id"])]
              for p in sorted(poc, key=lambda p: p["no"])]
    src = (f'水保署 BigGIS 崩塌報告專區：<a href="{esc(meta["url"])}">{esc(meta["title"])}</a>（發布 {esc(meta["date"])}）' if meta.get("url")
           else esc(meta.get("source", "自備點位")))
    poc_txt = (f"{len(poc)} 處、{f1(sum(p['ha'] for p in poc), 2)} ha，落在 {len(by_su)} 個坡面：優先人工判讀 {tiers.get('優先人工判讀', 0)}、一般判讀 {tiers.get('一般判讀', 0)}、資料不足 {tiers.get('資料不足', 0)}。"
               f"其中 {road} 個坡面外推 100 m 內有道路、{bld} 個有建物（OpenStreetMap，數字是下限），{ws} 個位在土石流潛勢溪流集水區、{aff} 個碰到潛勢溪流影響範圍。"
               if poc else "沒有點位落在高雄 POC 坡面內。")
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>{esc(meta['event'])} 新生崩塌坡面報告</title>
<style>
@font-face{{font-family:NotoTC;src:url('{FONT.as_uri()}')}}
@page{{size:A4;margin:16mm 14mm 16mm 14mm}}
body{{font-family:NotoTC,'Microsoft JhengHei',sans-serif;font-size:10pt;color:#1f2a24;line-height:1.5}}
h1{{font-size:17pt;color:#2c5f2d;margin:0 0 2pt}} h2{{font-size:12.5pt;color:#2c5f2d;border-bottom:1.5pt solid #2c5f2d;padding-bottom:2pt;margin:14pt 0 6pt;break-after:avoid}}
.sub{{color:#5e6b63;font-size:9pt}} .note{{background:#fff7e6;border-left:3pt solid #ff6f00;padding:5pt 8pt;font-size:9pt;margin:8pt 0}}
.kpi{{display:flex;gap:6pt;margin:6pt 0}} .kpi div{{flex:1;background:#eef3ee;border-radius:4pt;padding:6pt 8pt}} .kpi b{{display:block;font-size:15pt;color:#2c5f2d}}
table{{border-collapse:collapse;width:100%;font-size:8.5pt;margin:4pt 0 6pt}} th{{background:#2c5f2d;color:#fff;font-weight:600;text-align:left;padding:3pt 4pt}}
td{{border-bottom:.5pt solid #d5ddd6;padding:2.5pt 4pt}} .n{{text-align:right}} tr{{break-inside:avoid}} thead{{display:table-header-group}}
a{{color:#1565c0;text-decoration:none}} img.map{{width:100%;max-height:200mm;object-fit:contain;border:.5pt solid #ccc;break-inside:avoid}} p{{margin:3pt 0}} .small{{font-size:8.5pt;color:#5e6b63}}
</style></head><body>
<h1>{esc(meta['event'])}｜新生崩塌坡面報告</h1>
<div class="sub">自動產製・TASTI_SatDashboard 坡地韌性哨兵・{time.strftime('%Y-%m-%d %H:%M')}・資料：{src}</div>
<div class="note">本報告由程式從新生崩塌點位自動產製，<b>不是官方報告</b>；崩塌由水保署以衛星影像人工圈繪，統計表逐列與官方統計表比對。高雄 POC 段落加上坡面證據鏈（候選層級、Sentinel-2 紅邊、道路建物、土石流潛勢溪流），只供巡查排序參考，是否現勘由人決定。</div>
<h2>一、摘要</h2>
<div class="kpi"><div><b>{n:,}</b>處新生崩塌（全區）</div><div><b>{f1(ha, 1)}</b>ha 合計面積</div><div><b>{len(town)}</b>個鄉鎮區</div><div><b>{len(by_su)}</b>個高雄 POC 坡面</div></div>
<p>影像日期 {esc(dates[0] if dates else '—')} 至 {esc(dates[-1] if dates else '—')}；{('官方截至 ' + meta['done_md'] + ' 完成 ' + f1(meta['done_pct'], 1) + '% 警戒區判釋；') if meta.get('done_pct') else ''}高雄 POC：{poc_txt}</p>
<h2>二、使用影像</h2>{table(["影像","影像日期","處數","面積 ha","處數 %"], img_rows, (2,3,4))}
<h2>三、縣市及鄉鎮區統計</h2>
<p class="small">完成判釋比率取自官方報告（&lt;100% 表示部分地區受雲覆尚未判釋，沒有點位不代表沒有崩塌）。最後一欄比對官方統計表：{n_ok}/{n_cmp} 列一致。</p>
{table(["縣市","鄉鎮區","完成判釋","處數","面積 ha","與官方統計表"], t_rows, (2,3,4))}
<h2>四、面積分級</h2>{table(["面積（ha）","處數","處數 %","面積 ha","面積 %"], a_rows, (1,2,3,4))}
<h2>五、土石流潛勢溪流（點位所在）</h2>{table(["潛勢溪流","處數","面積 ha"], df_rows, (1,2)) if df_rows else '<p class="small">沒有點位標註土石流潛勢溪流。</p>'}
{f'<h2>六、高雄 POC 地圖</h2><img class="map" src="data:image/jpeg;base64,{map_b64}"><p class="small">底圖：內政部 20 m DTM 陰影；坡面：SUv2（集水區重疊法），顏色為候選層級；橘點：新生崩塌。</p>' if map_b64 else ''}
{f'<h2>七、高雄 POC 坡面清單（依崩塌面積排序）</h2><p class="small">占有效坡面＝崩塌面積 ÷ 坡度 ≥15° 的有效坡面面積；坡面中位數約只看得到占有效坡面兩成以上的變化。點 SU 編號開啟該坡面的證據鏈頁。</p>' + table(["坡面","候選層級","處數","面積 ha","占有效坡面","NDRE 持續下降","道路 100 m（km）","建物 100 m","潛勢溪流集水區","影響範圍（保全住戶）"], su_rows, (2,3,4,6,7)) if su_rows else ''}
{f'<h2>八、高雄 POC 新生崩塌列表</h2>' + table(["編號","鄉鎮區","村里","面積 ha","影像日期","影像","潛勢溪流","坡面"], p_rows, (0,3)) if p_rows else ''}
<h2>九、方法與限制</h2>
<p>點位：{src}。PDF 表格以程式解析（跨頁切開的列逐欄接回、原始錯字照實修正並記錄）；完整點位見同目錄 <b>landslides.csv</b>（UTF-8）與 <b>landslides.geojson</b>。</p>
<p>坡面：SUv2 集水區重疊法（內政部 20 m DEM），點位落在哪個坡面以 TWD97 座標判定。候選層級為 POC 規則 v1（未納入本報告事件），只排「先看誰」，不給風險分數。道路與建物來自 OpenStreetMap（ODbL），山區常缺漏；土石流潛勢溪流來自水保署 BigGIS 公開圖層。</p>
<p>自動化的分工：坡面尺度的 Sentinel-2 紅邊指標在崩塌占坡面 ≥20% 時約抓到 35%、全部只有約 3%（2025–2026 高雄官方點位檢驗；官方崩塌中位數 0.3 ha），所以<b>圈繪新生崩塌必須由人做</b>；圈繪完成後，本報告的統計、地圖、列表與坡面證據鏈可在幾分鐘內自動產出。</p>
</body></html>"""


def render_pdf(html_text, pdf_path, title):
    from playwright.sync_api import sync_playwright
    tmp = pdf_path.with_suffix(".html")
    tmp.write_text(html_text, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(tmp.as_uri(), wait_until="load")
        pg.evaluate("document.fonts.ready")
        foot = f'<div style="font-size:7pt;color:#777;width:100%;text-align:center;font-family:sans-serif">{esc(title)}・自動產製（非官方報告）・<span class="pageNumber"></span>/<span class="totalPages"></span></div>'
        pg.pdf(path=str(pdf_path), format="A4", print_background=True, display_header_footer=True, header_template="<span></span>",
               footer_template=foot, margin={"top": "14mm", "bottom": "16mm", "left": "12mm", "right": "12mm"})
        b.close()
    tmp.unlink()


CSV_COLS = ["report_id", "event", "no", "county", "town", "village", "area_ha", "x_twd97", "y_twd97", "lon", "lat", "debris_stream", "sensor", "image_date",
            "su_id", "su_tier", "su_eff_ha", "ndre_drop_date", "road_100m_m", "buildings_100m", "debris_watershed", "su_page"]


def write_files(key, meta, pts, ctx, cov):
    d = OUT / str(key)
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in pts:
        rows.append([meta.get("id", ""), meta["event"], p["no"], p["county"], p["town"], p["vill"], p["ha"], int(p["x"]), int(p["y"]), p["lon"], p["lat"],
                     p["df"] or "", p["img"] or "", p["img_date"] or "", p["su_id"] or "", p["tier"] or "", p["eff_ha"] if p["eff_ha"] is not None else "",
                     (p["ndre_drop"] or {}).get("date", ""), "" if p["road100"] is None else p["road100"], "" if p["bld100"] is None else p["bld100"], p["ws"],
                     f"{SITE}/su#{p['su_id']}" if p["su_id"] else ""])
    with open(d / "landslides.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        w.writerows(rows)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [r[9], r[10]]}, "properties": dict(zip(CSV_COLS, r))} for r in rows]
    (d / "landslides.geojson").write_text(json.dumps({"type": "FeatureCollection", "name": f"{meta['event']} 新生崩塌", "features": feats}, ensure_ascii=False), encoding="utf-8")
    t0 = time.time()
    render_pdf(build_html(meta, pts, cov, ctx, poc_map(pts, ctx)), d / "report.pdf", f"{meta['event']} 新生崩塌坡面報告")
    poc = [p for p in pts if p["su_id"]]
    return {"key": str(key), "id": meta.get("id"), "event": meta["event"], "date": meta.get("date"), "url": meta.get("url"), "n": len(pts),
            "ha": round(sum(p["ha"] for p in pts), 2), "n_poc": len(poc), "n_su": len({p["su_id"] for p in poc}),
            "img_from": min((p["img_date"] for p in pts if p["img_date"]), default=None), "img_to": max((p["img_date"] for p in pts if p["img_date"]), default=None),
            "files": {"pdf": f"reports/{key}/report.pdf", "csv": f"reports/{key}/landslides.csv", "geojson": f"reports/{key}/landslides.geojson"},
            "pdf_kb": round((d / "report.pdf").stat().st_size / 1024), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=int)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--event")
    a = ap.parse_args()
    global FP
    FP = font_manager.FontProperties(fname=str(FONT))
    plt.rcParams["font.family"] = FP.get_name()
    font_manager.fontManager.addfont(str(FONT))
    ctx = load_context()
    idx_f = OUT / "index.json"
    index = {e["key"]: e for e in json.loads(idx_f.read_text(encoding="utf-8"))["reports"]} if idx_f.exists() else {}
    jobs = []
    if a.csv:
        key = re.sub(r"\W+", "_", a.event or Path(a.csv).stem)
        jobs.append((key, {"event": a.event or Path(a.csv).stem, "title": a.event, "source": f"自備點位 {Path(a.csv).name}"}, from_csv(a.csv), {}))
    else:
        reps = json.loads((POC / "reports_parsed.json").read_text(encoding="utf-8"))
        for r in reps:
            if a.all or r["id"] == a.report:
                jobs.append((r["id"], r, [dict(p) for p in r["points"]], r["coverage"]))
    for key, meta, pts, cov in jobs:
        e = write_files(key, meta, enrich(pts, ctx), ctx, cov)
        index[e["key"]] = e
        print(e["key"], e["event"], "處", e["n"], "POC", e["n_poc"], "坡面", e["n_su"], "PDF", e["pdf_kb"], "KB", e["seconds"], "s")
    out = sorted(index.values(), key=lambda e: e.get("date") or "", reverse=True)
    idx_f.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d"), "site": SITE, "reports": out}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
