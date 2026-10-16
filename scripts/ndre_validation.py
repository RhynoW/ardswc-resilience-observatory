"""NDRE 持續下降（候選層級 P3）的分層人工盲判：抽樣、判讀圖、判讀表；判讀完成後對答案。
分層（v2 坡面、有效坡面 ≥3 ha、排除已用於實務測試／專家盲判的坡面）：
  N1 官方圖層期間（下降日 ≤2025-03-25）的下降，且年度圖層類別有裸露（非「無崩塌」）  8 個
  N2 官方圖層期間的下降，年度圖層「無崩塌」                                          8 個
  N3 官方圖層之後（下降日 >2025-03-25）的下降                                        10 個
  N4 對照：無下降、無事件、無照片、年度圖層「無崩塌」                                   6 個
判讀圖（不顯示層別、NDRE、事件、照片）：Sentinel-2 真彩色「前」（參考日前 10–150 天）、「後」（參考日後 20–200 天）、「最新」（2026-06～09），
  以及 Wayback 參考日前最後一張、之後第一張（有才放）。參考日＝下降日；對照組的參考日從下降日分布隨機抽，避免從影像日期看出組別。白線＝有效坡面。
判讀選項：A 後期出現新增或擴大的裸露（崩塌）；B 有裸露但前後無明顯變化；C 無裸露、無變化（植生）；D 有變化但不是崩塌（伐木、工程、道路、河床、雲影、季節）；E 無法判讀。
用法：python scripts/ndre_validation.py build    → poc/ndre_val/review_ndre.html（單一檔案，影像內嵌）、key.json（判讀者不可見）
      python scripts/ndre_validation.py score <判讀結果.json>   → poc/ndre_val/result.json
"""
import base64
import json
import random
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import su_panel_v2 as V2  # noqa: E402
import sentinel_assist as S2  # noqa: E402
import wayback_assist as WB  # noqa: E402
from landslide_incremental import grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "ndre_val"
LAYER_END = "2025-03-25"
SEED = 20261016
QUOTA = {"N1": 8, "N2": 8, "N3": 10, "N4": 6}
OPT = [("A", "後期出現新增或擴大的裸露（崩塌）"), ("B", "有裸露但前後無明顯變化"), ("C", "無裸露、無變化（植生）"),
       ("D", "有變化但不是崩塌（伐木、工程、道路、河床、雲影、季節）"), ("E", "無法判讀")]


def jl(p):
    return json.load(open(p, encoding="utf-8"))


def select():
    nd = jl(POC / "s2ts" / "su_ndre.json")["su"]
    T = {t["su_id"]: t for t in jl(POC / "su_table_eff.json")}
    mu = {u["su_id"]: u for u in jl(POC / "su_meta_eff.json")["units"]}
    ev = {e["su_id"]: e for e in jl(POC / "su_evidence.json")}
    recs = jl(REPO / "webapp/change_detect_viewer/static/poc/su/su_records.json")
    used = set()
    kp = POC / "su_usability" / "key_usability.json"
    if kp.exists():
        used = {v["su_id"] for v in jl(kp).values()}
    ok = lambda s: s in T and s not in used and mu[s]["eff_area_ha"] >= 3.0
    drops = {s: v["drop"] for s, v in nd.items() if v.get("drop") and ok(s)}
    pools = {
        "N1": [s for s, d in drops.items() if d["date"] <= LAYER_END and T[s]["category"] != "無崩塌"],
        "N2": [s for s, d in drops.items() if d["date"] <= LAYER_END and T[s]["category"] == "無崩塌"],
        "N3": [s for s, d in drops.items() if d["date"] > LAYER_END],
        "N4": [s for s, v in nd.items() if ok(s) and v.get("base") is not None and not v.get("drop") and T[s]["category"] == "無崩塌"
               and not (recs.get(s) or {}).get("events") and not ((ev.get(s) or {}).get("field_photos") or {}).get("classes")],
    }
    rng = random.Random(SEED)
    sel = []
    all_dates = sorted(d["date"] for d in drops.values())
    for k, q in QUOTA.items():
        pick = rng.sample(sorted(pools[k]), min(q, len(pools[k])))
        for s in pick:
            ref = drops[s]["date"] if k != "N4" else rng.choice(all_dates)
            sel.append({"su_id": s, "stratum": k, "ref_date": ref, "drop": drops.get(s), "category": T[s]["category"], "eff_ha": mu[s]["eff_area_ha"],
                        "i": T[s]["i"], "cx": T[s]["cx"], "cy": T[s]["cy"], "recent_delta": nd[s].get("recent_delta")})
    print({k: len(v) for k, v in pools.items()}, "抽出", len(sel))
    return sel


def pick_s2(lat, lon, d0, d1, prefer):
    try:
        sc = S2.search_scenes(lat, lon, d0, d1, 40.0, 600.0)
    except Exception:  # noqa: BLE001
        return None
    sc = sorted(sc, key=lambda x: (round(x["cloud"] / 10), abs((V2.dd(x["date"]) - prefer).days)))[:6]
    for s in sc:
        try:
            r = V2.s2_fetch(lat, lon, s["date"])
        except Exception:  # noqa: BLE001
            r = None
        if r:
            return s["date"], r
    return None


def cells_for(t):
    lat, lon = t["lat"], t["lon"]
    d0 = date.fromisoformat(t["ref_date"])
    cells = []
    for tag, a, b, pref in (("前", d0 - timedelta(days=150), d0 - timedelta(days=10), d0 - timedelta(days=30)),
                            ("後", d0 + timedelta(days=20), d0 + timedelta(days=200), d0 + timedelta(days=60)),
                            ("最新", date(2026, 6, 1), date(2026, 9, 30), date(2026, 9, 30))):
        r = pick_s2(lat, lon, a, b, pref)
        if r:
            ymd, (img, bnd) = r
            cells.append({"date": ymd, "kind": "S", "label": f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:]} Sentinel-2（{tag}，對比增強）", "img": V2.enhance(img), "bounds": bnd})
    try:
        fr = [f for f in WB.list_frames(lat, lon, V2.Z) if (f["res_m"] or 9) <= 0.6]
    except Exception:  # noqa: BLE001
        fr = []
    ref = t["ref_date"].replace("-", "")
    before = [f for f in fr if f["date"] <= ref]
    after = [f for f in fr if f["date"] > ref]
    for f, tag in ((before[-1] if before else None, "前"), (after[0] if after else None, "後")):
        if not f:
            continue
        try:
            img, b = WB.fetch_image(lat, lon, V2.HALF_M, f["release"], V2.Z)
            cells.append({"date": f["date"], "kind": "W", "label": f"{f['date'][:4]}-{f['date'][4:6]}-{f['date'][6:]} Wayback（{tag}，{f.get('res_m') or '?'} m）", "img": img, "bounds": b})
        except Exception:  # noqa: BLE001
            pass
    cells.sort(key=lambda c: c["date"])
    return cells


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    sel = select()
    tf, _ = grid()
    lab10 = np.load(POC / "su_lab10_eff.npy")
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    for t in sel:
        t["lon"], t["lat"] = to_ll.transform(t["cx"], t["cy"])
    order = list(range(len(sel)))
    random.Random(SEED + 1).shuffle(order)
    ids = {j: f"NV{n:02d}" for n, j in enumerate(order, 1)}

    def one(j):
        t = sel[j]
        cells = cells_for(t)
        img = V2.compose(cells, t, lab10, tf, to_m, None)
        if img is None:
            return ids[j], None, []
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 72])
        print(ids[j], t["stratum"], len(cells), flush=True)
        return ids[j], "data:image/jpeg;base64," + base64.b64encode(buf).decode(), [c["label"] for c in cells]

    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, range(len(sel))))
    key = {}
    for j, (nid, _, labels) in enumerate(res):
        key[nid] = {**{k: sel[j][k] for k in ("su_id", "stratum", "ref_date", "drop", "category", "eff_ha", "recent_delta", "lat", "lon")}, "panels": labels}
    (OUT / "key.json").write_text(json.dumps(dict(sorted(key.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    items = sorted((nid, uri) for nid, uri, _ in res)
    rows = "".join(
        f'<div class="it" id="it_{i}"><b>{i}</b>' + (f'<img src="{u}" alt="{i} 多時點影像">' if u else "<p>（影像取得失敗）</p>") + '<div class="r">'
        + "".join(f'<label><input type="radio" name="v_{i}" value="{k}" onchange="sv()"><b>{k}</b> {t}</label> ' for k, t in OPT)
        + '<br><span>信心：</span>' + "".join(f'<label><input type="radio" name="c_{i}" value="{c}" onchange="sv()">{c}</label> ' for c in ("高", "中", "低"))
        + f' <input id="n_{i}" placeholder="備註（例如：變化在白線外、雲影、伐木）" oninput="sv()"></div></div>' for i, u in items)
    ids_ = [i for i, _ in items]
    html = f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>NDRE 盲判</title>
<style>body{{font:15px/1.6 system-ui,'Noto Sans TC',sans-serif;margin:0;background:#f6f5f2;color:#222}}header{{position:sticky;top:0;background:#222;color:#fff;padding:8px 16px;z-index:9}}header button{{margin-left:12px}}
main{{max-width:1560px;margin:auto;padding:12px 16px}}.it{{background:#fff;border:1px solid #ddd;padding:8px;margin:0 0 16px}}img{{width:100%;display:block}}label{{margin-right:12px;white-space:nowrap}}input[id^=n_]{{width:40%}}.done{{border-left:6px solid #2a7}}</style>
<header><b>坡面變化盲判（{len(ids_)} 個坡面）</b><span id="p"></span><button onclick="exp()">匯出 JSON</button></header><main>
<p>每張圖是同一個坡面在幾個時間點的影像，由舊到新排列：Sentinel-2（10 m，已做對比增強、僅供目視）的「前」「後」「最新」三期，以及有的話 Wayback（0.3–0.5 m）在「前」「後」各一張。<b>白線＝坡度 ≥15° 的有效坡面範圍</b>（統計單元外框，不是裸露邊界）。請只依影像判斷這個坡面在「前」到「後」之間有沒有新增或擴大的裸露；不要查詢其他資料。Wayback 對地形資料常有 10–25 m 偏移，線與影像可能錯位，請以大致位置判斷。</p>
<p>選項：A 後期出現新增或擴大的裸露（崩塌）；B 有裸露但前後無明顯變化；C 無裸露、無變化（植生）；D 有變化但不是崩塌（伐木、工程、道路、河床、雲影、季節）；E 無法判讀（雲、陰影、解析度）。答案會暫存在這個瀏覽器，完成後按「匯出 JSON」。</p>{rows}</main>
<script>const IDS={json.dumps(ids_)};let S={{}};try{{S=JSON.parse(localStorage.getItem('ndre_val')||'{{}}')}}catch(e){{}}
function sv(){{IDS.forEach(i=>{{const v=document.querySelector('input[name=v_'+i+']:checked'),c=document.querySelector('input[name=c_'+i+']:checked');S[i]={{verdict:v?v.value:null,conf:c?c.value:null,note:document.getElementById('n_'+i).value}}}});try{{localStorage.setItem('ndre_val',JSON.stringify(S))}}catch(e){{}}prog()}}
function prog(){{let n=0;IDS.forEach(i=>{{const s=S[i],d=s&&s.verdict&&s.conf;document.getElementById('it_'+i).classList.toggle('done',!!d);if(d)n++}});document.getElementById('p').textContent='　'+n+' / '+IDS.length+' 已完成'}}
IDS.forEach(i=>{{const s=S[i];if(!s)return;if(s.verdict){{const e=document.querySelector('input[name=v_'+i+'][value='+s.verdict+']');if(e)e.checked=true}}if(s.conf){{const e=document.querySelector('input[name=c_'+i+'][value='+s.conf+']');if(e)e.checked=true}}document.getElementById('n_'+i).value=s.note||''}});prog();
function exp(){{const o={{}};IDS.forEach(i=>{{const s=S[i];if(s&&s.verdict)o[i]=s}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{{type:'application/json'}}));a.download='ndre_val_human.json';a.click()}}</script></html>"""
    (OUT / "review_ndre.html").write_text(html, encoding="utf-8")
    print("完成", len(ids_), "→", OUT / "review_ndre.html", round((OUT / "review_ndre.html").stat().st_size / 1e6, 1), "MB")


def score(path):
    key = jl(OUT / "key.json")
    hum = jl(path)
    by = {}
    for nid, k in key.items():
        h = hum.get(nid)
        if not h or not h.get("verdict"):
            continue
        by.setdefault(k["stratum"], []).append(h["verdict"])
    from collections import Counter
    res = {s: {"n": len(v), "verdicts": dict(Counter(v)), "A_share": round(v.count("A") / len(v), 2) if v else None,
               "A_share_excl_E": round(v.count("A") / max(len(v) - v.count("E"), 1), 2)} for s, v in sorted(by.items())}
    drops = [v for s, vs in by.items() if s != "N4" for v in vs]
    res["P3_precision_A_excl_E"] = round(drops.count("A") / max(len(drops) - drops.count("E"), 1), 2) if drops else None
    (OUT / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    elif sys.argv[1] == "score":
        score(sys.argv[2])
