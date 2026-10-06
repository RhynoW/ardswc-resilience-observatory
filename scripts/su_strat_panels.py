"""分層人工驗證的盲測圖與判讀表：對 su_strat_select.py 選出的 24 個坡面，各做一張 4 欄圖（Wayback 較早、Wayback 最新、
Sentinel-2 2024-04-04、Sentinel-2 2025-03-25；白線＝有效坡面（≥15°）輪廓），隨機編號 SV01…SV24，不顯示層別、事件、照片或系統資訊。
輸出 poc/su_strat/SVxx.png、key.json（編號對照與層別，判讀者不可見）、review.html（判讀表：5 選 1＋信心＋是否可見小型崩塌／落石＋備註；可匯出 JSON）。
判讀選項：A 近期有新增／擴大崩塌；B 有崩塌但近期無變化（舊崩塌持續或縮減）；C 無崩塌（植生／穩定）；D 有裸露但非崩塌（河床、道路、工程、農地）；E 無法判讀。
用法：python scripts/su_strat_panels.py
"""
import json
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from pyproj import Transformer

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import wayback_assist as WB  # noqa: E402
from landslide_incremental import grid  # noqa: E402
from su_wayback_sample import outline_px  # noqa: E402
from su_outline_offset_check import dem10, hillshade  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_strat"
Z, HALF_M, PW = 18, 380.0, 480
OPT = [("A", "近期有新增／擴大崩塌"), ("B", "有崩塌但近期無變化（舊崩塌持續或縮減）"), ("C", "無崩塌（植生／穩定）"), ("D", "有裸露但非崩塌（河床、道路、工程、農地）"), ("E", "無法判讀")]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def dd(s):
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def bar(img, text):
    from PIL import Image, ImageDraw, ImageFont
    b = Image.new("RGB", (img.shape[1], 30), (30, 30, 30))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msjh.ttc", 14)       # cv2.putText 無法繪中文，改用 PIL
    except Exception:  # noqa: BLE001
        font = ImageFont.load_default()
    ImageDraw.Draw(b).text((6, 6), text, fill=(255, 255, 255), font=font)
    return np.vstack([cv2.cvtColor(np.array(b), cv2.COLOR_RGB2BGR), img])


def wayback_cols(t, lab10, tf, to_m, shift=None):
    lon, lat = t["lon"], t["lat"]
    fr = [f for f in WB.list_frames(lat, lon, Z) if f["date"] >= "20170101" and (f["res_m"] or 9) <= 0.6]
    if not fr:
        return [None, None]
    late = fr[-1]
    early = [f for f in fr if (dd(late["date"]) - dd(f["date"])).days >= 365]
    early = early[-1] if early else None
    cols = []
    for f in (early, late):
        if f is None:
            cols.append(None)
            continue
        img, bounds = WB.fetch_image(lat, lon, HALF_M, f["release"], Z)
        h, w = img.shape[:2]
        vis = img.copy()
        polys = outline_px(lab10, t["i"], tf, bounds, w, h, to_m)
        for poly in polys:
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (255, 255, 255), 2, cv2.LINE_AA)
        tag = ""
        if shift is not None and f is late:                       # 最新一幀：另畫「依實測偏移平移後」的黃線（僅供對照；偏移來自 su_outline_offset_check.py）
            mpp = (bounds[2] - bounds[0]) / w * np.cos(np.radians(lat))
            dxp, dyp = shift[0] / mpp, shift[1] / mpp
            for poly in polys:
                cv2.polylines(vis, [(poly + np.array([dxp, dyp])).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 255), 2, cv2.LINE_AA)
            tag = f" 黃線＝依實測偏移({shift[0]:+.0f},{shift[1]:+.0f})m"
        vis = cv2.resize(vis, (PW, int(h * PW / w)), interpolation=cv2.INTER_AREA)
        cols.append(bar(vis, f"{f['date']} {f.get('res_m') or '?'}m Wayback" + tag))
    return cols


def s2_cols(t, lab10, tf, arrs):
    cx = int((t["cx"] - tf.c) // 10)
    cy = int((tf.f - t["cy"]) // 10)
    hw = int(HALF_M / 10)
    y0, y1, x0, x1 = max(cy - hw, 0), cy + hw, max(cx - hw, 0), cx + hw
    cols = []
    m = (lab10[y0:y1, x0:x1] == t["i"]).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for d, a in arrs.items():
        im = cv2.cvtColor(a[y0:y1, x0:x1], cv2.COLOR_RGB2BGR)
        s = PW / im.shape[1]
        im = cv2.resize(im, (PW, int(im.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
        for c in cs:
            cv2.polylines(im, [(c * s).astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        cols.append(bar(im, f"{d[:4]}-{d[4:6]}-{d[6:]} 10m Sentinel-2"))
    return cols


def dtm_col(t, tf, D, hs, slope, lab_orig_i, lab10_orig):
    cx = int((t["cx"] - tf.c) // 10)
    cy = int((tf.f - t["cy"]) // 10)
    hw = int(HALF_M / 10)
    y0, y1, x0, x1 = max(cy - hw, 0), cy + hw, max(cx - hw, 0), cx + hw
    base = np.clip((hs[y0:y1, x0:x1] * 255), 0, 255).astype(np.uint8)
    im = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    sl = slope[y0:y1, x0:x1]
    tint = np.zeros_like(im)
    tint[sl >= 30] = (0, 0, 255)
    tint[(sl >= 15) & (sl < 30)] = (0, 200, 255)
    mask = (sl >= 15)[..., None]
    im = np.where(mask, (0.62 * im + 0.38 * tint).astype(np.uint8), im)
    s = PW / im.shape[1]
    im = cv2.resize(im, (PW, int(im.shape[0] * s)), interpolation=cv2.INTER_CUBIC)
    for lab, color, th in ((lab10_orig == lab_orig_i, (255, 200, 0), 1), (None, (255, 255, 255), 2)):
        pass
    mfull = (lab10_orig[y0:y1, x0:x1] == lab_orig_i).astype(np.uint8)
    cs, _ = cv2.findContours(mfull, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        cv2.polylines(im, [(c * s).astype(np.int32)], True, (255, 200, 0), 1, cv2.LINE_AA)       # 青藍細線＝原始（完整）Slope Unit
    return bar(im, "DTM 20m 山陰影＋坡度(黃15–30°/紅≥30°) 青線=原始SU")


def main():
    OUT.mkdir(exist_ok=True)
    sel = json.load(open(OUT / "selected.json", encoding="utf-8"))
    lab10 = np.load(POC / "su_lab10_eff.npy")
    tf, _ = grid()
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    off = {}
    try:
        for r_ in json.load(open(OUT / "offset_check.json", encoding="utf-8")).get("rows", []):
            if "dx_m" in r_ and r_["resp"] >= 0.3:
                off[r_["su_id"]] = (r_["dx_m"], r_["dy_m"])
    except Exception:  # noqa: BLE001
        pass
    D = dem10(tf, *lab10.shape)
    hs = hillshade(D)
    gy_, gx_ = np.gradient(D, 10.0)
    slope = np.degrees(np.arctan(np.hypot(gx_, gy_)))
    lab10_orig = np.load(POC / "su_lab10.npy")
    orig_i = {t_["su_id"]: t_["i"] for t_ in json.load(open(POC / "su_table.json", encoding="utf-8"))}
    arrs = {}
    for d_ in ("20240404", "20250325"):
        a_ = np.load(POC / "s2" / f"rgb_{d_}.npy")
        a_[~np.load(POC / "s2" / f"valid_{d_}.npy")] = 128          # Copernicus 浮水印格（無效）以中灰填色，避免浮水印被誤判為地物
        arrs[d_] = a_
    for t in sel:
        t["lon"], t["lat"] = to_ll.transform(t["cx"], t["cy"])
    order = list(range(len(sel)))
    random.Random(20261006).shuffle(order)
    ids = {j: f"SV{n:02d}" for n, j in enumerate(order, 1)}

    def one(j):
        t = sel[j]
        try:
            w = wayback_cols(t, lab10, tf, to_m, off.get(t["su_id"]))
        except Exception as e:  # noqa: BLE001
            log("Wayback 失敗", ids[j], repr(e)[:80])
            w = [None, None]
        cols = [c for c in w + s2_cols(t, lab10, tf, arrs) + [dtm_col(t, tf, D, hs, slope, orig_i[t["su_id"]], lab10_orig)] if c is not None]
        h = max(c.shape[0] for c in cols)
        cols = [np.vstack([c, np.zeros((h - c.shape[0], c.shape[1], 3), np.uint8)]) if c.shape[0] < h else c for c in cols]
        panel = np.hstack(cols)
        cv2.putText(panel, ids[j], (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{ids[j]}.png"), panel)
        log(ids[j], "完成", "Wayback 欄", sum(c is not None for c in w))
        return ids[j]
    with ThreadPoolExecutor(3) as ex:
        list(ex.map(one, range(len(sel))))
    key = {ids[j]: {k: sel[j][k] for k in ("su_id", "su_uid", "stratum", "category", "eff_area_ha", "eff_slope", "explained", "new_ha_s2", "riv_frac", "ls_photos", "dis_photos", "quad", "lat", "lon")} for j in range(len(sel))}
    (OUT / "key.json").write_text(json.dumps(dict(sorted(key.items())), ensure_ascii=False, indent=1), encoding="utf-8")
    sid = sorted(key)
    rows = "".join(f'<div class="it" id="it_{i}"><b>{i}</b><a href="{i}.png" target="_blank"><img src="{i}.png" loading="lazy"></a><div class="r">'
                   + "".join(f'<label><input type="radio" name="v_{i}" value="{k}" onchange="sv()"><b>{k}</b> {t}</label> ' for k, t in OPT)
                   + f'<br><span>信心：</span>' + "".join(f'<label><input type="radio" name="c_{i}" value="{c}" onchange="sv()">{c}</label> ' for c in ("高", "中", "低"))
                   + f'　<label><input type="checkbox" id="s_{i}" onchange="sv()">可見小型崩塌或落石（&lt;0.5 ha）</label> <input id="n_{i}" placeholder="備註" oninput="sv()"></div></div>' for i in sid)
    html = f"""<!doctype html><meta charset="utf-8"><title>分層人工驗證判讀表</title><style>body{{font:15px/1.6 system-ui,'Noto Sans TC';margin:0;background:#f6f5f2}}header{{position:sticky;top:0;background:#222;color:#fff;padding:8px 16px;z-index:9}}header button{{margin-left:12px}}main{{max-width:1500px;margin:auto;padding:12px 16px}}.it{{background:#fff;border:1px solid #ddd;padding:8px;margin:0 0 16px}}img{{width:100%;display:block}}label{{margin-right:12px;white-space:nowrap}}input[id^=n_]{{width:28%}}.done{{border-left:6px solid #2a7}}</style>
<header><b>分層人工驗證判讀表（{len(sid)} 個坡面）</b><span id="p"></span><button onclick="exp()">匯出 JSON</button></header><main>
<p>每張圖是同一個坡面（白線＝≥15° 有效坡面輪廓）在四個時間點的影像：Wayback（0.3–0.5 m，較早與最新）、Sentinel-2（2024-04-04、2025-03-25；10 m，放大顯示）。請只依影像判斷，不要參考任何系統推薦。<b>注意：</b>白線是由 20 m DTM 導出的有效坡面，與 Sentinel-2／DTM 對位良好（POC 全區中位偏差約 1–3 m）；Wayback 影像本身對 DTM／Sentinel-2 常有 10–25 m 的偏移（查證見 offset_check.json），偏移明顯時 Wayback 欄會多一條黃線（依實測偏移平移後的輪廓，僅供對照）；最右欄是輪廓的原始參考（DTM 山陰影＋坡度分級，青線＝原始 Slope Unit）。請以輪廓的大致位置判斷，不要因邊界偏移而否定。選項：A 近期有新增／擴大崩塌；B 有崩塌但近期無變化；C 無崩塌；D 有裸露但非崩塌；E 無法判讀。</p>{rows}</main>
<script>const IDS={json.dumps(sid)};let S={{}};try{{S=JSON.parse(localStorage.getItem('sv')||'{{}}')}}catch(e){{}}
function sv(){{IDS.forEach(i=>{{const v=document.querySelector('input[name=v_'+i+']:checked'),c=document.querySelector('input[name=c_'+i+']:checked');S[i]={{verdict:v?v.value:null,conf:c?c.value:null,small:document.getElementById('s_'+i).checked,note:document.getElementById('n_'+i).value}}}});try{{localStorage.setItem('sv',JSON.stringify(S))}}catch(e){{}}prog()}}
function prog(){{let n=0;IDS.forEach(i=>{{const s=S[i],d=s&&s.verdict&&s.conf;document.getElementById('it_'+i).classList.toggle('done',!!d);if(d)n++}});document.getElementById('p').textContent='　'+n+' / '+IDS.length+' 已完成'}}
IDS.forEach(i=>{{const s=S[i];if(!s)return;if(s.verdict){{const e=document.querySelector('input[name=v_'+i+'][value='+s.verdict+']');if(e)e.checked=true}}if(s.conf){{const e=document.querySelector('input[name=c_'+i+'][value='+s.conf+']');if(e)e.checked=true}}document.getElementById('s_'+i).checked=!!s.small;document.getElementById('n_'+i).value=s.note||''}});prog();
function exp(){{const o={{}};IDS.forEach(i=>{{const s=S[i];if(s&&s.verdict&&s.conf)o[i]=s}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(o,null,1)],{{type:'application/json'}}));a.download='su_strat_human.json';a.click()}}</script>"""
    (OUT / "review.html").write_text(html, encoding="utf-8")
    log("完成", len(sid), "張 →", OUT)


if __name__ == "__main__":
    main()
