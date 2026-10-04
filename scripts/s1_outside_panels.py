"""花蓮 S1「目錄外大塊」人工複核圖：降軌 VH、K=2 的 ≥2 ha 偵測塊中，與事件多邊形（含 30 m）重疊 <20% 者。
每塊一張圖（O01…）：S2 2025-06-15（事前）／2025-10-11（事後）／2025-10-16（事後）真彩（10 m，放大 4 倍），S1 dB 差（藍＝下降，紅＝上升）；
黃色輪廓＝S1 偵測塊（僅畫在第 2、4 欄）。另輸出 key.json（座標、面積、坡度、平均 dB 差）與人工複核表 index.html（判讀存瀏覽器、可匯出）。
用法：python scripts/s1_outside_panels.py
"""
import json
import sys
import warnings
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import s2_area as A  # noqa: E402
import s1_area as S1  # noqa: E402
import reverse_detect as R  # noqa: E402
from landslide_incremental import mask  # noqa: E402
from pyproj import Transformer  # noqa: E402

K = 2.0
KEY = "desc105"
MIN_CELLS = 200
HALF = 60                      # 裁切半徑（格，10 m）→ ±600 m
UP = 4
OUT = A.AREA / "s1_review"


def main():
    OUT.mkdir(exist_ok=True)
    tf, (H, W) = A.grid()
    slope = A.slope(tf, H, W)
    steep = slope >= R.SLOPE_MIN
    ev = {k: g for k, g in A.events().items() if g.intersects(gpd.GeoSeries([A.box(*A.BBOX)], crs=4326).to_crs(3826).iloc[0]).any()}
    pre, dpre = S1.stack(KEY, "pre", "vh")
    post, dpost = S1.stack(KEY, "post", "vh")
    lo, hi = max(dpre), min(dpost)
    sel = [g for k, g in ev.items() if lo < k[1] <= hi]
    pol = gpd.GeoSeries(pd.concat(sel).values, crs=3826)
    ev_d = ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.values, crs=3826), tf, (H, W)), iterations=3)
    with np.errstate(all="ignore"):
        d = 10 * np.log10(np.maximum(post, 1e-6)) - 10 * np.log10(np.maximum(pre, 1e-6))
    region = np.isfinite(d) & steep & (pre > 1e-4) & (post > 1e-4)
    med = float(np.median(d[region]))
    mad = float(np.median(np.abs(d[region] - med))) * 1.4826
    new = R.post((d < med - K * mad) & region) & region
    new = ndi.median_filter(new.astype(np.uint8), size=3).astype(bool) & region
    lab, n = ndi.label(ndi.binary_dilation(new, iterations=1), structure=np.ones((3, 3)))
    area = np.bincount(lab[new].ravel(), minlength=n + 1)
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    cands = []
    for i in np.flatnonzero(area >= MIN_CELLS):
        if i == 0:
            continue
        m = (lab == i) & new
        if (m & ev_d).sum() / max(m.sum(), 1) >= 0.2:
            continue
        rr, cc = np.nonzero(m)
        cy, cx = int(rr.mean()), int(cc.mean())
        lon, lat = to_ll.transform(tf.c + (cx + 0.5) * 10, tf.f - (cy + 0.5) * 10)
        cands.append({"m": m, "cy": cy, "cx": cx, "ha": float(m.sum() * 0.01), "lon": lon, "lat": lat, "slope": float(slope[m].mean()), "dB": float(d[m].mean())})
    cands.sort(key=lambda c: -c["ha"])
    imgs = {dd: np.load(A.OUT / f"rgb_{dd}.npy")[..., ::-1] for dd in ("20250615", "20251011", "20251016")}
    key = {}
    for j, c in enumerate(cands, 1):
        oid = f"O{j:02d}"
        y0, x0 = max(c["cy"] - HALF, 0), max(c["cx"] - HALF, 0)
        y1, x1 = min(y0 + 2 * HALF, H), min(x0 + 2 * HALF, W)
        cols = []
        cont, _ = cv2.findContours(c["m"][y0:y1, x0:x1].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for dd in ("20250615", "20251011", "20251016"):
            im = cv2.resize(imgs[dd][y0:y1, x0:x1].copy(), None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC)
            if dd == "20251011":
                cv2.polylines(im, [(cn * UP).astype(np.int32) for cn in cont], True, (0, 255, 255), 2)
            cv2.putText(im, dd, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cols.append(im)
        dd_img = np.clip((d[y0:y1, x0:x1] + 8) / 16, 0, 1)
        heat = cv2.applyColorMap((dd_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heat = cv2.resize(heat, None, fx=UP, fy=UP, interpolation=cv2.INTER_NEAREST)
        cv2.polylines(heat, [(cn * UP).astype(np.int32) for cn in cont], True, (255, 255, 255), 2)
        cv2.putText(heat, "S1 VH dB diff (blue=down)", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cols.append(heat)
        cv2.imwrite(str(OUT / f"{oid}.jpg"), np.hstack(cols), [cv2.IMWRITE_JPEG_QUALITY, 90])
        key[oid] = {"lon": round(c["lon"], 5), "lat": round(c["lat"], 5), "ha": round(c["ha"], 1), "mean_slope_deg": round(c["slope"]), "mean_dB_diff": round(c["dB"], 1)}
    (OUT / "key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    ids = list(key)
    opts = ["崩塌（新）", "崩塌（舊已存在）", "河床／河道變化", "人為（農地、道路、整地）", "雲影／山陰影／雷達疊掩", "其他", "無法判讀"]
    rows = "".join(f'<div class="it" id="it_{i}"><b>{i}</b>（{key[i]["ha"]} ha，坡度 {key[i]["mean_slope_deg"]}°，VH 平均 {key[i]["mean_dB_diff"]} dB，{key[i]["lat"]},{key[i]["lon"]}）<img src="{i}.jpg">'
                   + "".join(f'<label><input type="radio" name="v_{i}" value="{o}" onchange="sv()">{o}</label> ' for o in opts) + f'<input id="n_{i}" placeholder="備註" oninput="sv()"></div>' for i in ids)
    html = f"""<!doctype html><meta charset="utf-8"><title>花蓮 S1 目錄外大塊複核</title><style>body{{font:15px/1.6 system-ui;margin:16px;background:#f6f5f2}}.it{{background:#fff;border:1px solid #ddd;padding:8px;margin:0 0 14px}}img{{width:100%;display:block}}label{{margin-right:10px}}input[id^=n_]{{width:30%}}</style>
<h3>花蓮 S1「目錄外大塊」複核（{len(ids)} 塊）</h3><p>四欄：S2 2025-06-15（事前）、2025-10-11（事後，黃框＝S1 偵測塊）、2025-10-16（事後）、S1 VH dB 差（藍＝下降）。請依影像判讀這個黃框是否為崩塌。<button onclick="exp()">匯出 JSON</button></p>{rows}
<script>const IDS={json.dumps(ids)};let S={{}};try{{S=JSON.parse(localStorage.getItem('o')||'{{}}')}}catch(e){{}}
function sv(){{IDS.forEach(i=>{{const v=document.querySelector('input[name=v_'+i+']:checked');S[i]={{verdict:v?v.value:null,note:document.getElementById('n_'+i).value}}}});try{{localStorage.setItem('o',JSON.stringify(S))}}catch(e){{}}}}
IDS.forEach(i=>{{const s=S[i];if(!s)return;if(s.verdict){{const e=[...document.querySelectorAll('input[name=v_'+i+']')].find(x=>x.value===s.verdict);if(e)e.checked=true}}document.getElementById('n_'+i).value=s.note||''}});
function exp(){{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(S,null,1)],{{type:'application/json'}}));a.download='s1_outside_human.json';a.click()}}</script>"""
    (OUT / "index.html").write_text(html, encoding="utf-8")
    print(len(ids), "塊", json.dumps(key, ensure_ascii=False))


if __name__ == "__main__":
    main()
