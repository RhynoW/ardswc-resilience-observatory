"""Sentinel-1 RTC（地形輻射校正，10 m，Microsoft Planetary Computer，免帳號）事件前後變化偵測，對事件型崩塌目錄評估。
區域設定來自 s2_area.py 的模組全域（NAME、BBOX、AREA…），其他區域由 area_run_s1.py 改寫；預設為花蓮馬太鞍溪。
SAR 不受雲影響、RTC 已做地形校正；只比「同一軌道方向、同一相對軌道」的前後多景平均，避免幾何差異。
fetch：STAC 查詢 → 取各景 VH／VV（線性 γ⁰）→ 以 pyproj 座標轉換＋cv2.remap 取樣到 10 m TM2 分析格網（避開 rasterio 與 pyproj 的 PROJ 資料庫衝突）→ 快取 <AREA>/s1/。
detect：事前 PRE 景平均、事後 POST 景平均（線性功率平均後轉 dB）；差 d = 事後 dB − 事前 dB（崩塌通常使後向散射下降）；
      新增候選 = d < median(d) − K×MAD（穩健；K=4 為主，另報告 K=2,3,4 的敏感度；不用事件標籤選 K），3×3 開運算、移除 <0.1 ha；範圍＝兩期有效且坡度 ≥15°。
      評估：事件窗口（最晚事前景～最早事後景之間）事件聯集內／外新增機率與倍數、事件平移對照、事件被偵測比例、新增面積落在事件內的比例。
限制：陡坡有疊掩／陰影；植生含水與降雨使後向散射起伏（多景平均與穩健門檻緩解）；水體、道路、河床也會造成變化；單一事件對；
      事件判釋為水保署衛星判釋、非獨立真值；偵測偏保守（高特異、低召回），事件被偵測比例只反映門檻保守程度。
用法：python scripts/s1_area.py fetch|detect（花蓮）；其他區域見 area_run_s1.py
"""
import json
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage as ndi

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import s2_area as A  # noqa: E402
import reverse_detect as R  # noqa: E402
from landslide_incremental import mask  # noqa: E402

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
PRE_WIN = ("2025-05-15", "2025-07-05")
POST_WIN = ("2025-10-01", "2025-10-16")
ORBITS = {"desc105": ("descending", 105), "asc69": ("ascending", 69)}      # 其他區域由 discover() 決定
UTM_EPSG = "32651"
KS = (2.0, 3.0, 4.0)
K_MAIN = 4.0
UA = {"User-Agent": "Mozilla/5.0", "Content-Type": "application/json"}


def out_dir():
    return A.AREA / "s1"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jget(u, data=None):
    return json.loads(urllib.request.urlopen(urllib.request.Request(u, data=data, headers=UA), timeout=90).read())


def center():
    return (A.BBOX[0] + A.BBOX[2]) / 2, (A.BBOX[1] + A.BBOX[3]) / 2


def search(win, lon, lat):
    body = json.dumps({"collections": ["sentinel-1-rtc"], "bbox": [lon, lat, lon, lat], "datetime": f"{win[0]}/{win[1]}", "limit": 100}).encode()
    return jget(STAC + "/search", body)["features"]


def discover(min_pre=2, min_post=1):
    """找同時有事前與事後景的軌道（方向、相對軌道），依景數排序。"""
    lon, lat = center()
    pre, post = {}, {}
    for win, dct in ((PRE_WIN, pre), (POST_WIN, post)):
        for f in search(win, lon, lat):
            p = f["properties"]
            dct.setdefault((p.get("sat:orbit_state"), p.get("sat:relative_orbit")), []).append(f)
    out = {}
    for k in pre:
        if k in post and len(pre[k]) >= min_pre and len(post[k]) >= min_post:
            out[f"{'desc' if k[0] == 'descending' else 'asc'}{k[1]}"] = k
    return dict(sorted(out.items(), key=lambda kv: -(len(pre[kv[1]]) + len(post[kv[1]]))))


def fetch():
    """避開 rasterio 與 pyproj 的 PROJ 資料庫衝突：座標轉換全用 pyproj，rasterio 只做 COG 視窗讀取，cv2.remap 取樣。"""
    import cv2
    from pyproj import Transformer
    OUT = out_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    tf, (H, W) = A.grid()
    lon, lat = center()
    tok = jget(SAS)["token"]
    xs, ys = tf.c + (np.arange(W) + 0.5) * 10.0, tf.f - (np.arange(H) + 0.5) * 10.0
    X, Y = np.meshgrid(xs, ys)
    ux, uy = Transformer.from_crs(3826, int(UTM_EPSG), always_xy=True).transform(X.ravel(), Y.ravel())
    ux, uy = np.asarray(ux).reshape(H, W), np.asarray(uy).reshape(H, W)
    for key, (od, ro) in ORBITS.items():
        for tag, win in (("pre", PRE_WIN), ("post", POST_WIN)):
            items = [f for f in search(win, lon, lat) if f["properties"].get("sat:orbit_state") == od and f["properties"].get("sat:relative_orbit") == ro]
            items.sort(key=lambda f: f["properties"]["datetime"])
            items = items[-4:] if tag == "pre" else items[:4]      # 事前取最後 4 景、事後取最前 4 景（控制下載量）
            log(key, tag, [f["properties"]["datetime"][:10] for f in items])
            for f in items:
                d = f["properties"]["datetime"][:10]
                if str(f["properties"].get("proj:code") or f["properties"].get("proj:epsg")) not in (f"EPSG:{UTM_EPSG}", UTM_EPSG):
                    log("跳過（非預期 UTM 帶）", d, f["properties"].get("proj:code"), f["properties"].get("proj:epsg"))
                    continue
                for pol in ("vh", "vv"):
                    p = OUT / f"{key}_{d}_{pol}.npy"
                    if p.exists():
                        continue
                    href = f["assets"][pol]["href"] + "?" + tok
                    try:
                        with rasterio.open(href) as src:
                            col, row = (~src.transform) * (ux, uy)
                            c0, c1 = max(int(np.floor(col.min())) - 2, 0), min(int(np.ceil(col.max())) + 3, src.width)
                            r0, r1 = max(int(np.floor(row.min())) - 2, 0), min(int(np.ceil(row.max())) + 3, src.height)
                            if c1 <= c0 or r1 <= r0:
                                log("不涵蓋", d, pol)
                                continue
                            arr = src.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
                            nod = src.nodata
                        if nod is not None:
                            arr[arr == nod] = np.nan
                        dst = cv2.remap(arr, (col - c0 - 0.5).astype(np.float32), (row - r0 - 0.5).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                    except Exception as e:  # noqa: BLE001
                        log("失敗", key, d, pol, repr(e)[:100])
                        continue
                    np.save(p, dst)
                    log(key, d, pol, "valid", round(float(np.isfinite(dst).mean()), 3))


def stack(key, tag, pol="vh"):
    fs = sorted(out_dir().glob(f"{key}_*_{pol}.npy"))
    win = PRE_WIN if tag == "pre" else POST_WIN
    fs = [f for f in fs if win[0] <= f.stem.split("_")[1] <= win[1]]
    fs = fs[-4:] if tag == "pre" else fs[:4]
    arrs = [np.load(f) for f in fs]
    if not arrs:
        return None, []
    with np.errstate(all="ignore"):
        m = np.nanmean(np.stack(arrs), axis=0)
    return m, [f.stem.split("_")[1] for f in fs]


def detect():
    tf, (H, W) = A.grid()
    shape = (H, W)
    steep = A.slope(tf, H, W) >= R.SLOPE_MIN
    ev = A.events()
    bb = gpd.GeoSeries([A.box(*A.BBOX)], crs=4326).to_crs(3826).iloc[0]
    ev = {k: g for k, g in ev.items() if g.intersects(bb).any()}
    out = {"K_main": K_MAIN, "pre_win": PRE_WIN, "post_win": POST_WIN, "runs": {}}
    for key in ORBITS:
        for polz in ("vh", "vv"):
            pre, dpre = stack(key, "pre", polz)
            post, dpost = stack(key, "post", polz)
            if pre is None or post is None:
                continue
            lo, hi = max(dpre), min(dpost)
            sel = [g for k, g in ev.items() if lo < k[1] <= hi]
            if not sel:
                continue
            pol = gpd.GeoSeries(pd.concat(sel).values, crs=3826)
            U0 = ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.values, crs=3826), tf, shape), iterations=3)
            shifted = {s: ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.translate(xoff=s).values, crs=3826), tf, shape), iterations=3) for s in (3000.0, -3000.0, 6000.0)}
            with np.errstate(all="ignore"):
                d = 10 * np.log10(np.maximum(post, 1e-6)) - 10 * np.log10(np.maximum(pre, 1e-6))
            region = np.isfinite(d) & steep & (pre > 1e-4) & (post > 1e-4)
            med = float(np.median(d[region]))
            mad = float(np.median(np.abs(d[region] - med))) * 1.4826
            r = {"pre_dates": dpre, "post_dates": dpost, "window_events": [f"{k[1]} {k[0].split('_', 1)[-1][:14]}" for k, g in ev.items() if lo < k[1] <= hi],
                 "region_km2": round(float(region.sum() * 1e-4), 1), "d_median_dB": round(med, 2), "d_mad_dB": round(mad, 2),
                 "event_area_ha_in_region": round(float((U0 & region).sum() * 0.01), 1), "by_K": {}}
            for K in KS:
                thr = med - K * mad
                new = R.post((d < thr) & region) & region
                new = ndi.median_filter(new.astype(np.uint8), size=3).astype(bool) & region

                def ratio(U):
                    pin = float(new[U & region].sum() / max((U & region).sum(), 1))
                    pout = float(new[~U & region].sum() / max((~U & region).sum(), 1))
                    return round(pin, 4), round(pout, 4), round(pin / pout, 2) if pout else None
                pin, pout, rt = ratio(U0)
                r["by_K"][str(K)] = {"threshold_dB": round(thr, 2), "new_ha": round(float(new.sum() * 0.01), 1), "new_ha_in_event": round(float((new & U0).sum() * 0.01), 1),
                                     "share_new_in_event": round(float((new & U0).sum() / max(new.sum(), 1)), 3), "P(new|in)": pin, "P(new|out)": pout, "ratio": rt,
                                     "ratio_shifted": {str(int(s)) + "m": ratio(U)[2] for s, U in shifted.items()}, "event_detected_frac": round(float((new & U0).sum() / max((U0 & region).sum(), 1)), 3)}
                if K == K_MAIN:
                    np.save(out_dir() / f"new_{key}_{polz}.npy", new)
            out["runs"][f"{key}_{polz}"] = r
            log(key, polz, json.dumps({k: v for k, v in r.items() if k != "by_K"}, ensure_ascii=False), {k: (v["new_ha"], v["share_new_in_event"], v["ratio"], v["event_detected_frac"]) for k, v in r["by_K"].items()})
    (out_dir() / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"fetch": fetch, "detect": detect}[sys.argv[1]]()
