"""新區域（花蓮馬太鞍溪堰塞湖）Sentinel-2 事件前後變化偵測，對事件型目錄（樺加沙，2025-09-22）評估。
fetch：以 WMTS TRUE_COLOR（8 位元，z14 圖磚約 9.55 m/px）取範圍框內各日期，重採樣到 10 m TM2 分析格網（浮水印格視為無效）。
detect：同高雄 POC 規則（ExG<0.08 且 V≥70，3×3 開運算、移除 <0.1 ha；參數不重調）；新增裸露＝事後有、事前（膨脹 1 格）無，且坡度 ≥15°（DTM 20 m）。
      亮雲粗檢：V≥150 且飽和度 <0.25 的格比例，供判斷影像是否可用（不做遮罩）。
eval：事件多邊形（polygons.json 中 2025-09-22 樺加沙，含 30 m 緩衝）內／外新增機率與倍數、事件面積被偵測為新增的比例、
      事件平移 3 km／6 km 的倍數（空間相關對照）。
限制：8 位元顯示影像；單一事件；事件判釋為水保署衛星判釋、非獨立真值；堰塞湖水體與淹沒區的色彩變化可能被當作新增裸露；
      不同日期的雲量不同（見 bright_cloud_frac）。
用法：python scripts/s2_area.py fetch|detect
輸出：data/biggis_interp/areas/hualien_barrier_lake/s2/ 與 eval.json。
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import Polygon, box

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import sentinel_assist as S  # noqa: E402
import reverse_detect as R  # noqa: E402
from landslide_incremental import mask  # noqa: E402

NAME = "hualien_barrier_lake"
BBOX = (121.1456, 23.5494, 121.4456, 23.8494)
AREA = REPO / "data" / "biggis_interp" / "areas" / NAME
OUT = AREA / "s2"
DATES = ["20250325", "20250414", "20250514", "20250615", "20250822", "20250921", "20251001", "20251011", "20251016", "20251230"]
PRE, POST = ["20250325", "20250414", "20250615"], ["20251001", "20251011", "20251016", "20251230"]      # 事前（以範圍內實測亮雲比例挑選，s2_area_select.py）、事後
RES = 10.0
EVT_RANGE = ("2025-05-01", "2026-06-30")      # 讀入的事件日期範圍（其他區域由 area_run.py 改寫）
WINDOW = ("2025-05-14", "2025-10-16")        # 事前最晚～事後最早（含上限）之間的事件視為窗口事件
WORLD = 2 * math.pi * S.R_MERC
P = WORLD / (S.TILE_PX * 2 ** (S.TILE_MATRIX - 1))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def grid():
    x0, y0, x1, y1 = [float(v) for v in gpd.GeoSeries([box(*BBOX)], crs=4326).to_crs(3826).total_bounds]
    x0, y1 = math.floor(x0 / RES) * RES, math.ceil(y1 / RES) * RES
    w, h = int(math.ceil((x1 - x0) / RES)), int(math.ceil((y1 - y0) / RES))
    return from_origin(x0, y1, RES, RES), (h, w)


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    tf, (H, W) = grid()
    to_m = Transformer.from_crs(3826, 3857, always_xy=True)
    xs, ys = tf.c + (np.arange(W) + 0.5) * RES, tf.f - (np.arange(H) + 0.5) * RES
    X, Y = np.meshgrid(xs, ys)
    mx, my = to_m.transform(X.ravel(), Y.ravel())
    gx, gy = (np.asarray(mx) + WORLD / 2) / P, (WORLD / 2 - np.asarray(my)) / P
    ix, iy = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
    col, row = ix // S.TILE_PX, iy // S.TILE_PX
    px, py = (ix % S.TILE_PX).astype(np.int32), (iy % S.TILE_PX).astype(np.int32)
    tiles = sorted(set(zip(row.tolist(), col.tolist())))
    log("圖磚", len(tiles), "格網", H, W)
    iid = S.instance_id()
    wm = np.zeros((S.TILE_PX, S.TILE_PX), bool)
    wm[S.TILE_PX - S.WATERMARK_BOX_PX[1]:, :S.WATERMARK_BOX_PX[0]] = True
    key = row * 100000 + col
    order = np.argsort(key, kind="stable")
    ks = key[order]
    b = np.flatnonzero(np.diff(ks)) + 1
    starts, ends = np.concatenate([[0], b]), np.concatenate([b, [len(ks)]])
    for ymd in DATES:
        if (OUT / f"rgb_{ymd}.npy").exists():
            continue

        def get(rc):
            try:
                return rc, S._fetch_tile(iid, ymd, "TRUE_COLOR", rc[0], rc[1])
            except Exception as e:  # noqa: BLE001
                log("失敗", rc, repr(e)[:80])
                return rc, None
        with ThreadPoolExecutor(4) as ex:
            got = dict(ex.map(get, tiles))
        rgb = np.zeros((H * W, 3), np.uint8)
        valid = np.zeros(H * W, bool)
        for s, e in zip(starts, ends):
            k = int(ks[s])
            t = got.get((k // 100000, k % 100000))
            if t is None:
                continue
            idx = order[s:e]
            rgb[idx] = t[py[idx], px[idx]][:, ::-1]
            valid[idx] = ~wm[py[idx], px[idx]]
        np.save(OUT / f"rgb_{ymd}.npy", rgb.reshape(H, W, 3))
        np.save(OUT / f"valid_{ymd}.npy", valid.reshape(H, W))
        log(ymd, "完成；有效", round(float(valid.mean()), 3))


def feats(d):
    rgb = np.load(OUT / f"rgb_{d}.npy").astype(np.float32)
    v = np.load(OUT / f"valid_{d}.npy")
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    mx = rgb.max(axis=2)
    sat = (mx - rgb.min(axis=2)) / (mx + 1e-6)
    cloud = float(((mx >= 150) & (sat < 0.25) & v).sum() / max(v.sum(), 1))
    return exg, mx, v, cloud


def slope(tf, H, W):
    with rasterio.open(REPO / "data" / "dtm" / "tw_dtm20.tif") as r:
        x0, y1 = tf.c, tf.f
        win = rasterio.windows.from_bounds(x0 - 200, y1 - H * RES - 200, x0 + W * RES + 200, y1 + 200, r.transform).round_offsets().round_lengths()
        dem = r.read(1, window=win).astype(np.float32)
        wtf = r.window_transform(win)
        dem[dem == r.nodata] = np.nan
    gy_, gx_ = np.gradient(dem, 20.0)
    sl = np.degrees(np.arctan(np.hypot(gx_, gy_)))
    xs, ys = x0 + (np.arange(W) + 0.5) * RES, y1 - (np.arange(H) + 0.5) * RES
    cj = np.clip(((xs - wtf.c) / 20.0).astype(int), 0, sl.shape[1] - 1)
    ri = np.clip(((wtf.f - ys) / 20.0).astype(int), 0, sl.shape[0] - 1)
    return np.nan_to_num(sl[np.ix_(ri, cj)], nan=0.0)


def events():
    pj = json.load(open(REPO / "data" / "biggis_interp" / "polygons.json", encoding="utf-8"))
    by = {}
    for p in pj["polys"]:
        e = pj["events"][p[0]]
        if not (EVT_RANGE[0] <= e["date"] <= EVT_RANGE[1]):
            continue
        xy = p[2]
        by.setdefault((e["event"], e["date"]), []).append(Polygon(list(zip(xy[0::2], xy[1::2]))).buffer(0))
    return {k: gpd.GeoSeries(v, crs=4326).to_crs(3826) for k, v in by.items()}


def detect():
    tf, (H, W) = grid()
    shape = (H, W)
    F = {d: feats(d) for d in DATES}
    sl = slope(tf, H, W)
    steep = sl >= R.SLOPE_MIN
    ev = events()
    bb = gpd.GeoSeries([box(*BBOX)], crs=4326).to_crs(3826).iloc[0]
    ev = {k: g for k, g in ev.items() if g.intersects(bb).any()}
    log("窗口內事件", [(k[0].split("_", 1)[-1], k[1], len(v)) for k, v in ev.items()])

    def det(d):
        e, v, va, _ = F[d]
        return R.post((e < 0.08) & (v >= 70) & va)
    out = {"bbox": BBOX, "dates": DATES, "slope_ge15_km2": round(float(steep.sum() * 1e-4), 1), "valid_frac": {d: round(float(F[d][2].mean()), 3) for d in DATES},
           "bright_cloud_frac": {d: round(F[d][3], 3) for d in DATES}, "events": [{"event": k[0], "date": k[1], "n_polys": len(v)} for k, v in ev.items()], "pairs": {}}
    log(out["valid_frac"], out["bright_cloud_frac"])
    cache = {d: det(d) for d in DATES}
    pol = None      # 窗口內事件聯集（事前 2025-05-14 之後～事後 2025-10-16 之前：丹娜絲薇帕 0721、0728 豪雨、楊柳、樺加沙）
    sel = [g for k, g in ev.items() if WINDOW[0] < k[1] <= WINDOW[1]]
    if sel:
        pol = gpd.GeoSeries(pd.concat(sel).values, crs=3826)
    log("窗口事件多邊形", len(pol) if pol is not None else 0, "面積 ha", round(float(pol.area.sum() / 1e4), 1) if pol is not None else 0)
    U0 = ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.values, crs=3826), tf, shape), iterations=3) if pol is not None else np.zeros(shape, bool)
    shifted = {s: ndi.binary_dilation(mask(gpd.GeoDataFrame(geometry=pol.translate(xoff=s).values, crs=3826), tf, shape), iterations=3) for s in (3000.0, -3000.0, 6000.0)} if pol is not None else {}
    for a in PRE:
        for b in POST:
            region = F[a][2] & F[b][2] & steep
            new = R.post(cache[b] & ~ndi.binary_dilation(cache[a], iterations=1)) & region
            avail = region & ~cache[a]

            def ratio(U):
                pin = float(new[U & avail].sum() / max((U & avail).sum(), 1))
                pout = float(new[~U & avail].sum() / max((~U & avail).sum(), 1))
                return round(pin, 4), round(pout, 4), round(pin / pout, 2) if pout else None
            pin, pout, rt = ratio(U0)
            r = {"region_km2": round(float(region.sum() * 1e-4), 1), "new_ha": round(float(new.sum() * 0.01), 1), "new_ha_in_event": round(float((new & U0).sum() * 0.01), 1),
                 "share_new_in_event": round(float((new & U0).sum() / max(new.sum(), 1)), 3), "P(new|in)": pin, "P(new|out)": pout, "ratio": rt,
                 "ratio_shifted": {str(int(s)) + "m": ratio(U)[2] for s, U in shifted.items()},
                 "event_area_ha_in_region": round(float((U0 & region).sum() * 0.01), 1), "event_detected_as_new_frac": round(float((new & U0).sum() / max((U0 & region).sum(), 1)), 3),
                 "event_bare_post_frac": round(float((cache[b] & U0 & region).sum() / max((U0 & region).sum(), 1)), 3), "event_bare_pre_frac": round(float((cache[a] & U0 & region).sum() / max((U0 & region).sum(), 1)), 3)}
            out["pairs"][f"{a}->{b}"] = r
            np.save(OUT / f"new_{a}_{b}.npy", new)
            log(a, b, r)
    (OUT / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    {"fetch": fetch, "detect": detect}[sys.argv[1]]()
