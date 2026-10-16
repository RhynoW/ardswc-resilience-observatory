"""其他 POC 區域的坡面單元（v2 集水區重疊法）與 Sentinel-2 NDRE 時間序列（方法同高雄：slope_units_v2.py、s2_ndre_series.py）。
目前設定：H 花蓮（馬太鞍溪堰塞湖）、A 太魯閣（2024-04-03 花蓮地震）。
  - 坡面：內政部 20 m DTM 裁區域＋3 km 緩衝，正／反 DEM 子集水區疊合（河網門檻 20 ha）、<2 ha 併入鄰居；有效坡面＝3×3 平均坡度 ≥15°、去 <0.2 ha 碎塊。
  - NDRE：Planetary Computer Sentinel-2 L2A，SCL 去雲、有效坡面中位數（有效像元 ≥50%），每日扣全區中位數（距平）；
    持續下降＝下降日本身低於基準 −0.08、其後 6 筆至少 3 筆低於、之後中位數與 60 天後中位數都低於。
  - 花蓮的基準期改為 2024-07-01～2025-06-30（2024-04-03 花蓮地震之後、2025-07 颱風之前的一整年）；觀測到 2026-09-30。
  - 對照：hualien.json 的事件目錄圖層（2025-07 丹娜絲薇帕、0728 豪雨；含 688 ha 超大崩塌）落在坡面內的面積。
用法：python scripts/region_su_ndre.py H su|fetch|build|all
輸出：data/biggis_interp/areas/<name>/su_ndre/（labels.npz、units.json、items/、su_ndre.json、summary.json）
限制：與高雄相同（門檻看過高雄資料後訂、未獨立驗證；高雄人工盲判屬實約 2 成）。
"""
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
REG = {"H": {"name": "hualien", "bbox": (121.1456, 23.5494, 121.4456, 23.8494), "win": ("2024-07-01", "2026-09-30"), "base": ("2024-07-01", "2025-06-30")},
       # 太魯閣：2024-04-03 花蓮地震；基準期為地震前一整年，觀測到 2025-03（避開 2025 年颱風）
       "A": {"name": "taroko", "bbox": (121.45, 24.05, 121.75, 24.35), "win": ("2023-04-01", "2025-03-31"), "base": ("2023-04-01", "2024-03-31")}}
BUF_M = 3000.0
THR, MIN_VALID, LATE_DAYS = 0.08, 0.5, 60
SCL_OK = (4, 5, 6, 7)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def cfg(rid):
    c = dict(REG[rid])
    c["out"] = REPO / "data" / "biggis_interp" / "areas" / c["name"] / "su_ndre"
    c["out"].mkdir(parents=True, exist_ok=True)
    return c


def tm2_bbox(bbox):
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 3826, always_xy=True)
    xs, ys = t.transform([bbox[0], bbox[2], bbox[0], bbox[2]], [bbox[1], bbox[1], bbox[3], bbox[3]])
    return min(xs), min(ys), max(xs), max(ys)


def su(c):
    import rasterio
    from rasterio.windows import from_bounds
    import slope_units_v2 as SU
    x0, y0, x1, y1 = c["tm2"] = tm2_bbox(c["bbox"])
    work = c["out"] / "work"
    work.mkdir(exist_ok=True)
    dem_tif = work / "dem.tif"
    if not dem_tif.exists():
        with rasterio.open(REPO / "data" / "dtm" / "tw_dtm20.tif") as src:
            win = from_bounds(x0 - BUF_M, y0 - BUF_M, x1 + BUF_M, y1 + BUF_M, src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=win)
            tf = src.window_transform(win)
            nod = src.nodata
        arr = np.where(arr == nod, -9999.0, arr).astype("float32")
        import slope_units as V1
        prof = dict(driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype="float32", transform=tf, nodata=-9999.0, crs=V1.TM2_WKT)
        with rasterio.open(dem_tif, "w", **prof) as d:
            d.write(arr, 1)
    SU.WORK, SU.DEM_V1 = work, dem_tif
    dem, valid, tf, lab = SU.build(SU.STREAM_HA, SU.MIN_SU_HA)
    key_eff, slope = SU.effective(dem, valid, lab)
    n = int(lab.max()) + 1
    rr, cc = np.nonzero(lab >= 0)
    cnt = np.bincount(lab[rr, cc], minlength=n)
    cx = np.bincount(lab[rr, cc], weights=tf.c + (cc + 0.5) * 20, minlength=n) / np.maximum(cnt, 1)
    cy = np.bincount(lab[rr, cc], weights=tf.f - (rr + 0.5) * 20, minlength=n) / np.maximum(cnt, 1)
    eff = np.bincount(key_eff[key_eff >= 0], minlength=n) * 0.04
    order = np.lexsort((dem[rr, cc], lab[rr, cc]))
    ls = lab[rr, cc][order]
    first = np.r_[True, ls[1:] != ls[:-1]]
    low = {int(l_): (int(r_), int(c_)) for l_, r_, c_ in zip(ls[first], rr[order][first], cc[order][first])}
    ver = f"SUv2-D20m-WO-S{SU.STREAM_HA:g}-M{SU.MIN_SU_HA:g}-E15s3-F{SU.EFF_MIN_HA:g}"
    units = []
    for k in range(n):
        if not (x0 <= cx[k] <= x1 and y0 <= cy[k] <= y1) or eff[k] <= 0:
            continue
        r0, c0 = low[k]
        sid = f"SU2-{int((tf.c + (c0 + 0.5) * 20) // 20)}-{int((tf.f - (r0 + 0.5) * 20) // 20)}"
        units.append({"su_id": sid, "su_uid": f"{sid}@{ver}", "key": k, "area_ha": round(float(cnt[k] * 0.04), 2), "eff_ha": round(float(eff[k]), 2),
                      "cx": round(float(cx[k]), 1), "cy": round(float(cy[k]), 1)})
    np.savez_compressed(c["out"] / "labels.npz", key20=lab.astype(np.int32), key_eff=key_eff.astype(np.int32), x0=tf.c, y0=tf.f, res=20.0)
    (c["out"] / "units.json").write_text(json.dumps({"su_version": ver, "region": c["name"], "units": units}, ensure_ascii=False), encoding="utf-8")
    log("坡面（區域內、有有效坡面）", len(units), "全部", n, SU.stats(lab))


def sub_labels(c):
    L = np.load(c["out"] / "labels.npz")
    lab = L["key_eff"]
    keys = [u["key"] for u in json.load(open(c["out"] / "units.json", encoding="utf-8"))["units"]]
    m = np.isin(lab, keys)
    rr, cc = np.nonzero(m)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    return np.where(m[r0:r1, c0:c1], lab[r0:r1, c0:c1], -1), float(L["x0"]) + c0 * 20.0, float(L["y0"]) - r0 * 20.0, int(lab.max()) + 1


def fetch(c):
    import s2_ndre_series as NS
    sub, x0, y0, n = sub_labels(c)
    # 借用高雄版的 fetch：把標籤、輸出位置、時間窗換成本區域
    NS.labels = lambda: (sub, x0, y0, n)
    NS.ITEMS = c["out"] / "items"
    NS.WIN = c["win"]
    NS.fetch()


def build(c):
    import datetime as _dt
    sub, x0, y0, n = sub_labels(c)
    units = json.load(open(c["out"] / "units.json", encoding="utf-8"))["units"]
    by_date = {}
    for f in sorted((c["out"] / "items").glob("*.npz")):
        d = f.stem.split("_")[0]
        z = np.load(f)
        v, nd = z["valid"].astype(np.float32), z["ndre"].astype(np.float32)
        if d in by_date:
            pv, pn = by_date[d]
            take = v > pv
            by_date[d] = (np.where(take, v, pv), np.where(take, nd, pn))
        else:
            by_date[d] = (v, nd)
    dates = sorted(by_date)
    V = np.stack([by_date[d][0] for d in dates])
    N = np.stack([by_date[d][1] for d in dates])
    N[V < MIN_VALID] = np.nan
    reg = np.array([np.nanmedian(row) if np.isfinite(row).sum() >= 300 else np.nan for row in N])
    N = N - reg[:, None]
    N[~np.isfinite(reg)] = np.nan
    B = c["base"]
    out, stats = {}, Counter()
    for u in units:
        s = N[:, u["key"]]
        okd = [(d, float(x)) for d, x in zip(dates, s) if np.isfinite(x)]
        base = [x for d, x in okd if B[0] <= d <= B[1]]
        after = [(d, x) for d, x in okd if d > B[1]]
        rec = {"n_obs": len(okd), "n_base": len(base), "n_after": len(after), "drop": None}
        if len(base) >= 3 and len(after) >= 6:
            b = float(np.median(base))
            rec["base"] = round(b, 3)
            for k in range(len(after)):
                nxt = [x for _, x in after[k:k + 6]]
                rest = [x for _, x in after[k:]]
                late = [x for d, x in after[k:] if (_dt.date.fromisoformat(d) - _dt.date.fromisoformat(after[k][0])).days >= LATE_DAYS]
                if (after[k][1] < b - THR and len(nxt) >= 6 and sum(x < b - THR for x in nxt) >= 3 and np.median(rest) < b - THR
                        and len(late) >= 3 and np.median(late) < b - THR):
                    rec["drop"] = {"date": after[k][0], "delta": round(float(np.median(rest) - b), 3)}
                    break
            stats["drop" if rec["drop"] else "no_drop"] += 1
        else:
            stats["insufficient"] += 1
        out[u["su_id"]] = rec
    # 事件目錄（hualien.json 的 events 圖層）落在坡面有效範圍內的面積
    ev_ha = event_area(c, units)
    drops = {s: r["drop"] for s, r in out.items() if r["drop"]}
    with_ev = {s for s, a in ev_ha.items() if a >= 1.0}
    summ = {"region": c["name"], "base": B, "window": [dates[0], dates[-1]], "n_dates": len(dates), "n_su": len(units), "stats": dict(stats),
            "drop_months": dict(sorted(Counter(d["date"][:7] for d in drops.values()).items())),
            "su_with_event_ge1ha": len(with_ev), "drops_with_event": len(set(drops) & with_ev),
            "event_su_detected": round(len(set(drops) & with_ev) / max(len(with_ev), 1), 2),
            "drop_share_in_event_su": round(len(set(drops) & with_ev) / max(len(drops), 1), 2),
            "base_rate_event_su": round(len(with_ev) / max(len(units), 1), 3)}
    (c["out"] / "su_ndre.json").write_text(json.dumps({"su": out, "event_ha": ev_ha}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (c["out"] / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, indent=1))


def event_area(c, units):
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    L = np.load(c["out"] / "labels.npz")
    lab = L["key_eff"]
    tf = from_origin(float(L["x0"]), float(L["y0"]), 20.0, 20.0)
    j = json.load(open(REPO / "webapp" / "change_detect_viewer" / "static" / "poc" / f"{c['name']}.json", encoding="utf-8"))
    fc = j["layers"]["events"]
    geoms = [shape(f["geometry"]) for f in fc["features"]] if isinstance(fc, dict) else [shape(f["geometry"]) for f in fc]
    g = gpd.GeoSeries(geoms, crs=4326).to_crs(3826)
    em = features.rasterize(((x, 1) for x in g if not x.is_empty), out_shape=lab.shape, transform=tf, fill=0, dtype="uint8").astype(bool)
    n = int(lab.max()) + 1
    a = np.bincount(lab[(lab >= 0) & em], minlength=n) * 0.04
    return {u["su_id"]: round(float(a[u["key"]]), 2) for u in units}


if __name__ == "__main__":
    rid, step = sys.argv[1], sys.argv[2]
    c = cfg(rid)
    c["tm2"] = tm2_bbox(c["bbox"])
    for name, fn in (("su", su), ("fetch", fetch), ("build", build)):
        if step in (name, "all"):
            fn(c)
