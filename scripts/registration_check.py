"""事實查核：Wayback／衛星影像在山區是否「正射且前後期可對位」？
方法：不同日期影像以「梯度強度影像 + 相位相關（cv2.phaseCorrelate）」估計區塊位移（像素→公尺），報告位移分布並與坡度關聯。
  part wb_kh   高雄 POC 已快取的 Wayback 兩期（2022-08-26、2023-11-02）逐圖磚（z17，約 1.2 m/px），依坡度分組。
  part wb_hl   花蓮：在陡坡取樣點，列出該點 Wayback 各期（含 SRC_ACC 標稱精度、感測器），挑兩期取 z17 鑲嵌後估計位移。
  part s2      Sentinel-2（Copernicus WMTS，10 m）既有兩期陣列的區塊位移（高雄 2024-04-04 vs 2025-03-25；花蓮四期）。
解讀：位移只是「兩期相對錯位」的下限估計（植生／季節變化會使相位相關失敗或偏向 0，已用響應值 >0.1 的區塊）；
      > 約 2 個像素（Wayback 約 2–3 m、S2 約 20 m）的相對錯位就會在邊緣、小崩塌產生偽變化。
輸出：data/biggis_interp/areas/registration_check.json
用法：python scripts/registration_check.py wb_kh|wb_hl|s2
"""
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
sys.path.insert(0, str(REPO / "scripts"))
OUTJ = REPO / "data" / "biggis_interp" / "areas" / "registration_check.json"
POC = REPO / "data" / "biggis_interp" / "poc"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gradmag(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return np.hypot(gx, gy)


def block_shifts(a, b, blk=128, stride=64, min_resp=0.1):
    """a、b 同尺寸；回傳 [(cy, cx, dx, dy, resp)]（像素；b 相對 a 的位移）。"""
    ga, gb = gradmag(a), gradmag(b)
    win = cv2.createHanningWindow((blk, blk), cv2.CV_32F)
    out = []
    H, W = ga.shape
    for y in range(0, H - blk + 1, stride):
        for x in range(0, W - blk + 1, stride):
            pa, pb = ga[y:y + blk, x:x + blk], gb[y:y + blk, x:x + blk]
            if pa.std() < 1e-3 or pb.std() < 1e-3:
                continue
            (dx, dy), resp = cv2.phaseCorrelate(pa, pb, win)
            if resp >= min_resp:
                out.append((y + blk // 2, x + blk // 2, dx, dy, resp))
    return out


def summarize(shifts, m_per_px):
    if not shifts:
        return {"n": 0}
    d = np.array([math.hypot(s[2], s[3]) for s in shifts]) * m_per_px
    dx, dy = np.array([s[2] for s in shifts]) * m_per_px, np.array([s[3] for s in shifts]) * m_per_px
    return {"n": len(d), "median_m": round(float(np.median(d)), 2), "p90_m": round(float(np.percentile(d, 90)), 2), "mean_dx_m": round(float(dx.mean()), 2), "mean_dy_m": round(float(dy.mean()), 2),
            "frac_gt_2px": round(float((d > 2 * m_per_px).mean()), 3), "frac_gt_5px": round(float((d > 5 * m_per_px).mean()), 3)}


def save(key, val):
    d = json.load(open(OUTJ, encoding="utf-8")) if OUTJ.exists() else {}
    d[key] = val
    OUTJ.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")


def wb_kh():
    import rasterio
    from pyproj import Transformer
    rng = np.random.default_rng(20261004)
    d1, d2 = POC / "wb_tiles" / "2022-08-26", POC / "wb_tiles" / "2023-11-02"
    names = sorted(set(p.name for p in d1.glob("*.jpg")) & set(p.name for p in d2.glob("*.jpg")))
    names = list(rng.choice(names, min(2500, len(names)), replace=False))
    r = rasterio.open(POC / "su_work" / "dem.tif")
    dem = r.read(1).astype(np.float32)
    dem[dem < -100] = np.nan
    gy, gx = np.gradient(dem, 20.0)
    sl = np.degrees(np.arctan(np.hypot(gx, gy)))
    to = Transformer.from_crs(4326, 3826, always_xy=True)
    n = 2 ** 17
    rows = []
    for nm in names:
        tx, ty = [int(v) for v in nm[:-4].split("_")]
        lon = (tx + 0.5) / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
        x, y = to.transform(lon, lat)
        ci, ri = int((x - r.bounds.left) / 20), int((r.bounds.top - y) / 20)
        if not (0 <= ri < sl.shape[0] and 0 <= ci < sl.shape[1]) or np.isnan(sl[ri, ci]):
            continue
        a = cv2.imdecode(np.frombuffer((d1 / nm).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.frombuffer((d2 / nm).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if a is None or b is None:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(gradmag(a), gradmag(b), cv2.createHanningWindow((256, 256), cv2.CV_32F))
        rows.append((float(sl[ri, ci]), dx, dy, resp, lat))
    mpp = 156543.03392 * math.cos(math.radians(23.15)) / n
    A = np.array(rows)
    res = {"n_tiles": len(A), "m_per_px": round(mpp, 3)}
    for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 40), (40, 90)):
        m = (A[:, 0] >= lo) & (A[:, 0] < hi) & (A[:, 3] >= 0.1)
        res[f"slope_{lo}-{hi}"] = summarize([(0, 0, a[1], a[2], a[3]) for a in A[m]], mpp)
    res["all_conf"] = summarize([(0, 0, a[1], a[2], a[3]) for a in A[A[:, 3] >= 0.1]], mpp)
    res["frac_tiles_conf"] = round(float((A[:, 3] >= 0.1).mean()), 3)
    log(json.dumps(res, ensure_ascii=False))
    save("wayback_kaohsiung_2022-08-26_vs_2023-11-02", res)


def point_info(release, lat, lon):
    import wayback_assist as W
    base = release["meta_url"]
    for layer in range(5, 9):
        q = (f"{base}/{layer}/query?f=json&geometry={lon:.6f},{lat:.6f}&geometryType=esriGeometryPoint&inSR=4326&spatialRel=esriSpatialRelIntersects"
             "&outFields=SRC_DATE2,SRC_RES,SRC_ACC,SRC_DESC,NICE_NAME&returnGeometry=false")
        try:
            fs = W._get_json(q, 20, retries=1).get("features") or []
        except Exception:  # noqa: BLE001
            continue
        for f in fs:
            a = f["attributes"]
            if a.get("SRC_DATE2"):
                return {"acc_m": a.get("SRC_ACC"), "sensor": a.get("SRC_DESC"), "res": a.get("SRC_RES"), "name": a.get("NICE_NAME")}
    return {}


def wb_hl():
    import wayback_assist as W
    import s2_area as A
    tf, (H, Wd) = A.grid()
    sl = A.slope(tf, H, Wd)
    ok = np.load(A.OUT / "valid_20251016.npy") & (sl >= 30)
    rng = np.random.default_rng(20261004)
    rr, cc = np.nonzero(ok)
    pick = rng.choice(len(rr), 400, replace=False)
    from pyproj import Transformer
    to = Transformer.from_crs(3826, 4326, always_xy=True)
    pts = []
    for i in pick:
        x, y = tf.c + (cc[i] + .5) * 10, tf.f - (rr[i] + .5) * 10
        lon, lat = to.transform(x, y)
        pts.append((lat, lon, float(sl[rr[i], cc[i]])))
    res = {"points": []}
    used = 0
    for lat, lon, s in pts:
        if used >= 5:
            break
        try:
            fr = [f for f in W.list_frames(lat, lon) if f["date"] >= "20170101" and (f["res_m"] or 9) <= 0.6]
        except Exception as e:  # noqa: BLE001
            log("list_frames 失敗", repr(e)[:60])
            continue
        if len(fr) < 2:
            continue
        rels = {r["num"]: r for r in W.releases()}
        for f in fr:
            f.update(point_info(rels[f["release"]], lat, lon))
        log(f"點 {lat:.4f},{lon:.4f} 坡度 {s:.0f}°：", [(f["date"], f["res_m"], f.get("acc_m"), f.get("sensor")) for f in fr])
        # 取日期相差最大的兩期與最近的兩期（各一對）
        pairs = [(fr[0], fr[-1])] + ([(fr[-2], fr[-1])] if len(fr) > 2 else [])
        entry = {"lat": lat, "lon": lon, "slope_deg": round(s), "frames": [{k: f.get(k) for k in ("date", "res_m", "acc_m", "sensor", "name")} for f in fr], "pairs": []}
        for a, b in pairs:
            try:
                ia, _ = W.fetch_image(lat, lon, 450, a["release"])
                ib, _ = W.fetch_image(lat, lon, 450, b["release"])
            except Exception as e:  # noqa: BLE001
                log("取像失敗", repr(e)[:60])
                continue
            mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** 17
            sh = block_shifts(ia, ib, blk=128, stride=64)
            (gdx, gdy), gresp = cv2.phaseCorrelate(gradmag(ia), gradmag(ib), cv2.createHanningWindow(ia.shape[1::-1], cv2.CV_32F))
            entry["pairs"].append({"a": a["date"], "b": b["date"], "global_shift_m": round(math.hypot(gdx, gdy) * mpp, 2), "global_resp": round(float(gresp), 3), **summarize(sh, mpp)})
            log("  ", a["date"], b["date"], entry["pairs"][-1])
        res["points"].append(entry)
        used += 1
    save("wayback_hualien_steep_points", res)


def s2():
    import s2_area as A
    import reverse_detect as R
    from landslide_incremental import grid as kgrid
    out = {}
    # 花蓮
    tf, (H, W) = A.grid()
    sl = A.slope(tf, H, W)
    dates = A.DATES
    imgs = {d: np.load(A.OUT / f"rgb_{d}.npy") for d in dates}
    vs = {d: np.load(A.OUT / f"valid_{d}.npy") for d in dates}
    rows = []
    for a, b in (("20250514", "20251016"), ("20250822", "20251016"), ("20250921", "20251001"), ("20251001", "20251016")):
        va = vs[a] & vs[b]
        ga, gb = imgs[a][..., ::-1].copy(), imgs[b][..., ::-1].copy()
        sh = block_shifts(ga, gb, blk=64, stride=64)
        sh = [s for s in sh if va[max(s[0] - 32, 0):s[0] + 32, max(s[1] - 32, 0):s[1] + 32].mean() > 0.9]
        by = {}
        for lo, hi in ((0, 15), (15, 30), (30, 90)):
            sel = [s for s in sh if lo <= sl[s[0], s[1]] < hi]
            by[f"slope_{lo}-{hi}"] = summarize(sel, 10.0)
        out[f"hualien_{a}_{b}"] = {"all": summarize(sh, 10.0), **by}
        log("花蓮", a, b, out[f"hualien_{a}_{b}"]["all"])
    # 高雄
    tf, (Hk, Wk) = kgrid()
    sk = R.slope10(tf, Hk, Wk)
    ra, rb = np.load(POC / "s2" / "rgb_20240404.npy")[..., ::-1].copy(), np.load(POC / "s2" / "rgb_20250325.npy")[..., ::-1].copy()
    va = np.load(POC / "s2" / "valid_20240404.npy") & np.load(POC / "s2" / "valid_20250325.npy")
    sh = block_shifts(ra, rb, blk=64, stride=64)
    sh = [s for s in sh if va[max(s[0] - 32, 0):s[0] + 32, max(s[1] - 32, 0):s[1] + 32].mean() > 0.9]
    by = {}
    for lo, hi in ((0, 15), (15, 30), (30, 90)):
        by[f"slope_{lo}-{hi}"] = summarize([s for s in sh if lo <= sk[s[0], s[1]] < hi], 10.0)
    out["kaohsiung_20240404_20250325"] = {"all": summarize(sh, 10.0), **by}
    log("高雄", out["kaohsiung_20240404_20250325"]["all"])
    save("sentinel2", out)


if __name__ == "__main__":
    {"wb_kh": wb_kh, "wb_hl": wb_hl, "s2": s2}[sys.argv[1]]()
