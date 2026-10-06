"""查證有效坡面輪廓（由 20 m DTM 導出）與影像的座標偏差：
(1) 高雄 POC 全區：Sentinel-2（2025-03-25）灰階 vs DTM 山陰影的區塊位移（128 px、步距 128），報告中位數與四分位——輪廓與 S2 的絕對關係。
(2) 24 個分層坡面：Wayback 最新影像（降到 10 m）vs 同範圍 Sentinel-2（2025-03-25）的整幅位移（相位相關，梯度影像）——Wayback 相對 S2／DTM 的偏移。
輸出 poc/su_strat/offset_check.json。位移符號：影像（mov）相對參考（ref）；+x＝東、+y＝南；單位公尺。
用法：python scripts/su_outline_offset_check.py
"""
import json
import math
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import rasterio
from pyproj import Transformer

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import wayback_assist as WB  # noqa: E402
from landslide_incremental import grid  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
OUT = POC / "su_strat"


def hp(a, s1=1.5, s2=15):
    a = a.astype(np.float32)
    return cv2.GaussianBlur(a, (0, 0), s1) - cv2.GaussianBlur(a, (0, 0), s2)


def gm(a, s=1.5):
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), s)
    return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))


def dem10(tf, H, W):
    with rasterio.open(POC / "su_work" / "dem.tif") as r:
        dem = r.read(1).astype(np.float32)
        wtf = r.transform
        nod = r.nodata
    dem = np.where(dem == nod, np.nan, dem)
    dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    xs = tf.c + (np.arange(W) + 0.5) * 10
    ys = tf.f - (np.arange(H) + 0.5) * 10
    mx = np.tile(((xs - wtf.c) / 20 - 0.5).astype(np.float32), (H, 1))
    my = np.tile(((wtf.f - ys) / 20 - 0.5).astype(np.float32)[:, None], (1, W))
    D = cv2.remap(dem, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return cv2.GaussianBlur(D, (0, 0), 1.5)


def hillshade(D, az=143, el=51):
    gy, gx = np.gradient(D, 10.0)
    sl = np.arctan(np.hypot(gx, gy))
    asp = np.arctan2(-gx, gy)
    z, a = math.radians(90 - el), math.radians(az)
    return (math.cos(z) * np.cos(sl) + math.sin(z) * np.sin(sl) * np.cos(a - asp)).astype(np.float32)


def blocks(ref, mov, blk=128, stride=128, minresp=0.05):
    H, W = ref.shape
    win = cv2.createHanningWindow((blk, blk), cv2.CV_32F)
    out = []
    for y in range(0, H - blk + 1, stride):
        for x in range(0, W - blk + 1, stride):
            pa, pb = ref[y:y + blk, x:x + blk], mov[y:y + blk, x:x + blk]
            if pa.std() < 1e-6 or pb.std() < 1e-6:
                continue
            (dx, dy), r = cv2.phaseCorrelate(pa, pb, win)
            if r >= minresp and abs(dx) < 15 and abs(dy) < 15:
                out.append((dx * 10, dy * 10, r, y + blk // 2, x + blk // 2))
    return np.array(out).reshape(-1, 5)


def main():
    tf, (H, W) = grid()
    res = {}
    D = dem10(tf, H, W)
    s2 = np.load(POC / "s2" / "rgb_20250325.npy")
    v = np.load(POC / "s2" / "valid_20250325.npy")
    g = cv2.cvtColor(s2[..., ::-1].copy(), cv2.COLOR_BGR2GRAY).astype(np.float32)
    g[~v] = np.nanmedian(g)
    hs = hillshade(D)
    b = blocks(hp(hs), hp(g))
    res["S2_vs_DTM_hillshade_POC"] = {"n": int(len(b)), "median_dx_m": round(float(np.median(b[:, 0])), 1), "median_dy_m": round(float(np.median(b[:, 1])), 1),
                                      "p25_dx": round(float(np.percentile(b[:, 0], 25)), 1), "p75_dx": round(float(np.percentile(b[:, 0], 75)), 1),
                                      "p25_dy": round(float(np.percentile(b[:, 1], 25)), 1), "p75_dy": round(float(np.percentile(b[:, 1], 75)), 1), "median_resp": round(float(np.median(b[:, 2])), 2)}
    print(res["S2_vs_DTM_hillshade_POC"], flush=True)
    sel = json.load(open(OUT / "selected.json", encoding="utf-8"))
    key = json.load(open(OUT / "key.json", encoding="utf-8"))
    sid2sv = {v_["su_id"]: k for k, v_ in key.items()}
    to_ll = Transformer.from_crs(3826, 4326, always_xy=True)
    to_tm2 = Transformer.from_crs(3857, 3826, always_xy=True)
    rows = []
    for t in sel:
        lon, lat = to_ll.transform(t["cx"], t["cy"])
        try:
            fr = [f for f in WB.list_frames(lat, lon, 18) if f["date"] >= "20170101" and (f["res_m"] or 9) <= 0.6]
            f = fr[-1]
            img, bounds = WB.fetch_image(lat, lon, 380.0, f["release"], 18)
        except Exception as e:  # noqa: BLE001
            print("失敗", t["su_id"], repr(e)[:60])
            continue
        h, w = img.shape[:2]
        minx, miny, maxx, maxy = bounds
        # Wayback → 10 m（以 S2 同一地面像元大小重取樣）：目標 76×76，像元中心 → 3857 → TM2 → S2 格網取樣
        n = 76
        gx = minx + (np.arange(n) + 0.5) * (maxx - minx) / n
        gy = maxy - (np.arange(n) + 0.5) * (maxy - miny) / n
        GX, GY = np.meshgrid(gx, gy)
        X, Y = to_tm2.transform(GX.ravel(), GY.ravel())
        cmap = ((np.asarray(X) - tf.c) / 10 - 0.5).reshape(n, n).astype(np.float32)
        rmap = ((tf.f - np.asarray(Y)) / 10 - 0.5).reshape(n, n).astype(np.float32)
        s2c = cv2.remap(g, cmap, rmap, cv2.INTER_LINEAR)
        vc = cv2.remap(v.astype(np.float32), cmap, rmap, cv2.INTER_NEAREST)
        wg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        wg = cv2.resize(wg, (n, n), interpolation=cv2.INTER_AREA)
        if vc.mean() < 0.8:
            rows.append({"sv": sid2sv.get(t["su_id"]), "su_id": t["su_id"], "wayback_date": f["date"], "note": "S2 視窗浮水印遮蔽 >20%，略過"})
            continue
        win = cv2.createHanningWindow((n, n), cv2.CV_32F)
        (dx, dy), r = cv2.phaseCorrelate(gm(s2c), gm(wg), win)
        rows.append({"sv": sid2sv.get(t["su_id"]), "su_id": t["su_id"], "wayback_date": f["date"], "dx_m": round(dx * 10, 1), "dy_m": round(dy * 10, 1), "resp": round(float(r), 2)})
        print(rows[-1], flush=True)
    ok = [r for r in rows if "dx_m" in r and r["resp"] >= 0.1]
    if ok:
        res["Wayback_latest_vs_S2_per_SU"] = {"n": len(ok), "median_dx_m": round(float(np.median([r["dx_m"] for r in ok])), 1), "median_dy_m": round(float(np.median([r["dy_m"] for r in ok])), 1),
                                              "median_abs_m": round(float(np.median([math.hypot(r["dx_m"], r["dy_m"]) for r in ok])), 1), "p90_abs_m": round(float(np.percentile([math.hypot(r["dx_m"], r["dy_m"]) for r in ok], 90)), 1)}
    res["rows"] = rows
    (OUT / "offset_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print({k: v_ for k, v_ in res.items() if k != "rows"})


if __name__ == "__main__":
    main()
