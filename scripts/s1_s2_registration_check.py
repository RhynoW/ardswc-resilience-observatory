"""查證 Sentinel-1 RTC 與 Sentinel-2 WMTS 在分析格網上是否有座標偏差：
區塊相位相關（128 px = 1.28 km，梯度強度影像）估計相對位移（公尺）：
  S1 降軌 vs S1 升軌（同模態、視線方向相反：若是 DEM 高度誤差造成，位移方向會相反）
  S1（降／升）vs S2（跨模態，只用地形邊緣：河道、稜線、崩塌疤）
  S2 事前 vs S2 事後（基準：同模態同來源）
  S1 前後（降軌）：同軌道前後
輸出各組位移的中位數、平均 dx／dy、依坡度分組。位移符號：mov 相對 ref，+x＝東、+y＝南（影像座標向下）。
用法：python scripts/s1_s2_registration_check.py
"""
import json
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
import s2_area as A  # noqa: E402
import s1_area as S1  # noqa: E402


def gm(img, sig=2.0):
    a = img.astype(np.float32)
    a = np.where(np.isfinite(a), a, np.nanmedian(a))
    a = cv2.GaussianBlur(a, (0, 0), sig)
    return np.hypot(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))


def shifts(ref, mov, blk=128, stride=128, minresp=0.05):
    ga, gb = gm(ref), gm(mov)
    win = cv2.createHanningWindow((blk, blk), cv2.CV_32F)
    out = []
    H, W = ga.shape
    for y in range(0, H - blk + 1, stride):
        for x in range(0, W - blk + 1, stride):
            pa, pb = ga[y:y + blk, x:x + blk], gb[y:y + blk, x:x + blk]
            if pa.std() < 1e-3 or pb.std() < 1e-3:
                continue
            (dx, dy), r = cv2.phaseCorrelate(pa, pb, win)
            if r >= minresp and abs(dx) < 20 and abs(dy) < 20:
                out.append((y + blk // 2, x + blk // 2, dx * 10.0, dy * 10.0, r))
    return np.array(out).reshape(-1, 5)


def summ(a, sl):
    if len(a) == 0:
        return {"n": 0}
    d = np.hypot(a[:, 2], a[:, 3])
    r = {"n": int(len(a)), "median_m": round(float(np.median(d)), 1), "p90_m": round(float(np.percentile(d, 90)), 1),
         "median_dx_m(東+)": round(float(np.median(a[:, 2])), 1), "median_dy_m(南+)": round(float(np.median(a[:, 3])), 1), "median_resp": round(float(np.median(a[:, 4])), 3)}
    s = sl[a[:, 0].astype(int), a[:, 1].astype(int)]
    for lo, hi in ((0, 15), (15, 30), (30, 90)):
        m = (s >= lo) & (s < hi)
        if m.sum() >= 5:
            r[f"slope_{lo}-{hi}"] = {"n": int(m.sum()), "median_m": round(float(np.median(d[m])), 1), "dx": round(float(np.median(a[m, 2])), 1), "dy": round(float(np.median(a[m, 3])), 1)}
    return r


def main():
    tf, (H, W) = A.grid()
    sl = A.slope(tf, H, W)
    S1.ORBITS = {"desc105": None, "asc69": None}
    pd_, _ = S1.stack("desc105", "pre", "vv")
    qd, _ = S1.stack("desc105", "post", "vv")
    pa_, _ = S1.stack("asc69", "pre", "vv")
    qa, _ = S1.stack("asc69", "post", "vv")
    db = lambda x: 10 * np.log10(np.maximum(x, 1e-6))
    s2 = {d: cv2.cvtColor(np.load(A.OUT / f"rgb_{d}.npy")[..., ::-1].copy(), cv2.COLOR_BGR2GRAY).astype(np.float32) for d in ("20250615", "20251011", "20251016")}
    val = np.load(A.OUT / "valid_20251011.npy")
    res = {}
    tests = {
        "S1降軌事前 vs S1升軌事前（VV）": (db(pd_), db(pa_)),
        "S1降軌事前 vs S1降軌事後（VV）": (db(pd_), db(qd)),
        "S2 06-15 vs S2 10-11（灰階）": (s2["20250615"], s2["20251011"]),
        "S2 10-11 vs S2 10-16（灰階）": (s2["20251011"], s2["20251016"]),
        "S2 10-11（ref）vs S1降軌事後 VV（mov）": (s2["20251011"], db(qd)),
        "S2 10-11（ref）vs S1升軌事後 VV（mov）": (s2["20251011"], db(qa)),
        "S2 06-15（ref）vs S1降軌事前 VV（mov）": (s2["20250615"], db(pd_)),
        "S2 06-15（ref）vs S1升軌事前 VV（mov）": (s2["20250615"], db(pa_)),
    }
    for name, (r, m) in tests.items():
        a = shifts(r, m)
        res[name] = summ(a, sl)
        print(name, json.dumps(res[name], ensure_ascii=False), flush=True)
    (A.AREA / "s1_s2_registration.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
