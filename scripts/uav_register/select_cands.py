"""候選篩選：單應矩陣假設地面近似平面，所以要挑「近垂直俯視 + 地形平緩」的空拍。
用本地 20 m DTM 算每個候選周邊 600 m 的地形起伏，並下載縮圖偵測天空（傾斜照的特徵），
再依關鍵字偏好有人工構造物（橋/堤/護岸/道路/聚落）的場景——這些才有可對位的線狀特徵。
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2, numpy as np

sys.path.insert(0, "F:/GitHub/ardswc-resilience-observatory/scripts")
import dtm20

HERE = Path(__file__).parent
SRC = dtm20.Dtm20Source()
GOOD_KW = ("橋", "堤", "護岸", "道路", "農路", "聚落", "河道", "淹水", "野溪", "整治", "潛壩", "固床")
BAD_KW = ("全景", "遠景", "空拍全")


def relief(lat, lon, span_km=0.6):
    """周邊地形起伏（m）與平均高程；起伏小 → 平面假設較成立。"""
    z, _, _, info = SRC.fetch_grid(lat, lon, span_km=span_km, res_m=20)
    z = z[np.isfinite(z)]
    if z.size < 50:
        return None
    return float(np.percentile(z, 95) - np.percentile(z, 5)), float(z.mean())


def sky_score(img):
    """上緣 12% 區域「像天空」的比例：亮且低飽和或偏藍，且紋理低。含天空 → 極傾斜，不適合單應。"""
    h = img.shape[0]
    top = img[:int(h * 0.12)]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    bright = hsv[..., 2] > 150
    lowsat = hsv[..., 1] < 70
    blue = (hsv[..., 0] > 90) & (hsv[..., 0] < 130)
    lap = cv2.Laplacian(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), cv2.CV_32F)
    flat = np.abs(lap) < 6
    return float((bright & (lowsat | blue) & flat).mean())


def thumb(eid):
    url = f"https://photo.ardswc.gov.tw/api/Media/{eid}/facebook"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 ardswc-uav-select"})
        b = urllib.request.urlopen(req, timeout=40).read()
        return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def main():
    import os
    rows = json.load(open(HERE / os.environ.get("UAV_CANDS", "uav_cands.json"), encoding="utf-8"))
    # 地形先篩（本地 DTM，很快）：起伏 < 220 m
    keep = []
    for r in rows:
        try:
            rl = relief(r["Lat"], r["Lng"])
        except Exception:
            continue
        if rl is None:
            continue
        rel, mean_z = rl
        if rel > 300:
            continue
        r["_relief"] = round(rel, 1); r["_elev"] = round(mean_z, 1)
        keep.append(r)
    print("地形平緩候選", len(keep), "/", len(rows), flush=True)

    # 去重（2 km）並依關鍵字 + 起伏排序
    def score(r):
        d = (r.get("Description") or "") + (r.get("Note") or "")
        kw = sum(k in d for k in GOOD_KW) - 3 * sum(k in d for k in BAD_KW)
        return (-kw, r["_relief"])

    keep.sort(key=score)
    from collections import Counter
    cell, picked = Counter(), []
    PER_CELL = int(__import__("os").environ.get("UAV_PER_CELL", "1"))
    for r in keep:
        k = (round(r["Lat"] * 160), round(r["Lng"] * 160))     # ~0.7 km 格（2026 紀錄高度集中，可用 UAV_PER_CELL 允許同格多張）
        if cell[k] >= PER_CELL:
            continue
        cell[k] += 1; picked.append(r)
    print("去重後", len(picked), flush=True)

    # 天空偵測（需下載縮圖，平行）
    top = picked[:300]
    with ThreadPoolExecutor(8) as ex:
        imgs = list(ex.map(thumb, [r["EventID"] for r in top]))
    out = []
    for r, im in zip(top, imgs):
        if im is None:
            continue
        r["_sky"] = round(sky_score(im), 3)
        if r["_sky"] > 0.08:
            continue
        out.append(r)
    out.sort(key=lambda r: (r["_sky"], r["_relief"]))
    print("近垂直候選", len(out), flush=True)
    for r in out[:20]:
        print(f"{r['EventID']} {r['DisasterYear']} {r['County']}{r['Town']} "
              f"relief {r['_relief']:>5} sky {r['_sky']:.3f} | {(r.get('Description') or '')[:40]}")
    json.dump(out[:int(os.environ.get("UAV_N","150"))], open(HERE / os.environ.get("UAV_OUT", "batch_cands.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
