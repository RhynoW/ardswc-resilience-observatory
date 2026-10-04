"""反向流程第 2 步：把 Wayback 影像（2022-08-26、2023-11-02）在「兩期皆有影像」的可成對面積內抓成 z17 圖磚並快取。

圖磚取自「包含該圖磚中心的拍攝範圍所屬的 Wayback 版本」（同一拍攝日在不同區域可能來自不同版本，wayback_footprints.json 的 release；多個範圍包含時取較新版本），
只存原始 JPEG，偵測時再以拍攝範圍多邊形遮罩。
可成對面積＝兩期 metadata 範圍交集（0.001° 格網）膨脹 1 格。快取：data/biggis_interp/poc/wb_tiles/<date>/<x>_<y>.jpg（已被 .gitignore 的 poc/ 涵蓋）。
僅使用 Wayback（Esri World Imagery Wayback，免金鑰；條款見 ARCHITECTURE.md），不使用 Google Earth 內容。
用法：python scripts/reverse_fetch_tiles.py [--dates 2022-08-26 2023-11-02]
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import wayback_assist as W  # noqa: E402

POC = REPO / "data" / "biggis_interp" / "poc"
BBOX = (120.55, 22.95, 120.95, 23.35)
STEP, N, Z = 0.001, 400, 17
DATES = ["2022-08-26", "2023-11-02"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def raster(fps, date):
    m = np.zeros((N, N), np.uint8)
    for f in fps:
        if f["date"] == date:
            for r in f["rings"]:
                p = np.array([[(x - BBOX[0]) / STEP, (BBOX[3] - y) / STEP] for x, y in r])
                cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def tiles_in(mask):
    x0, y0 = W.lonlat_to_tile(BBOX[0], BBOX[3], Z)
    x1, y1 = W.lonlat_to_tile(BBOX[2], BBOX[1], Z)
    out = []
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            n = 2 ** Z
            lon = (tx + 0.5) / n * 360 - 180
            import math
            lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
            ix, iy = int((lon - BBOX[0]) / STEP), int((BBOX[3] - lat) / STEP)
            if 0 <= ix < N and 0 <= iy < N and mask[iy, ix]:
                out.append((tx, ty))
    return out


def tile_release(fps, date, tl):
    """每個圖磚 → 包含其中心點的該日拍攝範圍的 release（取最新版本）；沒有包含者回傳 None。"""
    import math
    n = 2 ** Z
    out = {}
    cand = [f for f in fps if f["date"] == date]
    cand.sort(key=lambda f: f["release_date"], reverse=True)
    polys = [(f["release"], [np.array(r, np.float32) for r in f["rings"]]) for f in cand]
    for tx, ty in tl:
        lon = (tx + 0.5) / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 0.5) / n))))
        for rel, rings in polys:
            if any(cv2.pointPolygonTest(r.reshape(-1, 1, 2), (lon, lat), False) >= 0 for r in rings):
                out[(tx, ty)] = rel
                break
    return out


def main():
    dates = DATES if "--dates" not in sys.argv else sys.argv[sys.argv.index("--dates") + 1:]
    fps = json.load(open(POC / "wayback_footprints.json", encoding="utf-8"))
    ms = [raster(fps, d) for d in dates]
    both = np.logical_and.reduce(ms)
    both = cv2.dilate(both.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    np.save(POC / "paired_mask_001deg.npy", both)
    tl = tiles_in(both)
    log("可成對面積", round(float(both.mean()) * 1815), "km²；z17 圖磚", len(tl), "個 ×", len(dates), "期")
    for d in dates:
        rels = tile_release(fps, d, tl)
        outd = POC / "wb_tiles" / d
        outd.mkdir(parents=True, exist_ok=True)
        todo = [t for t in tl if t in rels and not (outd / f"{t[0]}_{t[1]}.jpg").exists()]
        log(d, "版本分布", {k: list(rels.values()).count(k) for k in set(rels.values())}, "無範圍包含（膨脹邊緣）", len(tl) - len(rels), "待抓", len(todo))
        (outd / "releases.json").write_text(json.dumps({f"{t[0]}_{t[1]}": r for t, r in rels.items()}), encoding="utf-8")
        fail = []

        def one(t):
            for _ in range(3):
                try:
                    b = W._http_get(W.TILE_URL.format(release=rels[t], z=Z, row=t[1], col=t[0]), timeout=30)
                    (outd / f"{t[0]}_{t[1]}.jpg").write_bytes(b)
                    return
                except Exception:  # noqa: BLE001
                    time.sleep(1)
            fail.append(t)
        with ThreadPoolExecutor(16) as ex:
            for n, _ in enumerate(ex.map(one, todo), 1):
                if n % 1000 == 0:
                    log(d, n, "/", len(todo), "失敗", len(fail))
        log(d, "完成，失敗", len(fail))
        (outd / "failed.json").write_text(json.dumps(fail), encoding="utf-8")


if __name__ == "__main__":
    main()
