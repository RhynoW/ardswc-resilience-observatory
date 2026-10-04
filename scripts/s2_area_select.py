"""挑選事件前後的 Sentinel-2 日期：以範圍框內「實測」的雲量（低解析度縮圖）取代 STAC 場景雲量。
對候選日期（STAC 場景雲量 ≤ 80%）各取 z11（約 76 m/px）縮圖鑲嵌，算：無資料（黑）比例、亮雲比例（V≥150、飽和度 <0.25、局部紋理低——
河床砂石偏粗糙，用紋理與飽和度分開；仍可能高估）。輸出 areas/<name>/s2/select.json（依實測雲量排序）。
用法：python scripts/s2_area_select.py
"""
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp" / "change_detect_viewer"))
import sentinel_assist as S  # noqa: E402

NAME = "hualien_barrier_lake"
BBOX = (121.1456, 23.5494, 121.4456, 23.8494)
OUT = REPO / "data" / "biggis_interp" / "areas" / NAME / "s2"
MATRIX = 11
WORLD = 2 * math.pi * S.R_MERC
P = WORLD / (S.TILE_PX * 2 ** (MATRIX - 1))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def merc(lon, lat):
    return S.R_MERC * math.radians(lon), S.R_MERC * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def main():
    lon, lat = (BBOX[0] + BBOX[2]) / 2, (BBOX[1] + BBOX[3]) / 2
    x0, y0 = merc(BBOX[0], BBOX[3])
    x1, y1 = merc(BBOX[2], BBOX[1])
    px0, px1 = (x0 + WORLD / 2) / P, (x1 + WORLD / 2) / P
    py0, py1 = (WORLD / 2 - y0) / P, (WORLD / 2 - y1) / P
    c0, c1, r0, r1 = int(px0) // 512, int(px1) // 512, int(py0) // 512, int(py1) // 512
    tiles = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
    log("縮圖圖磚", len(tiles))
    scenes = {}
    for a, b in ((date(2025, 1, 1), date(2025, 7, 31)), (date(2025, 8, 1), date(2025, 12, 31))):
        for s in S.search_scenes(lat, lon, a, b, max_cloud=80.0, half_m=20000):
            scenes[s["date"]] = s["cloud"]
    log("候選日期", len(scenes))
    iid = S.instance_id()
    wm = np.zeros((512, 512), bool)
    wm[512 - S.WATERMARK_BOX_PX[1]:, :S.WATERMARK_BOX_PX[0]] = True
    rows = []

    def one(ymd):
        try:
            mos = np.zeros(((r1 - r0 + 1) * 512, (c1 - c0 + 1) * 512, 3), np.uint8)
            for r, c in tiles:
                t = S._fetch_tile(iid, ymd, "TRUE_COLOR", r, c, matrix=MATRIX)
                mos[(r - r0) * 512:(r - r0 + 1) * 512, (c - c0) * 512:(c - c0 + 1) * 512] = t
            crop = mos[int(py0) - r0 * 512:int(py1) - r0 * 512 + 1, int(px0) - c0 * 512:int(px1) - c0 * 512 + 1]
            return ymd, crop
        except Exception as e:  # noqa: BLE001
            return ymd, repr(e)[:80]
    with ThreadPoolExecutor(4) as ex:
        for ymd, crop in ex.map(one, sorted(scenes)):
            if isinstance(crop, str):
                log(ymd, "失敗", crop)
                continue
            f = crop.astype(np.float32)
            b_, g_, r_ = f[..., 0], f[..., 1], f[..., 2]
            v = f.max(axis=2)
            nodata = float((v < 8).mean())
            sat = (v - f.min(axis=2)) / (v + 1e-6)
            m1 = cv2.blur(v, (5, 5))
            sd = np.sqrt(np.maximum(cv2.blur(v * v, (5, 5)) - m1 * m1, 0))
            cloud = float(((v >= 150) & (sat < 0.25) & (sd < 12) & (v >= 8)).mean())
            rows.append({"date": ymd, "scene_cloud": scenes[ymd], "nodata_frac": round(nodata, 3), "bright_flat_frac": round(cloud, 3)})
            log(ymd, rows[-1])
    rows.sort(key=lambda r: (r["nodata_frac"] > 0.2, r["bright_flat_frac"]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "select.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    log("最低 8：", [(r["date"], r["bright_flat_frac"], r["nodata_frac"]) for r in rows[:8]])


if __name__ == "__main__":
    main()
